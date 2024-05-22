import transformers
import argparse
import json
import sys
from transformers import pipeline
from datasets import load_dataset, Dataset
from dataclasses import dataclass
from typing import Literal, get_args
from tqdm import tqdm
from enum import Enum
import torch
import re
import os
from collections import Counter
import random
RANDOM_SEED = 42
random.seed(RANDOM_SEED)

# TODO: find way to truncate the generated text if too long for the model
PromptFeature = Literal['zero-shot', 'few-shot', 'chain-of-thought', 'self-consistency', 'positive-feedback', 'negative-feedback']

@dataclass
class Fallacy:
    """A dataclass to hold the fally text and their corresponding labels."""
    fallacy_text: str
    labels: list[str]

    def __str__(self) -> str:
        text = self.fallacy_text.rstrip('\n')
        return f'Text: "{text}"\nLabel(s): "{self.labels}"'

    def get_base_prompt(self, fallacy_options: list[str], model) -> str:
        """Return the basic prompt"""

        context = f"""An argument consists of an assertion called the conclusion and one or more assertions called premises, where the premises are intended to establish the truth of the conclusion. Premises or conclusions can be implicit in an argument. A fallacious argument is an argument where the premises do not entail the conclusion."""
        question = f"""Which one of these {len(fallacy_options)} fallacious argument types does the following text contain?"""
        unsure = f"""Respond "Unsure about answer" if not sure about the answer."""
        informal = {"slippery slope": "This fallacy occurs when it is claimed that a small step will inevitably lead to a chain of events, resulting in a significant negative outcome.",
                    "ad hominem": "This fallacy involves attacking a person's character or motives instead of addressing the substance of their argument.",
                    "appeal to (false) authority": "This fallacy occurs when an argument relies on the opinion or endorsement of an authority figure who may not have relevant expertise or whose expertise is questionable. When applicable, a scientific consensus is not an appeal to authority.",
                    "X appeal to majority": "This fallacy is based on claiming a truth or affirming something is good because many people think so.",
                    "nothing": "No fallacious argument.",
        }

        # prompt = f'What logical fallacy is used here?\nThe options are: {fallacy_options}.\n Text: "\n{self.fallacy_text}"'

        if model == 'mosaicml/mpt-7b-instruct':
            # prompt for mpt-7b (https://huggingface.co/datasets/mosaicml/dolly_hhrlhf?ref=blog.paperspace.com&row=0 / https://blog.paperspace.com/large-language-models-fine-tuning-mpt-7b-on-paperspace/)
            fallacy_options_string = " ".join(str(i) + '. ' + j + ': ' + informal[j] for i, j in enumerate(fallacy_options, 1))
            
            [x for x in fallacy_options if x!= "nothing"]+["nothing (no fallacious argument)"]
            prompt = f"""Below is an instruction that describes a task. Write a response that appropriately completes the request. ### Instruction: {context} The potential fallacious argument types are {fallacy_options_string} {question} Respond with one of the {len(fallacy_options)} options without providing an explanation. {unsure} Text: "{self.fallacy_text}" ### Response:"""
            return prompt

        # In the wild prompt (training data paper)
        if False:
            prompt = f"""You task is to detect a fallacy in the Text Snippet. The label can be “Slippery Slope”, “Appeal to Authority”, “Ad Hominem”, “Appeal to Majority” or “None”.
Text Snippet: {self.fallacy_text}
Label:
"""

        # MFALDA prompt
        if False:
            prompt = f"""Definitions:
- An argument consists of an assertion called the conclusion and one or more assertions called premises, where the premises are intended to establish the truth of the conclusion. Premises or conclusions can be implicit in an argument.
- A fallacious argument is an argument where the premises do not entail the conclusion.
Text: "{self.fallacy_text}"
Based on the above text, determine whether the following sentence is part of a fallacious argument or not. If it is, indicate the type(s) of fallacy without providing explanations. The potential types of fallacy include:
- {str(chr(10)+"- ").join(fallacy_options)}
Sentence: "{self.fallacy_text}"
Output:
"""

        # adjusted MFALDA prompt
        prompt = f"""Answer the question based on the context below. {unsure}
Context: {context}
The potential fallacious argument types are:
- {str(chr(10)+"- ").join([x+': '+informal[x] for x in fallacy_options])}
Question: {question}
Text: "{self.fallacy_text}"
Answer:
"""
        return prompt
    
    def get_few_shot_prompt(self, fallacy_options: list[str], n_shot: int) -> str:
        """Return a few-shot prompt"""
        raise NotImplementedError

    def get_chain_of_thought_prompt(self, prompt: str) -> str:
        """Alter the prompt to include chain of thought"""
        # idea: include formal descriptions of the fallacies according to the mfalda paper? Force model to link parts to formal symbols?

        # prompt = f'{prompt}\nLet\'s think in steps'
        prompt = f'{prompt}\nPlease also explain your thinking steps.'

        return prompt

    # Note that the self-consistency prompt is the same as the chain of thought prompt
    # idea: for self-consistency, use all 3 models?
    
    def get_positive_feedback_prompt(self, prompt: str, sentiment: Literal['positive', 'negative']) -> str:
        """Alter the prompt to include positive or negative feedback feedback"""

        if sentiment == 'positive':
            prompt = f'Good job on that last prompt! Let\'s try another one. {prompt}'
        elif sentiment == 'negative':
            prompt = f'That last attempt was horrendous! Try this one instead. {prompt}'

        return prompt

    def build_prompt(self, fallacy_options: list[str], prompt_features: list[PromptFeature], model, n_shot: int = 0) -> str:
        """Build a prompt based on the prompt features provided"""
        prompt = self.get_base_prompt(fallacy_options, model)
        if 'zero-shot' in prompt_features:
            pass
        elif 'few-shot' in prompt_features:
            raise NotImplementedError     

        if 'chain-of-thought' in prompt_features:
            prompt = self.get_chain_of_thought_prompt(prompt)
        elif 'self-consistency' in prompt_features:
            prompt = self.get_chain_of_thought_prompt(prompt)
            raise NotImplementedError
        
        if 'positive-feedback' in prompt_features:
            prompt = self.get_positive_feedback_prompt(prompt, 'positive')
        elif 'negative-feedback' in prompt_features:
            prompt = self.get_positive_feedback_prompt(prompt, 'negative')

        return prompt 


def parse_args() -> argparse.Namespace:
    """Parse the command-line arguments"""

    parser = argparse.ArgumentParser(description="Prompt LLMs for fallacy detection")

    parser.add_argument(
        '-m',
        '--model', 
        type=str, 
        default='google/flan-t5-small',
        help='Model to prompt'
    )

    parser.add_argument(
        '-d',
        '--dataset',
        type=str,
        default='data/dev/fallacy_corpus.jsonl',
        help='Fallacy dataset to prompt for'
    )

    parser.add_argument(
        '-l',
        '--logpath',
        default='',
        type=str,
        help='Log file to write the output and results to'
    )

    parser.add_argument(
        '-p',
        '--prompt-features',
        type=str,
        choices=get_args(PromptFeature),
        nargs='+',
        default=['zero-shot'],
        help='Features to use when generating a prompt'
    ) 

    parser.add_argument(
        '-n',
        '--n-shot',
        type=int,
        default=0,
        help='Number of examples to show in the few-shot prompt'
    )

    parser.add_argument(
        '-s',
        '--samples',
        type=int,
        default=99999,
        help='(Maximum) number of samples to classify'
    )

    parser.add_argument(
        '-c',
        '--force-cpu',
        action=argparse.BooleanOptionalAction,
        help='Use cpu even when gpu is available'
    )

    parser.add_argument(
        '--temp',
        type=float,
        default=1.0,
        help='Temperature for the model'
    )

    parser.add_argument(
        '--do-sample',
        type=bool,
        default=False,
        help='Sample from tokens (instead of selecting greedily)'
    )

    parser.add_argument(
        '--top-k',
        type=int,
        default=50,
        help='Top-k for the model'
    )

    # parse args
    args = parser.parse_args()

    # perform some checks
    if 'few-shot' in args.prompt_features and not args.n_shot:
        parser.error('The number of shots must be specified when using the few-shot prompt type')
    elif 'few-shot' not in args.prompt_features and args.n_shot:
        parser.error('The number of shots can only be specified when using the few-shot prompt type')

    if 'zero-shot' in args.prompt_features and 'few-shot' in args.prompt_features:
        parser.error('zero-shot and few-shot prompt types cannot be used together')

    if 'chain-of-thought' in args.prompt_features and 'self-consistency' in args.prompt_features:
        parser.error('chain-of-thought and self-consistency prompt types cannot be used together')
        
    if 'positive-feedback' in args.prompt_features and 'negative-feedback' in args.prompt_features:
        parser.error('positive-feedback and negative-feedback prompt types cannot be used together')
        
    # if args.samples < 5:
    #     parser.error('The number of samples can not be smaller than the number of types of fallacies')

    return args


def load_dataset(dataset_path: str) -> list[Fallacy]:
    """Convert jsonl lines to a list of prompts"""

    fallacies = []

    with open(dataset_path, 'r') as inp:
        dataset = inp.readlines()

    for line in dataset:
        example = json.loads(line)
        fallacies.append(Fallacy(example['text'], [label[2] for label in example['labels']]))
    
    return fallacies


def prompt_model(pipe: pipeline, prompts: list[str], logpath: str, temp, top_k, do_sample) -> list[str]:
    """Get model output for all the prompts"""

    generated_texts = []
    for prompt in tqdm(prompts, desc='Prompting model', leave=False):
        generated_text = pipe(
            prompt,
            do_sample=do_sample,
            temperature=temp,
            top_k=top_k,
            max_new_tokens=25
        )
        generated_texts.append(generated_text[0]['generated_text'])

        # temp code
        # print generated text and the prompt
        # print(f'Prompt: "{prompt}"\nGenerated text: "{generated_text[0]["generated_text"]}"\n')
        # # print model answer to test the extract_model_answer function
        # print(f'Model answer: "{extract_model_answer(generated_text[0]["generated_text"], ["slippery slope","X appeal to majority", "ad hominem", "appeal to (false) authority", "nothing"])}"\n')

        if "### Response:" in generated_texts[-1]: generated_texts[-1] = generated_texts[-1].split("### Response:")[-1]
    return generated_texts


def evaluate_generated_texts(fallacies: list[Fallacy], generated_texts: list[str], logpath: str, fallacy_options) -> None:
    """Checks if the model prediction is correct, and prints the number of correct predictions."""
    
    correct = []
    
    if logpath:
        outp = open(logpath, 'w')
    else:
        outp = sys.stdout

    for fallacy, generated_text in zip(fallacies, generated_texts):
        # write to logfile
        outp.write(f'{fallacy}\nGenerated text: "{generated_text}"\n')

        # extract the model answer
        model_answer = extract_model_answer(generated_text, fallacy_options)

        # default answer is 'nothing'
        if not model_answer: model_answer = 'nothing'

        # Check if the extracted answer is one of the labels
        if model_answer and any(model_answer.lower() == label.lower() for label in fallacy.labels):
            correct.append(1)
            outp.write('-> correct\n\n')
        else:
            correct.append(0)
            outp.write('-> incorrect\n\n')
    
    if logpath:
        outp.close()

    print(f'Got {sum(correct)} out of {len(fallacies)} correct')

    if logpath:
        with open(logpath, 'a', encoding='utf-8') as outp:
            outp.write(f'Got {sum(correct)} out of {len(fallacies)} correct\n')

    return correct
        
def extract_model_answer(generated_text, fallacy_options):
    """Extract the answer from generated text based on specified patterns or by frequency of occurrence."""
    # Normalize the generated text for more reliable matching
    normalized_text = generated_text.lower()

    # Pattern to capture 'answer is' or 'answer is:'
    pattern = re.compile(r"answer is:?\s*([\w\s]+)", re.IGNORECASE)
    match = pattern.search(normalized_text)
    if match:
        answer = match.group(1).strip()
        # Check if the extracted answer is a close match to any of the fallacy options
        for option in fallacy_options:
            if answer in option.lower():
                return option  # Return the matching fallacy option as it is in the list
        # If no close match, consider the raw extracted answer (This part could be refined)
        return answer

    # If no explicit answer pattern is found, look for the most mentioned fallacy in the options
    # Count occurrences of each fallacy option in the normalized text
    fallacy_count = {option: normalized_text.count(option.lower()) for option in fallacy_options}

    # Find the fallacy with the highest count in the text, if any are mentioned
    if fallacy_count:
        most_common_fallacy = max(fallacy_count, key=fallacy_count.get)
        if fallacy_count[most_common_fallacy] > 0:
            return most_common_fallacy  # Return the most common fallacy that actually appears

    return None  # Return None if no fallacies are mentioned or no pattern is matched

def get_balanced_fallacies(fallacies, fallacy_options, n):
    balanced_fallacies = []
    counts = dict.fromkeys(fallacy_options,0)
    n_class = -(-n // len(fallacy_options))
    # c = Counter([fallacy.labels[0] for fallacy in fallacies])
    # for key, count in [(key, count) for key, count in c.items() if count < n_class]:
    # min_c = min(c, key=c.get)

    r = list(range(len(fallacies)))
    random.shuffle(r)
    for i in r:
        if counts[fallacies[i].labels[0]] < n_class:
            balanced_fallacies.append(fallacies[i])
            counts[fallacies[i].labels[0]] += 1
            n -= 1
        if n == 0: break
    random.shuffle(balanced_fallacies)
    return balanced_fallacies

# https://stackoverflow.com/questions/13852700/create-file-but-if-name-exists-add-number
def uniquify(path):
    filename, extension = os.path.splitext(path)
    counter = 1

    while os.path.exists(path):
        path = filename + " (" + str(counter) + ")" + extension
        counter += 1

    return path

def save_results(correct, prompt_frame, generated_texts, fallacies, args):
    results = {'correct': correct,
              'prompt_frame': prompt_frame,
              'generated_texts': generated_texts,
              'fallacy_texts': [f.fallacy_text for f in fallacies],
              'fallacy_labels': [f.labels for f in fallacies],
              'model': args.model,
              'prompt_features': args.prompt_features,
              'n': len(fallacies),
              'temperature': args.temp,
              'top-k': args.top_k,
              'max_new_tokens': 25,
              'random_seed': RANDOM_SEED,
              'do_sample': args.do_sample
              }
    
    filename = uniquify(args.model.split("/")[-1] + '-' + str(len(fallacies)) + '-' + '-'.join(args.prompt_features) + '.txt')
    
    with open(filename, 'w') as f:
        f.write(json.dumps(results)) 

def main() -> None:
    """A script to prompt seq2seq LLMs for fallacy detection"""
    args = parse_args()

    if not args.force_cpu and torch.cuda.is_available():
        device = 'cuda'
    else:
        device = 'cpu'

    fallacies = load_dataset(args.dataset)
    fallacy_options = set(fallacy for fallacies in [fallacy.labels for fallacy in fallacies] for fallacy in fallacies)

    fallacies = get_balanced_fallacies(fallacies, fallacy_options, args.samples)
    empty_fallacy = Fallacy(['FALLACY_TEXT'], ['FALLACY_LABEL'])
    prompt_frame = empty_fallacy.build_prompt(fallacy_options, args.prompt_features, args.model, args.n_shot)
    prompts = [fallacy.build_prompt(fallacy_options, args.prompt_features, args.model, args.n_shot) for fallacy in fallacies]

    # pipe = pipeline('text2text-generation', model=args.model, device=device)


    if args.model == 'mosaicml/mpt-7b-instruct':
        # Adjust the model configuration specifically for mosaicml/mpt-7b-instruct
        config = transformers.AutoConfig.from_pretrained(args.model, trust_remote_code=True)
        # config.attn_config['attn_impl'] = 'triton'
        config.init_device = device
        model = transformers.AutoModelForCausalLM.from_pretrained(args.model, config=config, torch_dtype=torch.bfloat16, trust_remote_code=True)
        tokenizer = transformers.AutoTokenizer.from_pretrained("EleutherAI/gpt-neox-20b")
        pipe = transformers.pipeline('text-generation', model=model, tokenizer=tokenizer, device=device)
    else:
        # Default pipeline setup for other models
        pipe = transformers.pipeline('text2text-generation', model=args.model, device=device, use_fast=True, torch_dtype=torch.bfloat16, repetition_penalty=1.1)

    generated_texts = prompt_model(pipe, prompts, args.logpath, args.temp, args.top_k, args.do_sample)

    if args.logpath:
        with open(args.logpath, 'w', encoding='utf-8') as outp:
            for fallacy, generated_text in zip(fallacies, generated_texts):
                outp.write(f'{fallacy}\nGenerated text: "{generated_text}"\n\n')

    correct = evaluate_generated_texts(fallacies, generated_texts, args.logpath, fallacy_options)
    save_results(correct, prompt_frame, generated_texts, fallacies, args)
    
    
if __name__ == '__main__':
    main()
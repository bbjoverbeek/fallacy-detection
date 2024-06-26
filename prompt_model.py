import transformers
import argparse
import json
import sys
from transformers import pipeline
from datasets import load_dataset, Dataset
from dataclasses import dataclass
from typing import Literal, get_args
from tqdm import tqdm
import torch
import re
import os
import random
import pandas as pd
from huggingface_hub import login

RANDOM_SEED = 42
random.seed(RANDOM_SEED)

PromptFeature = Literal['zero-shot', 'few-shot', 'chain-of-thought', 'self-consistency', 'positive-feedback', 'negative-feedback']

@dataclass
class Fallacy:
    """A dataclass to hold the fallicy text and corresponding labels."""
    fallacy_text: str
    labels: list[str]
    def __str__(self) -> str:
        text = self.fallacy_text.rstrip('\n')
        return f'Text: "{text}"\nLabel(s): "{self.labels}"'

    def get_base_prompt(self, fallacy_options: list[str]) -> str:
        """Return the base prompt"""

        # Create dynamic mapping based on the given fallacy options
        informal_prefixes = ['AA', 'BB', 'CC', 'DD', 'EE', 'FF', 'GG', 'HH', 'II', 'JJ', 'KK', 'LL', 'MM', 'NN', 'OO',
                             'PP', 'QQ', 'RR', 'SS', 'TT', 'UU', 'VV', 'WW', 'XX', 'YY', 'ZZ', 'AB', 'AC', 'AD']
        informal = {fallacy: f"{prefix} {fallacy}" for fallacy, prefix in zip(fallacy_options, informal_prefixes)}
        fallacy_options_strings = "\n".join([informal[fallacy] for fallacy in fallacy_options])

        question = f"Which one of these {len(fallacy_options)} fallacious argument types does the following text contain?"
        choose_from = f"Please choose an answer from {', '.join(informal_prefixes[:len(fallacy_options)])}."

        prompt = f"""### Instruction:
Below is an instruction that describes a task. Write a response that appropriately completes the request.
Definitions:
    - An argument consists of an assertion called the conclusion and one or more assertions called premises, where the premises are intended to establish the truth of the conclusion. Premises or conclusions can be implicit in an argument.
    - A fallacious argument is an argument where the premises do not entail the conclusion.
The potential fallacious argument types are:
{fallacy_options_strings}

{question}
Text: "{self.fallacy_text}"

{choose_from}
### Response:"""

        return prompt
    
    def get_few_shot_prompt(self,  prompt, fallacy_options: list[str], n_shot: int) -> str:
        """Return a few-shot prompt"""
        informal_prefixes = ['AA', 'BB', 'CC', 'DD', 'EE', 'FF', 'GG', 'HH', 'II', 'JJ', 'KK', 'LL', 'MM', 'NN', 'OO',
                             'PP', 'QQ', 'RR', 'SS', 'TT', 'UU', 'VV', 'WW', 'XX', 'YY', 'ZZ', 'AB', 'AC', 'AD']
        informal = {fallacy: f"{prefix} {fallacy}" for fallacy, prefix in zip(fallacy_options, informal_prefixes)}
        fallacy_options_strings = "\n".join([informal[fallacy] for fallacy in fallacy_options])

        question = f"Which one of these {len(fallacy_options)} fallacious argument types does the following text contain?"
        choose_from = f"Please choose an answer from {', '.join(informal_prefixes[:len(fallacy_options)])}."

        prompt = f"""### Instruction:
Below is an instruction that describes a task. Write a response that appropriately completes the request.
Definitions:
    - An argument consists of an assertion called the conclusion and one or more assertions called premises, where the premises are intended to establish the truth of the conclusion. Premises or conclusions can be implicit in an argument.
    - A fallacious argument is an argument where the premises do not entail the conclusion.
The potential fallacious argument types are:
{fallacy_options_strings}
"""
        # Add examples
        prompt+= '\nNow consider the following samples as exaples: \n'
        global dev_dataset
        for _ in range(n_shot):
            random_idx = random.randint(0, len(dev_dataset)-1)
            example = dev_dataset[random_idx]
            prompt += f"""{question}
            Text: {example.fallacy_text}
            ### Response: {example.labels}\n"""
        prompt += f"""{question}
Text: {self.fallacy_text}
{choose_from}
### Response:"""
        return prompt        
        
        

    def get_chain_of_thought_prompt(self, prompt: str) -> str:
        """Alter the prompt to include chain of thought"""
        prompt = prompt.replace(
            "### Response:",
            "\nBefore identifying the fallacy, explain your reasoning thoroughly. Your explanation should clarify why the specific fallacy applies to the given statement. This step is crucial!\nIf you do not explain your reasoning, you will not receive credit for this question.\n### Response:"
        )
        return prompt

    # Note that the self-consistency prompt is the same as the chain of thought prompt
    
    def get_positive_feedback_prompt(self, prompt: str, sentiment: Literal['positive', 'negative']) -> str:
        """Alter the prompt to include positive or negative feedback feedback"""

        if sentiment == 'positive':
            prompt = f'Your last response was great! Let\'s try another one. {prompt}'
        elif sentiment == 'negative':
            prompt = f'Your last response was awful! Try this one instead. {prompt}'

        return prompt

    def build_prompt(self, fallacy_options: list[str], prompt_features: list[PromptFeature], model, n_shot: int = 0) -> str:
        """Build a prompt based on the prompt features provided"""
        prompt = self.get_base_prompt(fallacy_options)
        if 'zero-shot' in prompt_features:
            pass
        elif 'few-shot' in prompt_features:
            prompt = self.get_few_shot_prompt(prompt, fallacy_options=fallacy_options, n_shot=n_shot)     

        if 'chain-of-thought' in prompt_features:
            prompt = self.get_chain_of_thought_prompt(prompt)
        elif 'self-consistency' in prompt_features:
            prompt = self.get_chain_of_thought_prompt(prompt)
            
        
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
        '--prompt_features',
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
        default=0.7,
        help='Temperature for the model'
    )

    parser.add_argument(
        '--do-sample',
        action=argparse.BooleanOptionalAction,
        help='Sample from tokens with temperature and top-k (instead of selecting greedily)'
    )

    parser.add_argument(
        '--top-k',
        type=int,
        default=50,
        help='Top-k for the model'
    )
    parser.add_argument(
        '-r',
        '--repeat',
        type=int,
        default=1,
        help ='Number of repetitions for self-consistency'
    )
    parser.add_argument(
        '-cl',
        '--classification_level',
        default= 0,
        type=int,
        help= 'Level of classification, 0: binary, 1: 4 classes, 2: more fine grained classes'
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
    if args.repeat > 1 and 'self-consistency' not in args.prompt_features:
        parser.error('Prompt can not be repeated for the other features than the self-consistency')
    if args.repeat==1 and 'self-consistency' in args.prompt_features: 
        parser.error('Number of repetition needs to be more than 1 for self-consistency')
    return args



def mapped_labels(labels):
    """Convert the labels (of test/val) into appropiate label for classififcation level =1 """

    # Define the categories using a dictionary
    fallacy_categories = {
        "ad hominem": "fallacy of credibility", "ad populum": "fallacy of credibility", "appeal to nature": "fallacy of credibility",
        "appeal to (false) authority": "fallacy of credibility", "guilt by association": "fallacy of credibility", 
        "appeal to tradition": "fallacy of credibility", "tu quoque": "fallacy of credibility", "fallacy of relevance": "fallacy of credibility",

        "hasty generalization": "fallacy of logic", "causal oversimplification": "fallacy of logic", "false dilemma": "fallacy of logic",
        "straw man": "fallacy of logic", "false causality": "fallacy of logic", "false analogy": "fallacy of logic", 
        "slippery slope": "fallacy of logic", "circular reasoning": "fallacy of logic", "equivocation": "fallacy of logic", 
        "fallacy of division": "fallacy of logic",

        "appeal to ridicule": "appeal to emotion", "appeal to fear": "appeal to emotion", "appeal to worse problems": "appeal to emotion",
        "appeal to anger": "appeal to emotion", "appeal to positive emotion": "appeal to emotion", "appeal to pity": "appeal to emotion"
    }

    # Replace labels using list comprehension and dictionary lookup
    return [fallacy_categories.get(label, "no fallacy") for label in labels]
    

def load_dataset(dataset_path: str, CL: int) -> list[Fallacy]:
    """Convert jsonl lines to a list of prompts 
    and also maps the labels of the texts based on the classification level"""

    fallacies = []

    with open(dataset_path, 'r') as inp:
        dataset = inp.readlines()
    
    for line in dataset:
        example = json.loads(line)
        fallacy = Fallacy(example['text'], [label[2] for label in example['labels']])
        if len(fallacy.labels)==0 or fallacy.labels==['nothing']:
            fallacy.labels=['no fallacy']
        fallacy.labels = [label.lower() for label in fallacy.labels] #converting the labels into lowercase
        
        if CL==0: #for binary classification
            fallacy.labels = list(set(['no fallacy' if element=='no fallacy' else "fallacy" for element in fallacy.labels]))
            
        elif CL==1: #for classification level 1 (no fallacy, FOC, FOL, FOE) 
            fallacy.labels = mapped_labels(fallacy.labels)
        else:
            pass
        fallacies.append(fallacy)
    
    return fallacies


def prompt_model(pipe: pipeline, prompts: list[str], logpath: str, temp, top_k, do_sample, repeat) -> list[str]:
    """Get model output for all the prompts"""

    all_generated_texts = []
    for prompt in tqdm(prompts, desc='Prompting model', leave=False):
        generated_texts = []
        for _ in range(repeat):
            generated_text = pipe(
                prompt,
                do_sample=do_sample,
                temperature=temp,
                top_k=top_k,
                max_new_tokens=800 if repeat==1 else 125
            )[0]['generated_text']
            if "### Response:" in generated_text: generated_text = generated_text.split("### Response:")[-1]
            generated_texts.append(generated_text)
        all_generated_texts.append(generated_texts)

    return all_generated_texts


def evaluate_generated_texts(fallacies: list[Fallacy], generated_texts: list[str], logpath: str, fallacy_options) -> None:
    """Returns extracted answers from model outputs and whether they were correct."""
    correct = []
    model_answers = []
    
    if logpath:
        outp = open(logpath, 'w')
    else:
        outp = sys.stdout

    for fallacy, texts in zip(fallacies, generated_texts):
        fallacy_correct = []
        fallacy_model_answers = []

        for generated_text in texts:
            # outp.write(f'{fallacy}\nGenerated text: "{generated_text}"\n')

            model_answer = extract_model_answer(generated_text, fallacy_options)
            fallacy_model_answers.append(model_answer)

            if model_answer and any(model_answer.lower() == label.lower() for label in fallacy.labels):
                fallacy_correct.append(1)
                # outp.write('-> correct\n\n')
            else:
                fallacy_correct.append(0)
                # outp.write('-> incorrect\n\n')

        correct.append(fallacy_correct)
        model_answers.append(fallacy_model_answers)
    
    if logpath:
        outp.close()

    print(f'Got {sum(sum(correct_list) for correct_list in correct)} out of {sum(len(correct_list) for correct_list in correct)} correct')

    if logpath:
        with open(logpath, 'a', encoding='utf-8') as outp:
            outp.write(f'Got {sum(sum(correct_list) for correct_list in correct)} out of {sum(len(correct_list) for correct_list in correct)} correct\n')

    return correct, model_answers

        
def extract_model_answer(generated_text, fallacy_options):
    """Extract the answer from generated text based on specified patterns or by frequency of occurrence."""

    # Check if "### Response:" is in the generated text and remove anything before that
    if "### Response:" in generated_text:
        response_text = generated_text.split("### Response:")[-1].strip()
    else:
        response_text = generated_text.strip()

    # Normalize the response text for more reliable matching
    normalized_text = response_text.lower()

    # Define the prefixes for fallacy options
    informal_prefixes = ['aa', 'bb', 'cc', 'dd', 'ee', 'ff', 'gg', 'hh', 'ii', 'jj', 'kk', 'll', 'mm', 'nn', 'oo', 'pp',
                         'qq', 'rr', 'ss', 'tt', 'uu', 'vv', 'ww', 'xx', 'yy', 'zz', 'ab', 'ac', 'ad']
    informal_map = {prefix: fallacy.lower() for prefix, fallacy in zip(informal_prefixes, fallacy_options)}

    # Pattern to capture prefixed answers
    pattern = re.compile(r"\b(aa|bb|cc|dd|ee|ff|gg|hh|ii|jj|kk|ll|mm|nn|oo|pp|qq|rr|ss|tt|uu|vv|ww|xx|yy|zz|ab|ac|ad)\b", re.IGNORECASE)
    match = pattern.search(normalized_text)
    if match:
        answer_prefix = match.group(1).lower()
        try: 
            return informal_map[answer_prefix]
        except KeyError:
            pass

    # If no prefixed answer is found, look for the most mentioned fallacy in the options
    fallacy_count = {option.lower(): normalized_text.count(option.lower()) for option in fallacy_options}

    # Find the fallacy with the highest count in the text, if any are mentioned
    if fallacy_count:
        most_common_fallacy = max(fallacy_count, key=fallacy_count.get)
        if fallacy_count[most_common_fallacy] > 0:
            return most_common_fallacy

    # If no clear answer is found, check for the last word in the fallacy options
    last_word_fallacy_count = {option.lower(): normalized_text.count(option.split()[-1].lower()) for option in
                               fallacy_options}

    # Find the fallacy with the highest count of the last word
    if last_word_fallacy_count:
        most_common_last_word_fallacy = max(last_word_fallacy_count, key=last_word_fallacy_count.get)
        if last_word_fallacy_count[most_common_last_word_fallacy] > 0:
            return most_common_last_word_fallacy

    return None  # Return None if no fallacies are mentioned or no pattern is matched


def get_balanced_fallacies(fallacies, fallacy_options, n):
    """Get a subset of samples from the dev set with an even number for each fallacy type."""
    balanced_fallacies = []
    counts = dict.fromkeys(fallacy_options,0)
    n_class = -(-n // len(fallacy_options))

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
    """Make sure not to overwrite file and instead change name."""
    filename, extension = os.path.splitext(path)
    counter = 1

    while os.path.exists(path):
        path = filename + " (" + str(counter) + ")" + extension
        counter += 1

    return path

def save_results(correct, model_answers, prompt_frame, prompts, generated_texts, fallacies, args):
    """Save results to file."""

    results = {'prompt_frame': prompt_frame,
              'model': args.model,
              'prompt_features': args.prompt_features,
              'n_samples': len(fallacies),
              'temperature': args.temp,
              'top-k': args.top_k,
              'max_new_tokens': 800,
              'random_seed': RANDOM_SEED,
              'do_sample': args.do_sample,
              'level': args.classification_level,
              'self-consistency repetitions': args.repeat,
              'n_shot': args.n_shot
              }
    
    results_details = {str(i+1):
                        {
                        'fallacy_text': fallacies[i].fallacy_text,
                        'true_labels': fallacies[i].labels,
                        'prompt': prompts[i],
                        'model_outputs': generated_texts[i],
                        'predicted_fallacies': model_answers[i],
                        'is_correct': correct[i]
                        }
                for i in range(len(fallacies))}
    results.update(results_details)

    os.makedirs("results", exist_ok=True)
    run_name = f"{args.model.split('/')[-1]}-{args.classification_level}-{'-'.join(args.prompt_features)}"
    if 'few-shot' in args.prompt_features: run_name += f"-n{args.n_shot}"
    if 'self-consistency' in args.prompt_features: run_name += f"-n{args.repeat}"
    run_name += ".json"

    filename = uniquify(os.path.join("results", run_name))
    with open(filename, 'w') as f:
        f.write(json.dumps(results))


def remove_duplicates(dev_data, test_data):
    """Remove the samples from the dev_dataset that also exist in test_dataset"""
    test_texts = set(fallacy.fallacy_text for fallacy in test_data)
    return [fallacy for fallacy in dev_data if fallacy.fallacy_text not in test_texts]


def main() -> None:
    """A script to prompt seq2seq LLMs for fallacy detection"""
    args = parse_args()

    if not args.force_cpu and torch.cuda.is_available():
        device = 'cuda'
    else:
        device = 'cpu'

    fallacies = load_dataset(args.dataset, args.classification_level)
    global dev_dataset
    dev_dataset = load_dataset(dataset_path='data/dev/fallacy_corpus.jsonl', CL=args.classification_level)

    if args.prompt_features=='few-shot':        
        dev_dataset = remove_duplicates(dev_data=dev_dataset, test_data=fallacies)
        dev_dataset = get_balanced_fallacies(dev_dataset)

    fallacy_options = set(pd.read_json('classification_level.jsonl', lines=True)['labels'][args.classification_level])

    if args.samples < 9999: # This is only true if --samples is given and therefore if we are using the dev set
        fallacies = get_balanced_fallacies(fallacies, fallacy_options, args.samples)
        
    empty_fallacy = Fallacy(['FALLACY_TEXT'], ['FALLACY_LABEL'])
    prompt_frame = empty_fallacy.build_prompt(fallacy_options, args.prompt_features, args.model, args.n_shot)
    prompts = [fallacy.build_prompt(fallacy_options, args.prompt_features, args.model, args.n_shot) for fallacy in fallacies]

    if args.model == 'mosaicml/mpt-7b-instruct':
        # Adjust the model configuration specifically for mosaicml/mpt-7b-instruct
        config = transformers.AutoConfig.from_pretrained(args.model, trust_remote_code=True)
        # config.attn_config['attn_impl'] = 'triton'
        config.init_device = device
        model = transformers.AutoModelForCausalLM.from_pretrained(args.model, config=config, torch_dtype=torch.bfloat16, trust_remote_code=True)
        tokenizer = transformers.AutoTokenizer.from_pretrained("EleutherAI/gpt-neox-20b")
        pipe = transformers.pipeline('text-generation', model=model, tokenizer=tokenizer, device=device)
    elif args.model == "meta-llama/Meta-Llama-3-8B-Instruct":
        # Adjust the model configuration specifically for meta-llama/Meta-Llama-3-8B-Instruct
        login(token="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")
        config = transformers.AutoConfig.from_pretrained(args.model, trust_remote_code=True)
        config.init_device = device
        tokenizer = transformers.AutoTokenizer.from_pretrained(args.model)
        model = transformers.AutoModelForCausalLM.from_pretrained(args.model, config=config, torch_dtype=torch.bfloat16,
                                                                  trust_remote_code=True)
        pipe = transformers.pipeline('text-generation', model=model, tokenizer=tokenizer, device=device)
    else:
        # Default pipeline setup for other models
        pipe = transformers.pipeline('text2text-generation', model=args.model, use_fast=True, torch_dtype=torch.bfloat16, repetition_penalty=1.1)

    generated_texts = prompt_model(pipe, prompts, args.logpath, args.temp, args.top_k, args.do_sample, args.repeat)

    if args.logpath:
        with open(args.logpath, 'w', encoding='utf-8') as outp:
            for fallacy, generated_text in zip(fallacies, generated_texts):
                outp.write(f'{fallacy}\nGenerated text: "{generated_text}"\n\n')

    correct, model_answers = evaluate_generated_texts(fallacies, generated_texts, args.logpath, fallacy_options)
    save_results(correct, model_answers, prompt_frame, prompts, generated_texts, fallacies, args)
    
if __name__ == '__main__':
    main()
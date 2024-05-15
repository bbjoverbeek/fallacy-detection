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
from collections import Counter

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

    def get_base_prompt(self, fallacy_options: list[str]) -> str:
        """Return the basic prompt"""

        prompt = f'What logical fallacy is used here?\nThe options are: {fallacy_options}.\n Text: "\n{self.fallacy_text}"'

        return prompt

    def get_few_shot_prompt(self, fallacy_options: list[str], n_shot: int) -> str:
        """Return a few-shot prompt"""
        raise NotImplementedError

    def get_chain_of_thought_prompt(self, prompt: str) -> str:
        """Alter the prompt to include chain of thought"""

        # prompt = f'{prompt}\nLet\'s think in steps'
        prompt = f'{prompt}\nPlease also explain your thinking steps.'

        return prompt

    # Note that the self-consistency prompt is the same as the chain of thought prompt
    
    def get_positive_feedback_prompt(self, prompt: str, sentiment: Literal['positive', 'negative']) -> str:
        """Alter the prompt to include positive or negative feedback feedback"""

        if sentiment == 'positive':
            prompt = f'Good job on that last prompt! Let\'s try another one. {prompt}'
        elif sentiment == 'negative':
            prompt = f'That last attempt was horrendous! Try this one instead. {prompt}'

        return prompt

    def build_prompt(self, fallacy_options: list[str], prompt_features: list[PromptFeature], n_shot: int = 0) -> str:
        """Build a prompt based on the prompt features provided"""
        prompt = self.get_base_prompt(fallacy_options)
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


def prompt_model(pipe: pipeline, prompts: list[str], logpath: str) -> list[str]:
    """Get model output for all the prompts"""

    generated_texts = []
    for prompt in tqdm(prompts, desc='Prompting model', leave=False):
        generated_text = pipe(
            prompt,
            do_sample=True,
            temperature=0.7,
            top_k=40,
            max_new_tokens=200
        )
        generated_texts.append(generated_text[0]['generated_text'])

        # temp code
        # print generated text and the prompt
        # print(f'Prompt: "{prompt}"\nGenerated text: "{generated_text[0]["generated_text"]}"\n')
        # # print model answer to test the extract_model_answer function
        # print(f'Model answer: "{extract_model_answer(generated_text[0]["generated_text"], ["slippery slope","X appeal to majority", "ad hominem", "appeal to (false) authority", "nothing"])}"\n')

    return generated_texts


def evaluate_generated_texts(fallacies: list[Fallacy], generated_texts: list[str], logpath: str, fallacy_options) -> None:
    """Checks if the model prediction is correct, and prints the number of correct predictions."""
    
    correct = 0
    
    if logpath:
        outp = open(logpath, 'w')
    else:
        outp = sys.stdout

    for fallacy, generated_text in zip(fallacies, generated_texts):
        # write to logfile
        outp.write(f'{fallacy}\nGenerated text: "{generated_text}"\n')

        # extract the model answer
        model_answer = extract_model_answer(generated_text, fallacy_options)

        # Check if the extracted answer is one of the labels
        if model_answer and any(model_answer.lower() == label.lower() for label in fallacy.labels):
            correct += 1
            outp.write('-> correct\n\n')
        else:
            outp.write('-> incorrect\n\n')
    
    if logpath:
        outp.close()

    print(f'Got {correct} out of {len(fallacies)} correct')

    if logpath:
        with open(logpath, 'a', encoding='utf-8') as outp:
            outp.write(f'Got {correct} out of {len(fallacies)} correct\n')
        
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


def main() -> None:
    """A script to prompt seq2seq LLMs for fallacy detection"""

    if torch.cuda.is_available():
        device = 'cuda'
    else:
        device = 'cpu'

    # device = 'cpu'

    args = parse_args()

    fallacies = load_dataset(args.dataset)
    fallacy_options = set(fallacy for fallacies in [fallacy.labels for fallacy in fallacies] for fallacy in fallacies)
    prompts = [fallacy.build_prompt(fallacy_options, args.prompt_features, args.n_shot) for fallacy in fallacies]

    pipe = pipeline('text2text-generation', model=args.model, device=device)
    generated_texts = prompt_model(pipe, prompts, args.logpath)

    if args.logpath:
        with open(args.logpath, 'w', encoding='utf-8') as outp:
            for fallacy, generated_text in zip(fallacies, generated_texts):
                outp.write(f'{fallacy}\nGenerated text: "{generated_text}"\n\n')

    evaluate_generated_texts(fallacies, generated_texts, args.logpath. fallacy_options)
    
if __name__ == '__main__':
    main()
import argparse
import json
import sys
from transformers import pipeline
from datasets import load_dataset, Dataset
from dataclasses import dataclass
from typing import Literal, get_args
from tqdm import tqdm
from enum import Enum

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

        prompt = f'What logical fallacy is used here?"\n{self.fallacy_text}"\nThe options are: {fallacy_options}.'

        return prompt

    def get_few_shot_prompt(self, fallacy_options: list[str], n_shot: int) -> str:
        """Return a few-shot prompt"""
        raise NotImplementedError

    def get_chain_of_thought_prompt(self, prompt: str) -> str:
        """Alter the prompt to include chain of thought"""

        prompt = f'{prompt}\nLet\'s think in steps'
        
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

    return generated_texts


def evaluate_generated_texts(fallacies: list[Fallacy], generated_texts: list[str], logpath: str) -> None:
    """Checks if the model prediction is correct, and prints the number of correct predictions."""
    
    correct = 0
    
    if logpath:
        outp = open(logpath, 'w')
    else:
        outp = sys.stdout

    for fallacy, generated_text in zip(fallacies, generated_texts):
        # write to logfile
        outp.write(f'{fallacy}\nGenerated text: "{generated_text}"\n')
        
        # TODO improve the comparison code below
        if any([label for label in fallacy.labels if label in generated_text.lower()]):
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
        

def main() -> None:
    """A script to prompt seq2seq LLMs for fallacy detection"""

    args = parse_args()

    fallacies = load_dataset(args.dataset)
    fallacy_options = set(fallacy for fallacies in [fallacy.labels for fallacy in fallacies] for fallacy in fallacies)
    prompts = [fallacy.build_prompt(fallacy_options, args.prompt_features, args.n_shot) for fallacy in fallacies]

    pipe = pipeline('text2text-generation', model=args.model)
    generated_texts = prompt_model(pipe, prompts, args.logpath)

    if args.logpath:
        with open(args.logpath, 'w', encoding='utf-8') as outp:
            for fallacy, generated_text in zip(fallacies, generated_texts):
                outp.write(f'{fallacy}\nGenerated text: "{generated_text}"\n\n')

    evaluate_generated_texts(fallacies, generated_texts, args.logpath)
    
if __name__ == '__main__':
    main()
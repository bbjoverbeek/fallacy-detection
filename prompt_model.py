import argparse
import json
from transformers import pipeline
from datasets import load_dataset, Dataset
from dataclasses import dataclass
from typing import Literal
from tqdm import tqdm

# TODO: find way to truncate the generated text if too long for the model

@dataclass
class Fallacy:
    """A dataclass to hold the fally text and their corresponding labels."""
    text: str
    labels: list[str]

    def __str__(self) -> str:
        text = self.text.rstrip('\n')
        return f'Text: "{text}"\nLabel(s): "{self.labels}"'

    def get_zero_shot_prompt(self, fallacy_options: list[str]) -> str:
        """Return the zero-shot prompt"""

        prompt = f'What logical fallacy is used here?"\n{self.text}"\nThe options are: {fallacy_options}.'

        return prompt

    def get_few_shot_prompt(self, fallacy_options: list[str], n_shot: int) -> str:
        """Return a few-shot prompt"""
        raise NotImplementedError

    def get_chain_of_thought_prompt(self, fallacy_options: list[str]) -> str:
        """Return a chain-of-thought prompt"""
        prompt = f'{self.get_zero_shot_prompt(fallacy_options)}\nLet\'s think in steps'

    # Note that the self-consistency prompt is the same as the chain of thought prompt
    
    def get_positive_feedback_prompt(self, fallacy_options: list[str], sentiment: Literal['positive', 'negative']) -> str:
        """Return a positive-feedback prompt"""

        if sentiment == 'positive':
            prompt = f'Good job on that last prompt! Let\'s try another one. {self.get_zero_shot_prompt(fallacy_options)}'
        elif sentiment == 'negative':
            prompt = f'That last attempt was horrendous! Try this one instead. {self.get_zero_shot_prompt(fallacy_options)}'

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

    return parser.parse_args()


def load_dataset(dataset_path: str) -> list[Fallacy]:
    """Convert jsonl lines to a list of prompts"""

    fallacies = []

    with open(dataset_path, 'r') as inp:
        dataset = inp.readlines()

    for line in dataset:
        example = json.loads(line)
        fallacies.append(Fallacy(example['text'], [label[2] for label in example['labels']]))
    
    return fallacies


def prompt_model(pipe: pipeline, fallacies: list[Fallacy], logpath: str) -> list[str]:
    """Prompt the model for all fallacies"""

    # get all possible labels to use in the prompt
    fallacy_options = set()
    for fallacy in fallacies:
        for label in fallacy.labels:
            fallacy_options.add(label)

    generated_texts = []
    for fallacy in tqdm(fallacies, desc='Prompting model', leave=False):
        generated_text = pipe(
            fallacy.get_zero_shot_prompt(fallacy_options), 
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

    with open(logpath, 'w', encoding='utf-8') as outp:
        for fallacy, generated_text in zip(fallacies, generated_texts):
            # write to logfile
            outp.write(f'{fallacy}\nGenerated text: "{generated_text}"\n')
            
            # TODO improve the comparison code below
            if any([label for label in fallacy.labels if label in generated_text.lower()]):
                correct += 1
                outp.write('-> correct\n\n')
            else:
                outp.write('-> incorrect\n\n')

    print(f'Got {correct} out of {len(fallacies)} correct')

    if logpath:
        with open(logpath, 'a', encoding='utf-8') as outp:
            outp.write(f'Got {correct} out of {len(fallacies)} correct\n')
        

def main() -> None:
    """A script to prompt seq2seq LLMs for fallacy detection"""

    args = parse_args()

    pipe = pipeline('text2text-generation', model=args.model)

    fallacies = load_dataset(args.dataset)

    generated_texts = prompt_model(pipe, fallacies, args.logpath)

    if args.logpath:
        with open(args.logpath, 'w', encoding='utf-8') as outp:
            for fallacy, generated_text in zip(fallacies, generated_texts):
                outp.write(f'{fallacy}\nGenerated text: "{generated_text}"\n\n')

    evaluate_generated_texts(fallacies, generated_texts, args.logpath)
    
if __name__ == '__main__':
    main()
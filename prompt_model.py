import argparse
from transformers import pipeline
from datasets import load_dataset, Dataset
from dataclasses import dataclass
from enum import Enum

PromptTechnique = Enum('PromptTechnique', ['zero-shot', 'few-shot', 'chain-of-thought', 'self-consistency', 'positive_feedback'])

@dataclass
class Prompt:
    """A dataclass to hold the prompts and their corresponding labels"""

def parse_args() -> argparse.Namespace:
    """Parse the command-line arguments"""

    parser = argparse.ArgumentParser(description="Prompt LLMs for fallacy detection")

    parser.add_argument('-m','--model', type=str, default='google/flan-t5-small', help='Model to prompt',)

    parser.add_argument('-d','--dataset', type=str, default='data/dev/fallacy_corpus.jsonl', help='Fallacy dataset to prompt for',)

    return parser.parse_args()

def load_dataset(dataset: str, prompt_technique: Prompt) -> Dataset:
    pass


def prompt_model(pipe: pipeline, dataset: Dataset) -> None:
    """Prompt the model for fallacy detection on the dataset"""
    dataset = ['Is the following a fallacy, and if yes, what fallacy is it?: "Understood, I would agree.  I\'m posting it because it\'s from the BBC so I expect it to be well sourced by them."']

    for example in dataset:
        result = pipe(example, do_sample=True, temperature=0.7, max_new_tokens=100)
        print(result)


def main() -> None:
    """A script to prompt seq2seq LLMs for fallacy detection"""

    args = parse_args()

    pipe = pipeline('text2text-generation', model='google/flan-t5-small')

    prompt_model(pipe, [])

    
if __name__ == '__main__':
    main()
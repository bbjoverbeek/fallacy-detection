"""Convert the fallacy_corpus.json to the format of the mafalda_gold_standard.jsonl format"""

import json
import argparse
import pandas as pd

FallacyCorpus = dict[str, list[list[str, str]]]

def parse_args() -> argparse.Namespace:
    """Parse the command-line arguments"""

    parser = argparse.ArgumentParser(description="Pre-process the fallacy corpus")

    parser.add_argument(
        '-i','--input', type=str, default='data/dev/fallacy_corpus.json', help='File to pre-process', )

    parser.add_argument('-o','--output', type=str, default='data/dev/fallacy_corpus.jsonl', help='Output filename',)
    
    return parser.parse_args()

def map_labels(label: str) -> str:
    """Map the fallacy_corpus labels (left) to the mafalda_gold_standard.jsonl format (right)
    An 'X' prefix means that the label does not occur in the mafalda_gold_standard.jsonl format"""

    switcher = {
        'Slipperyslope': 'slippery slope',
        'AppealtoAuthority': 'appeal to (false) authority',
        'AdHominem': 'ad hominem',
        'AppealtoMajority': 'X appeal to majority',
        'None': 'nothing',
    }
    
    return [[0, 1, switcher[label]]]

def change_format(fallacy_corpus: FallacyCorpus) -> pd.DataFrame:
    """Convert the fallacy_corpus.json to the format of the mafalda_gold_standard.jsonl format"""

    fallacy_corpus_df = pd.DataFrame()

    # get rid of the splits and create pd df
    for split, examples in fallacy_corpus.items():
        batch = pd.DataFrame(examples, columns=['text', 'labels'])
        fallacy_corpus_df = pd.concat([batch, fallacy_corpus_df])

    # map the labels to the mafalda_gold_standard.jsonl format (with fake spans)
    fallacy_corpus_df['labels'] = fallacy_corpus_df['labels'].apply(map_labels)

    return fallacy_corpus_df

def main() -> None:
    """Pre-process the fallacy corpus"""

    args = parse_args()

    # load json file
    with open (args.input, 'r') as inp:
        fallacy_corpus = json.load(inp)

    # convert the format
    fallacy_corpus_df = change_format(fallacy_corpus)
    
    # save the new format to a jsonl file
    fallacy_corpus_df.to_json(args.output, orient='records', lines=True)

if __name__ == '__main__':
    main()
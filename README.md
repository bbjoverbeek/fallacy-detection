# fallacy-detection

## Steps to recreate the experiment:

#### Set up the environment (in Python 3.10.12):

```bash
python3 -m venv env
source env/bin/activate
pip3 install -r requirements.txt
```

### Development setup:

#### Download the development file:

```bash
wget https://raw.githubusercontent.com/raruidol/ArgumentMining23-Fallacy/main/data/fallacy_corpus.json -O data/dev/fallacy_corpus.json
```

#### Preprocess the development data:

```bash
python3 preprocess.py --input data/dev/fallacy_corpus.json --output data/dev/fallacy_corpus.jsonl
```

#### Prompt the model:

_Run different prompts by altering the command-line arguments. See `python3 prompt_model.py --help` for options._

```bash
python3 prompt_model.py --input data/dev/fallacy_corpus.jsonl
```

### Test setup:

#### Download the test file:

```bash
wget https://raw.githubusercontent.com/ChadiHelwe/MAFALDA/main/datasets/gold_standard_dataset.jsonl -O data/test/mafalda_gold_standard_dataset.jsonl
```

#### Prompt the model:

_Run different prompts by altering the command-line arguments. see `python3 prompt_model.py --help for options`_

```bash
python3 prompt_model.py --input data/test/mafalda_gold_standard_dataset.jsonl
```

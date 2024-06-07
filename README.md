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

_Run different prompts by altering the command-line arguments. see `python3 prompt_model.py --help` for options_

```bash
python3 prompt_model.py --input data/test/mafalda_gold_standard_dataset.jsonl
```

### Reproducible tests:

To make the tests reproducible we have created [`run_experiment.sh`](./run_experiment.sh). This script will set up a folder containing all the necessary files to run the experiment. The script will also run the experiment and save the results in a logfile. The model and the prompt can be changed in the script.

### Running on Habrok:

Because the large models do not work on any laptop, we ran the experiments on Habrok. The [`run_habrok.sh`](./run_habrok.sh) script will run all the experiments for one model. Once logged in to a GPU node the script can be submitted using `sbatch run_habrok.sh`. Do not forget to uncomment the model you wish to run. Once runtime is granted, the script will load the Python module, install the dependencies in a venv, and run all 11 experiments for all 3 levels, resulting in 33 experiments total. 

Once the program is runnning, it is possible to keep an eye on it using the `watch -n 5 jobinfo 999999` command. If you enter the correct jobid, this command will refresh the jobinfo command every 5 seconds so you can stay up-to-date. 
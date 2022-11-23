# Deep Learning with Free-Text Rationale for Cryptic Crosswords

## Abtract

Cryptic Crosswords are puzzles characterised by the need to overcome extreme ambiguity in clues, to the point where even humans struggle to solve them. Thus they provide a ready-made environment for learning and evaluating the ability of Language Models to learn the nuances of natural language and perform complex disambiguation tasks. Owing to the recency of work concerning Cryptic Crosswords, no investigations into the effect of rationales on the answer accuracy have taken place, although they have had significant success on other NLP problems. We examine the use of a self-rationalising model, which simultaneously predicts the answer to a cryptic clue and an associated free-text rationale, on the answer accuracy. We use T5 as our base model and include Curriculum Learning in our training process. We find that there is additional predictive ability in the free-text rationales, but that our language models are unable to learn to produce the rationales with good-enough quality to exploit it. Thus, in certain cases, even though the overall accuracy remains low, the use of a self-rationalising model does lead to slight improvement.

[View the full paper](Research%20Paper.pdf)

## Structure

### Directories

- `cryptics/` contains the data, code and driver code for running experiments.
- `curriculum_models/` contains the results of training our curriculum models (which initialise other models). Note that the actual model is not present.
- `experiments/` contains the results of the various experiments run. Note that the trained models are not present.

### Scripts

- `compress_predictions.py` is a helper script to compress and decompress the files with predictions.
  - `python3 compress_predictions.py compress` will zip the predictions
  - `python3 compress_predictions.py uncompress` will unzip the predictions
- `convert_table_json.py` is a helper script to convert our original text-based method of displaying predictions to a more efficient json representation. Note that this is a relic.
  - `python3 convert_table_json.py`
- `copy_generated_predictions.py` is the script used to copy all the generated rational (for all our splits) to the data directory for use in future training or evaluation cycles.
  - `python3 copy_generated_predictions.py`
- `sort_predictions.py` will sort the content of all the `generated_predictions.json` is alphabetical order by clue.
  - `python3 sort_predictions.py sort` will perform the aforementioned sorting
  - `python3 sort_predictions.py` will compare two files of sorted predictions and print all the cases where the clues match. Note that the source code will have to be changed to change which two files are compared.

### Run Files

- `slurm.sh` is the slurm file used with sbatch to launch a job.
- `run.sh` is a run file to run sbatch with unique arguments (per job) and creates the destination folder for the experiment.

## Run Experiments

### Configuration Files

- the configuration files contain the parameters and values for a job.
- they are all found in `cryptics/configurations/`.
- the valid parameters to be set can be found by perusing the `cryptics/arguments.py` and looking at the `Seq2SeqTrainingArguments` documentation from Hugging Face.
- all the files which currently exist are the files used to perform our experiments.

### Run on the cluster

- give the `run.sh` file executable permissions.
- `./run.sh <partition> <experiment folder> <configuration file>`
  - the `<experiment folder>` starts one level nested into the `experiments/` directory.
  - the `<configuration file>` is relative to the `cryptics/` directory.

### Run locally (not recommended)

- `cd cryptics`
- `python3 run_cryptics.py <experiment folder> <configuration file>`
  - the same conditions on the `<experiment folder>` and `<configuration file>` as the previous section apply.

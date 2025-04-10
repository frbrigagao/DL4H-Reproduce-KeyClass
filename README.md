# KeyClass: Text Classification with Label-Descriptions Only

This repository is an attempt at reproducing the paper [Classifying Unstructured Clinical Notes via Automatic Weak Supervision](https://arxiv.org/pdf/2206.12088) for the CS598 - Deep Learning for Healthcare class at UIUC Spring 2025. 

The original code for the paper was obtained from the authors' Github repository: https://github.com/autonlab/KeyClass

# 1. Repository Structure

- `assets`: images used in this README file.
- `config_files`: `.yaml` configuration files used for each reproducibility test.
- `keyclass`: original KeyClass implementation with some adaptions.
- `keyclass_multilabel`: attempt at modifying KeyClass to handle MIMIC-3, a multiclass multi-label classification problem.
- `original_data`: original datasets for training and evaluation. 
    - A subfolder for each dataset `imdb`, `amazon`, `dbpedia`, and `agnews`.
- `pretrained_models`: pre-trained models provided by the researchers.
- `results`: Results for each training run (one folder for each run).
    - Contains subfolders `/embeddings`, `/metrics`, and `/models`.
- `logs`: Stores logs for each training run.
- `scripts`: helpful scripts to run the code. Adapted from the original source code.
- `pyproject.toml`: project configuration file used by `uv` python package manager. 
- `README.md`: this README file.

# 2. Requirements

## 2.1 Hardware Requirements

The project was tested on Linux machines running Debian/Ubuntu, with an RTX 4090 GPU, and at least 64GB of RAM.

## 2.2 Environment Setup

First, have **CUDA 12.4+**  installed on the system. A guideline is available on the [Nvidia website](https://developer.nvidia.com/cuda-12-4-0-download-archive). 

Install the `uv` python package manager, available at https://github.com/astral-sh/uv.

On Debian/Ubuntu, run this command to install the dependencies to compile the `slycot` package:
``` shell
sudo apt install gfortran liblapack-dev libopenblas-dev
```

After the previous command completes, **within the project folder**, run:

- `source .venv/bin/activate` to activate the project's virtual environment.
- `uv sync` to install the necessary python packages. 

*If you want to use another python environment (conda or pip), the necessary packages are listed in the `pyproject.toml` file.*

In addition, this project supports [Weights & Biases](https://wandb.ai/) for model training tracking. If you want to use this feature, please first create your account and authenticate with the API key using `uvx wandb login`.

## 2.3 Getting the Datasets, Original Models, and Results

Finally, to download the datasets, original pre-trained models, and results, run this command:
```shell
cd scripts # Must be in the scripts folder 
uv run get_data.py
``` 

The script will ask for confirmation before dowloading each file. **Atention**: this script will **not** provide the MIMIC-3 dataset.

## 2.4 MIMIC-3 Dataset Preprocessing

The MIMIC-3 dataset must be previously obtained by the user.

After obtaining it, please copy the `DIAGNOSES_ICD.csv` and `NOTEEVENTS.csv` files from the dataset to the `mimic_preprocessing/data` folder.

Run the following commands that will generate the final train and validation sets:
```shell 
uv run /mimic_preprocessing/createAdmissionNoteTable.py # This will create the train and validation .csv files.
uv run /mimic_preprocessing/generateFinalFiles.py # This will create the final .txt files with the train and validation sets.
```

# 3. Training

To train a single model with the default configuration parameters established in the original paper, choose one of the `.yaml`configuration files in `/config_files/` and run this command:
``` shell
cd scripts # Need to be in the /scripts folder 
uv run run_all.py --config ../config_files/config_imdb.yml --use_wandb 1 # if Weights & Biases is set up 
uv run run_all.py --config ../config_files/config_imdb.yml --use_wandb 0 # if Weights & Biases is NOT set up 
```

# 4. Ablations/Extensions

## 4.1 Experiment 1 - Testing different batch sizes and learning rates

To run the first experiment mentioned in the planning document, run this command:
```shell
cd scripts # Need to be in the /scripts folder 
uv run run_experiments_1.py --use_wandb --keep_configs # If you want Weights & Biases logging and to keep the temporary config files.
uv run run_experiments_1.py # Only local logs
```

**Attention**: This script might take several hours to complete due to the number of experiments.

## 4.2 Experiment 2 - Testing different number of labeling functions for the DBPedia dataset 



# 5. Evaluation

To evaluate the model on ??, run:
``` python 
uv run ???
```

# 6. Pre-trained Models

You can download pretrained models here:

# 7. Results

Here are the reproducibility results in comparison to the original paper's:


# 7. License and Contributing 

MIT License

Copyright (c) 2022 Carnegie Mellon University, Auton Lab.

Permission is hereby granted, free of charge, to any person obtaining a copy of this software and associated documentation files (the "Software"), to deal in the Software without restriction, including without limitation the rights to use, copy, modify, merge, publish, distribute, sublicense, and/or sell copies of the Software, and to permit persons to whom the Software is furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.

See [MIT LICENSE](https://github.com/autonlab/KeyClass/blob/main/LICENSE) in the original repository for details.
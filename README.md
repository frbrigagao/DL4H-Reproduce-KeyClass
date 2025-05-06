# Reproducibility Project - KeyClass: Text Classification with Label-Descriptions Only

This repository is an attempt at reproducing the paper [Classifying Unstructured Clinical Notes via Automatic Weak Supervision](https://arxiv.org/pdf/2206.12088) for the CS598 - Deep Learning for Healthcare at UIUC - Spring 2025. 

The original code for the paper was obtained from the authors' Github repository: https://github.com/autonlab/KeyClass

**This version includes modifications to attempt to support multi-label classification, for the MIMIC-III dataset, as described in the paper but not included in the original code repository.**

A video presentation of our results is available at [Illinois MediaSpace](https://mediaspace.illinois.edu/media/t/1_6cxc5zvw).

# 1. Repository Structure

## 1.1 Initial Folders and Files

| **Folder / File**                       | **Description**                                                                                                                |
|----------------------------------------|--------------------------------------------------------------------------------------------------------------------------------|
| `config_files/`                        | Contains `.yml` configuration files for each dataset.                                                                          |
|                                        | Includes:                                                                                                                      |
|                                        | - [`config_mimic_filtered_descriptions.yml`](/config_files/config_mimic_filtered_descriptions.yml)                             |
|                                        | - [`config_mimic_unfiltered_descriptions.yml`](/config_files/config_mimic_unfiltered_descriptions.yml)                         |
| `keyclass/`                            | Original KeyClass implementation for single-label classification.                                                              |
| `keyclass_multilabel/`                 | Modified KeyClass version supporting multi-label classification.                                                               |

---

### `mimic_preprocessing/`: MIMIC Dataset Generation

| **Component**                            | **Description**                                                                                                   |
|-----------------------------------------|-------------------------------------------------------------------------------------------------------------------|
| `00_generate_icd9_descriptors.py`       | Generates keyword descriptors for the 19 top-level ICD-9 categories. Results already included in MIMIC `.yml` configuration files at `/config_files`.     |
| `01_create_admission_note_table.R`      | Processes MIMIC CSVs (`NOTEEVENTS.csv` and `DIAGNOSES_ICD.csv`) into intermediate files. Adapted from [FasTag](https://github.com/rivas-lab/FasTag).     |
| `02_generate_mimic_train_test_files.py` | Generates final dataset files: `train.txt`, `test.txt`, `train_labels.txt`, `test_labels.txt`, `labels.txt`.     |
| `mimic_csv_files/`                      | Directory to place raw MIMIC files (`NOTEEVENTS.csv` and `DIAGNOSES_ICD.csv`).                                 |
| `intermediate_files/`                   | Stores intermediate files created by the R script.                                                                    |
| `output_mimic_files/`                   | Final MIMIC dataset output folder.                                                                                |
| 📊 [`mimic_data_visualization.ipynb`](/mimic_preprocessing/mimic_data_visualization.ipynb) | Notebook for visual exploration of the generated MIMIC dataset.                        |

---

### `scripts/`: Experiment Management and Training

| **Script**                     | **Purpose**                                                                                     |
|--------------------------------|--------------------------------------------------------------------------------------------------|
| `get_data.py`                  | Downloads benchmark datasets, pre-trained models, and example results.                          |
| `run_all.py`                   | Runs training/evaluation for original KeyClass (single-label).                                 |
| `run_all_multilabel.py`        | Runs training/evaluation for MIMIC multi-label variant.                                         |
| `run_experiments.py`           | Executes multiple experiments on benchmark datasets (optional W&B + Dropbox integration).       |
| `run_experiments_multilabel.py`| Same as above, but for the MIMIC dataset.                                                       |
| `dropbox_upload.py`            | Uploads experiment outputs to Dropbox. Requires credentials.                                    |

---

### Miscellaneous Files

| **File**                                   | **Description**                                                                               |
|--------------------------------------------|-----------------------------------------------------------------------------------------------|
| 📊 [`datasets_visualizations.ipynb`](/datasets_visualizations.ipynb) | Data exploration notebook for IMDb, Amazon, AGNews, and DBPedia.           |
| [`experiment_summary.csv`](/experiment_summary.csv)               | Consolidated CSV of experiment results.                                                       |
| 📊 [`experiment_summary.ipynb`](/experiment_summary.ipynb)           | Notebook for plotting and analyzing experiment outcomes.                                      |

## 1.2 Folders Generated By Scripts (Upon Execution)

### Folders Generated by Scripts (Upon Execution)

| **Folder**            | **Purpose**                                                                 | **Created By**                                                                                   | **Contents / Structure**                                             |
|------------------------|------------------------------------------------------------------------------|---------------------------------------------------------------------------------------------------|------------------------------------------------------------------------|
| `original_data`        | Stores the raw datasets.                                                     | `scripts/get_data.py`                                                                            | One subfolder per dataset: `imdb`, `amazon`, `dbpedia`, `agnews`, `mimic`. |
| `pretrained_models`    | Stores pre-trained models for benchmark datasets.                            | `scripts/get_data.py`                                                                            | Pre-trained model files provided by the researchers.                                               |
| `results`              | Stores results for each training run.                                        | `scripts/run_all.py`, `scripts/run_all_multilabel.py`                                            | One subfolder per run containing `/embeddings`, `/metrics`, `/models`. |
| `logs`                 | Stores `.log` files for each training run.                                   | `scripts/run_all.py`, `scripts/run_all_multilabel.py`                                            | One log file per run.                                                  |
| `results_csv`          | Stores CSV files with experiment details and final results.                  | `scripts/run_experiments.py`, `scripts/run_experiments_multilabel.py`                            | CSV files for individual or grouped experiments.                       |

## 1.3 Other Files

| **Name**           | **Description**                                                                   |
|--------------------|-----------------------------------------------------------------------------------|
| `pyproject.toml`   | Project configuration file used by the `uv` Python package manager.               |
| `README.md`        | This README file.                                                                 |
| `assets/`          | Folder containing images and visual assets used in the README.                    |

# 2. Requirements

## 2.1 Hardware Requirements

The project utilized multiple Linux machines depending on the dataset, as listed below.

| **Device Type** | **CPU Cores** (Phys/Log) | **GPU (NVIDIA)**      | **VRAM (GB)** | **RAM (GB)** | **# Exp.** |
|-----------------|---------------------------|------------------------|---------------|--------------|------------|
| CPU             | 32 / 64                   | -                      | -             | 503.7        | 3          |
| Cuda            | 24 / 32                   | RTX 4090               | 24.0          | 125.6        | 56         |
| Cuda            | 64 / 128                  | H100 80GB HBM3         | 79.6          | 503.4        | 1          |
| Cuda            | 128 / 255                 | RTX 4090               | 24.0          | 2003.8       | 26         |
| Cuda            | 192 / 192                 | H100 80GB HBM3         | 79.6          | 2267.5       | 6          |


Most experiments will be able to run on a 32-core CPU, an RTX 4090 GPU 24GB, and at least 128GB of RAM.

For some training runs with DBPedia, Amazon, and MIMIC, better GPUs or running directly on CPU will be required.

## 2.2 Environment Setup

First, have **CUDA 12.4+**  installed on the system. A guideline is available on the [Nvidia website](https://developer.nvidia.com/cuda-12-4-0-download-archive). 

Install the `uv` python package manager, using the instructions at https://github.com/astral-sh/uv#installation.
```shell
# Basic install command for Debian/Ubuntu
curl -LsSf https://astral.sh/uv/install.sh | sh

source $HOME/.local/bin/env
```

On Debian/Ubuntu, run this command to install the dependencies to compile the `slycot` package and to run R scripts (required for MIMIC dataset preprocessing):
``` shell
sudo apt install gfortran liblapack-dev libopenblas-dev
sudo apt install r-base r-base-dev
sudo apt install r-cran-stringr r-cran-data.table r-cran-dplyr r-cran-lubridate r-cran-caret r-cran-tibble
```
*(Note: If you encounter issues installing R packages via apt, you might need to install them within R itself using `install.packages(c("stringr", "data.table", ...))`)*

After the previous command completes, run the following in sequence:
```shell 
# Clone this repository 
git clone https://github.com/frbrigagao/DL4H-Reproduce-KeyClass.git 
# cd into folder
cd DL4H-Reproduce-KeyClass
# Install the necessary packages
uv sync 
# Activate virtual environment
source .venv/bin/activate
source $HOME/.local/bin/env
```

*If you want to use another python environment (conda or pip), the necessary packages are listed in the `pyproject.toml` file.*

In addition, this project supports [Weights & Biases](https://wandb.ai/) for model training tracking. If you want to use this feature, please first create your account and authenticate with the API key using `uvx wandb login`.

## 2.3 Getting the Benchmark Datasets, Original Models, and Results

To download the **benchmark datasets** (IMDb, Amazon, AGNews, and DBPedia), researchers' pre-trained models (for benchmarks), and results (for benchmarks), run this command:
```shell
cd scripts # Must be in the scripts folder 
uv run get_data.py
``` 

The script will ask for confirmation before dowloading each file. 

**Attention**: this script will **not** provide the MIMIC-3 dataset.

## 2.4 MIMIC-3 Dataset Preprocessing

KeyClass requires specific files from the MIMIC-III database to be preprocessed into a custom dataset before running experiments.

**Steps:**

1.**Obtain MIMIC-III**: Obtain the MIMIC-III Clinical Database (v1.4 or compatible) through the official process at [PhysioNET](https://mimic.physionet.org/).

2.**Place Raw Files**: Copy the `DIAGNOSES_ICD.csv` and `NOTEEVENTS.csv` files from the MIMIC-III dataset into the `mimic_preprocessing/mimic_csv_files/` directory.

3.**Generate ICD-9 Descriptors (optional)**: Our generated descriptors are already included in the `/config_files/config_mimic_unfiltered.yml` and `/config_files/config_mimic_filtered.yml`. 

However, if you want to manually generate them, run the following script:
```shell
cd mimic_preprocessing # Navigate to the preprocessing directory
# This will output the top 30 keywords per ICD-9 top-level category 
# and save the results to target_icd9_descriptors.txt
uv run 00_generate_icd9_descriptors.py --num_keywords_per_cat 30 --output_file target_icd9_descriptors.txt 
```
The script also accepts filtering common keywords that are common in over % percentage of the categories.

In the `config_mimic_filtered.yml` we used the top 30 keywords per category after removing common keywords that were shared in over 30% of the 19 categories. 

The list was generated with this command:
```shell
cd mimic_preprocessing # Navigate to the preprocessing directory
uv run 00_generate_icd9_descriptors.py --num_keywords_per_cat 30 --shared_keyword_threshold 30 --output_file target_icd9_descriptors_filtered.txt
```
4.**Run R Script**: Execute the R script to process the raw MIMIC-III CSV files (`DIAGNOSES_ICD.csv` and `NOTEEVENTS.csv`) and create intermediate files. 

The script filters discharge notes, extracts the associated ICD-9 codes, maps these codes to top-level categories, and splits the data into training and testing sets.
```shell
cd mimic_preprocessing # Navigate to the preprocessing directory
Rscript 01_create_admission_note_table.R
```
This will generate `icd9NotesDataTable_train.csv` and `icd9NotesDataTable_test.csv` in the `mimic_preprocessing/intermediate_files/` directory.

**Credit**: Adapted from the original script from [FasTag's](https://github.com/rivas-lab/FasTag/tree/master/src/textPreprocessing) Github repository [(Venkataraman et al. (2020)](https://doi.org/10.1371/journal.pone.0234647). Adapted for the current project.

5.**Data Analysis & Visualizations (optional)**: We provide an Jupyter notebook that explores the files generated on step 3. To start Jupyter/JupyterLab on the project folder, run the following command:
```shell
uv run --with jupyter jupyter lab
```
Open the web link on your browser and navigate to `/mimic_preprocessing/mimic_data_visualization.ipynb`.

6.**Run Python Script**: Execute the Python script to convert the intermediate files into the final format required by the KeyClass pipeline.
```shell
cd mimic_preprocessing # Navigate to the preprocessing directory
uv run 02_generate_mimic_train_test_files.py
```
This will create the following files in the `mimic_preprocessing/output_mimic_files/` directory:

| **Filename**        | **Description**                                                                 |
|---------------------|---------------------------------------------------------------------------------|
| `train.txt`         | Training clinical notes (one per line).                                         |
| `test.txt`          | Testing clinical notes (one per line).                                          |
| `train_labels.txt`  | Multi-hot encoded labels for training data (e.g., `"010...1"`).                 |
| `test_labels.txt`   | Multi-hot encoded labels for testing data.                                      |
| `labels.txt`        | Names of the 19 top-level ICD-9 categories.                                     |

7.**Copy Output Files**: Copy the output files from the `mimic_preprocessing/output_mimic_files/mimic` to `../original_data/mimic` folder using these commands.
```shell
# Creates the folder if it does not exist
mkdir -p ../original_data/mimic
# Copies generated files to the folder 
# Will ask for confirmation if overwriting
cp -ir output_mimic_files/mimic/* ../original_data/mimic/
```

# 3. Training and Evaluation

## 3.1 Configuration Files

The main parameters for each training run can be configured using the provided `.yml` files at `/config_files`.

We've provided the original configuration files for each dataset according to the hyperparameters used in the KeyClass paper.

However, the original KeyClass repository does not provide a configuration file for the MIMIC-derived dataset. 

As a result, experiments were performed using a custom file that retained the paper's reported hyperparameters and those common among the datasets. 

To create a custom configuration just clone one of the files and save it with a new name, changing the desired parameters.

## 3.2 Executing a Training Run

To train and evaluate a single model, first choose a `.yaml` configuration file (e.g., `config_imdb.yml` for IMDb or `config_mimic_unfiltered_descriptions.yml` for MIMIC-III after preprocessing).

### 3.2.1 Single-Label Datasets (IMDB, AGNews, Amazon, DBPedia)

``` shell
cd scripts # Need to be in the /scripts folder 

# Use run_all.py for single-label datasets
# Example for IMDB
# if Weights & Biases is set up
uv run run_all.py --config ../config_files/config_imdb.yml --use_wandb 1 
# if not using Weight & Biases
uv run run_all.py --config ../config_files/config_imdb.yml --use_wandb 0 
```

### 3.2.2 Multi-Label Dataset (MIMIC)

``` shell
# Example for MIMIC-III (ensure preprocessing is done first):
# if Weights & Biases is set up
uv run run_all_multilabel_.py --config ../config_files/config_mimic_unfiltered_descriptions.yml --use_wandb 1
# if not using Weight & Biases
uv run run_all_multilabel.py --config ../config_files/config_mimic_unfiltered_descriptions.yml --use_wandb 0
```

## 3.3 Training Output Files

The scripts will create a **unique experiment name** for the training run, that will be shown at the start.

| **Path**                                                       | **Description**                                                                                   |
|----------------------------------------------------------------|---------------------------------------------------------------------------------------------------|
| `/results/[dataset]/[experimentname]/data_embeddings`          | Embedded dataset files.                                                                          |
| `/results/[dataset]/[experimentname]/metrics/`                 | Metrics for the models.                                                                          |
| └── `train_label_model_with_ground_truth.txt`                  | Metrics for label model.                                                                         |
| └── `test_end_model_with_ground_truth.txt`                     | Metrics for downstream classifier model.                                                          |
| └── `test_end_model_with_ground_truth_self_trained.txt`        | Metrics for downstream self-trained model.                                                        |
| └── `test_end_model_with_ground_truth_category_specific_by_category.json` | Category-specific F1 scores (only for MIMIC).                                       |
| `/results/[dataset]/[experimentname]/models`                   | Generated models (downstream classifier and self-trained model).                                 |
| `/results/[dataset]/[experimentname]/predictions`              | Pickle files with the models' predictions.                                                       |

The log file for the training run will be saved at `/logs/[experimentname].log`.

# 4. Executing Multiple Training Runs with Different Hyperparameters

To facilitate **executing multiple training runs automatically** with **different hyperparameters**, we've developed the following scripts:

| **Script**                       | **Description**                                                                 |
|----------------------------------|---------------------------------------------------------------------------------|
| `run_experiments.py`            | Executes training runs for benchmark datasets (single-label classification).   |
| `run_experiments_multilabel.py` | Executes training runs for the MIMIC dataset (multi-label classification).     |

Modify the following variables at the start of each script:

| **Parameter**             | **Description**                                                                 |
|---------------------------|----------------------------------------------------------------------------------|
| `EXPERIMENT_CSV_PREFIX`   | Prefix of the CSV filename that will store the results.                         |
| `DATASETS`                | Array with the list of datasets to be tested.                                   |
| `LEARNING_RATE`           | Array with the end model learning rates to be tested per dataset.               |
| `BATCH_SIZE`              | Array with the end model batch sizes to be tested per dataset.                  |
| `LABEL_MODELS`            | Array with the label models to be tested per dataset.                           |
| `LABELING_FUNCTIONS`      | Array with the number of labeling functions to be tested per dataset.           |
| `NGRAM_RANGE`             | Array with the n-gram range to be tested (only for `run_experiment_multilabel.py`). |
| `DATASET_DETAIL`          | Array with the type of descriptors to use (only for `run_experiment_multilabel.py`). |

Additionally, you can pass the following optional command-line parameters:

### Optional Command-Line Flags

| **Flag**            | **Description**                                                                                             |
|---------------------|-------------------------------------------------------------------------------------------------------------|
| `--use_wandb`       | If you want to use Weights & Biases to log each experiment.                                                 |
| `--keep_configs`    | If you want to keep the temporary `.yml` configuration file generated for each experiment.                 |
|                     | These will be stored in the `/config_files` folder with the structure `temp_config_[experimentname].yml`.   |
| `--use_dropbox`     | If you want to upload the resulting files to Dropbox after each experiment.                                |
|                     | Please add your credentials in `dropbox_upload.py` if you want to use this feature.                         |

Example:
```shell
cd scripts # Need to be in the /scripts folder 

# Will execute experiments for the benchmark datasets
# Will logs runs and keep temporary .yml config files
# Will save resulting files to Dropbox
uv run run_experiments.py --use_wandb --keep_configs --use_dropbox

# Will execute experiments for the MIMIC dataset
# Will logs runs and keep temporary .yml config files
# Will save resulting files to Dropbox
uv run run_experiments_multilabel.py --use_wandb --keep_configs --use_dropbox
```

# 5. Evaluation

| **Dataset**         | **Metrics**                                         |
|--------------------------|-----------------------------------------------------|
| Benchmark Datasets       | Accuracy, Precision, Recall                         |
| MIMIC-derived Dataset    | Aggregate F1 Score, Precision, Recall               |

Model evaluation is performed **automatically at the end of each training run** in `run_all.py` or `run_all_multilabel.py`.

Results are saved in the corresponding `/results/[dataset]/[experiment_name]/metrics/` directory and logged to `.log` file (and Weights & Biases if enabled).

# 6. Pre-trained Models

## 6.1 Original KeyClass Pre-Trained Models

You can download the original KeyClass pretrained models **for the benchmark datasets** using the `scripts/get_data.py` script (see Section 2.3).

## 6.2 Our Pre-Trained Models (Reproducibility Study)

Files generated by each training run, including the models, are available at our [Dropbox folder](https://www.dropbox.com/scl/fo/7xpliie7wpnqc677t2cwa/AK4IGXkCmLn-1UTFUWHSHpQ?rlkey=sq7ytrn04k2p953gj29hi771g&st=g9jhwp98&dl=0).

# 7. Results (Reproducibility Study)

This reproducibility study performed a total of 139 experiments for the five datasets.

Files generated by each training run, including evaluation metrics, are available at our [Dropbox folder](https://www.dropbox.com/scl/fo/7xpliie7wpnqc677t2cwa/AK4IGXkCmLn-1UTFUWHSHpQ?rlkey=sq7ytrn04k2p953gj29hi771g&st=g9jhwp98&dl=0). 

Due to MIMIC's dataset restrictions, the generated embeddings were removed for this dataset.

We also provide a compilation of all experiments data and results, including evaluation metrics, in the [`experiment_summary.csv`](/experiment_summary.csv) file and this [Google Sheet](https://docs.google.com/spreadsheets/d/1eHx2QmqcPsGohQ6mkNt-vAasfkkUmOfar1KnJM2FNjc/edit?usp=sharing).

In addition, all training runs were logged and are available at [Weights & Biases](https://wandb.ai/fb8-university-of-illinois-urbana-champaign/dl4h-reproduce-keyclass/table).

## 7.1 Training Statistics

### 7.1.1 Datasets

| **Dataset** | **Classes** | **# Train** | **Train %** | **# Test** | **Test %** |
|-------------|-------------|-------------|--------------|------------|-------------|
| **AGNews**<br/><sub><i>News Topics</i></sub> | 4 | 120,000 | 94.04% | 7,600 | 5.96% |
| **DBPedia**<br/><sub><i>Wikipedia Categories</i></sub> | 14 | 560,000 | 88.89% | 70,000 | 11.11% |
| **IMDb**<br/><sub><i>Movie Reviews</i></sub> | 2 | 25,000 | 50.00% | 25,000 | 50.00% |
| **Amazon**<br/><sub><i>Amazon Reviews</i></sub> | 2 | 3,600,000 | 90.00% | 400,000 | 10.00% |
| **MIMIC-III**<br/><sub><i>Discharge Notes</i></sub> | 19 | 39,541 | 75.00% | 13,181 | 25.00% |


Dataset Statistics from Gao et al. (2022). Models were trained on the training set, but do not have access to labels. Unlike other datasets, MIMIC is a multi-label classification task where each clinical note is assigned to all relevant categories. All datasets, except MIMIC, are balanced.

### 7.1.2 Compute Statistics

**Hardware Configurations and Number of Completed Experiments per Configuration:**
| **Device Type** | **CPU Cores** (Phys/Log) | **GPU (NVIDIA)**      | **VRAM (GB)** | **RAM (GB)** | **# Exp.** |
|-----------------|---------------------------|------------------------|---------------|--------------|------------|
| CPU             | 32 / 64                   | -                      | -             | 503.7        | 3          |
| Cuda            | 24 / 32                   | RTX 4090               | 24.0          | 125.6        | 56         |
| Cuda            | 64 / 128                  | H100 80GB HBM3         | 79.6          | 503.4        | 1          |
| Cuda            | 128 / 255                 | RTX 4090               | 24.0          | 2003.8       | 26         |
| Cuda            | 192 / 192                 | H100 80GB HBM3         | 79.6          | 2267.5       | 6          |


**Compute Hours and Experiment Status per Dataset:**

| **Dataset** | **CPU Time** | **GPU Time** | **Finished** | **Failed** | **Interrupted** | **Total** |
|-------------|--------------|--------------|--------------|------------|------------------|-----------|
| AGNews      | 0h 0min      | 11h 38min    | 24           | 0          | 0                | 24        |
| Amazon      | 0h 0min      | 121h 12min   | 12           | 1          | 1                | 14        |
| DBPedia     | 9h 55min     | 145h 56min   | 22           | 5          | 0                | 27        |
| IMDb        | 0h 0min      | 3h 58min     | 12           | 0          | 3                | 15        |
| MIMIC       | 16h 51min    | 161h 35min   | 26           | 30         | 3                | 59        |
| **Total**   | **26h 46min**| **444h 19min**| **96**       | **36**     | **7**            | **139**   |


**Training Epochs (label model, downstream classifier and self-trained model) and Average Runtime:**

| **Dataset** | **Min Epochs** | **Mean Epochs** | **Max Epochs** | **Average Runtime per Epoch** |
|-------------|----------------|------------------|----------------|----------------------------|
| Amazon      | 120            | 144.8            | 257            | 5m 48s                     |
| AGNews      | 118            | 216.9            | 419            | 0m 08s                     |
| DBPedia     | 107            | 552.8            | 1519           | 0m 55s                     |
| IMDb        | 113            | 128.2            | 134            | 0m 09s                     |
| MIMIC       | 217            | 217.0            | 217            | 1m 43s                     |

## 7.2 Results - Benchmark Datasets (Single-Label)

![Results](/assets/01_benchmark_results.png)

The performance on the four benchmark datasets was evaluated based on *accuracy*, comparing the label, downstream classifier, and self-trained models against the values reported in Tables 5 and 6 of the paper.

**Discussion of Benchmark Results:**

- **AGNews & IMDb:**  
  Results for the self-trained model were close to the paper's Table 6 values (within `0.003` and `0.010` absolute difference, respectively), suggesting successful reproduction with the paper's default settings. Label model accuracies also closely matched. The downstream end model (without self-training) showed slightly lower accuracy for AGNews (`-0.012`) and slightly higher for IMDb (`+0.008`) compared to the paper.

- **Amazon:**  
  Label model performance (`0.727` for data programming, `0.719` for majority vote) was *higher* than reported results (`0.580`, `0.652`). Consequently, the downstream classifier model also performed better (`0.858` vs. `0.832` for data programming). However, the self-trained model accuracy (`0.918`) was slightly *lower* than the paper's Table 6 result (`0.941`), but close to results on Table 5 (`0.928`). The reasons for label model discrepancy might relate to subtle differences in keyword extraction.

- **DBPedia:**  
  This dataset showed the largest discrepancy. Using the paper's stated `15` labeling functions, the self-trained model accuracy (`0.772`) was substantially *lower* than the reported results (`0.951`, `0.940`). Label model accuracies were *identical*, and the downstream model was only slightly lower (`0.814` vs. `0.823`). As explored in our extensions/ablations, increasing the labeling functions to `30` improved the accuracy significantly (to `0.941`), suggesting possible use of more functions than stated. In fact, the `.yml` DBPedia configuration file ([available here](https://github.com/autonlab/KeyClass/blob/main/config_files/config_dbpedia.yml)) provided by the authors explicitly uses `37` labeling functions.

Overall, results were largely reproducible for AGNews and IMDb, with some discrepancies for Amazon and DBPedia.

## 7.3 Results - MIMIC-derived Dataset (Multi-Label)

<img src="assets/02_mimic_results.png" alt="Results" height="120"/>

Reproducing the MIMIC-derived dataset results required adapting the source code for multi-label classification and inferring missing hyperparameters. The table below compares the best results achieved in this study with the paper's Table 3.

**Discussion of MIMIC-III Results:**

Results indicate that the MIMIC-III multi-label classification task was **not successfully reproduced**.  
Our best F1 score with unfiltered class descriptions (`0.199`) is significantly *lower* than the paper's reported F1 score (`0.625`).

- **Recall:** Our models achieved much lower recall (`0.126` and `0.131`) compared to the paper's `0.896`.
- **Precision:** Our models achieved higher precision (`0.602` and `0.543`) compared to the paper's `0.507`.
- **F1 Score:** This resulted in a drastically lower F1 score.

**The discrepancies are likely due to a combination of factors:**

- **Lack of Original Multi-Label Code:**   The adaptations made might have been suboptimal or contained errors compared to the authors' implementation.

- **Missing Preprocessing Details:** The paper used TF-IDF to rank relevant parts of each discharge note but did not detail how it was applied nor provide the source code.   This reproduction used the full text of each note, introducing more noise in the keyword/key-phrase acquisition stage, which may have affected model performance.

- **Unspecified Hyperparameters:** Key hyperparameters like the number of labeling functions and n-gram range for the MIMIC-derived dataset were not provided. While different combinations were tested, the optimal combination used by the authors remains unknown.

- **Class Descriptions:** The exact keyword set might differ slightly, and the number of keywords per ICD-9 top-level class was *not* informed. Experiments with filtered keywords provided marginal improvement (`0.199` to `0.201`).

## 7.4 Results - Ablation 1 - Decreasing the Downstream Classfier End Model Learning Rate and Batch Size

Experiments were conducted varying the downstream model learning rate (LR) (`1e-3` vs. `1e-4`) and batch size (`128`, `64`, `32`) for the benchmark datasets, using both data programming (KeyClass's default) and majority vote for the label model.  

![Results](/assets/03_ablation1_results.png)

**General Trend:** Reducing the batch size often led to slightly better accuracy, especially for IMDb and AGNews datasets when using the default LR=`1e-3`. The default LR=`1e-3` generally outperformed `1e-4`, except for DBPedia at larger batch sizes (`128` and `64`).

**Best Configurations (Self-Trained Model Accuracy):**

**AGNews** 
  - Best accuracy (`0.874`) achieved with LR=`1e-3`, batch size `32`, and **data programming**, *surpassing* the paper's best reported accuracy.

**DBPedia** 
- Best accuracy (`0.931`) with LR=`1e-4`, batch size `64`, and *majority voting*, surpassing data programming (which peaked at `0.856` with LR=`1e-3` and batch size `32`). This approaches the paper’s reported result.

**IMDb** 
- Best accuracy (`0.922`) achieved with LR=`1e-3`, batch size `32`, and **data programming**. 
- *Majority voting* achieved the same score with LR=`1e-4` and batch size `32`. 
- Both configurations *surpassed* the paper's reported accuracy.

**Amazon**
- Best accuracy (`0.927`) with LR=`1e-3`, batch size `64`, and **data programming**, *within margin of error* of the paper’s Table 5 result (`0.928`).

**Conclusion**
- The paper's hyperparameters (LR=`1e-3`, batch size `128`) were reasonable. 
- However, smaller batch sizes (`32` or `64`) consistently yielded better accuracy across datasets.
- A lower learning rate (`1e-4`) did not consistently outperform the default setting.

## 7.5 - Results - Ablation 2 - Increasing Number of Labeling Functions for DBPedia Dataset

Experiments were run with an end model learning rate of `1e-3`, batch size of `128` (the paper's default), and a varying number of labeling functions (LFs): `15`, `30`, `50`, `100`, `250`, and `300`.

<img src="assets/04_ablation2_results.png" alt="Results" height="300"/>

**Data Programming**  
  - Accuracy initially improved significantly, peaking at `0.941 ± 0.001` with `30` LFs (up from `0.772` at `15` LFs), closely matching the paper's Table 5 score of `0.940 ± 0.001`.  
  - Beyond `30` LFs, performance slightly decreased and plateaued around `0.900–0.908`.  
  - Hence, the theoretical claim mentioned in the paper — that data programming benefits from more labeling functions — was only *partially confirmed* for the DBPedia dataset.

**Majority Vote**  
  - Accuracy improved with more labeling functions at first, peaking at **`0.912 ± 0.001`** with `30` LFs and `0.909 ± 0.001` with `15` LFs.  
  - Performance degraded slightly at higher LF counts (`50`, `100`, `250`, `300`).  
  - Majority voting performed better than data programming at `15` LFs (`0.909` vs. `0.772`), but worse than data programming at the peak of `30` LFs (`0.912` vs. `0.941`).

**Conclusion**  
  - Increasing the number of labeling functions from the paper’s stated `15` significantly improved the accuracy of the self-trained model, matching the paper's Table 5 result at `30` LFs using data programming.  
  - Further increases in LF count did *not* help and occasionally *harmed* model performance.

# 8. License and Contributing 

MIT License

Permission is hereby granted, free of charge, to any person obtaining a copy of this software and associated documentation files (the "Software"), to deal in the Software without restriction, including without limitation the rights to use, copy, modify, merge, publish, distribute, sublicense, and/or sell copies of the Software, and to permit persons to whom the Software is furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.

See [MIT LICENSE](https://github.com/autonlab/KeyClass/blob/main/LICENSE) in the original repository for details.

# 9. References

Gao, C.; Goswami, M.; Chen, J.; and Dubrawski, A. 2022. Classifying Unstructured Clinical Notes via Auto-
matic Weak Supervision. In Lipton, Z.; Ranganath, R.; Sendak, M.; Sjoding, M.; and Yeung, S., eds., *Proceedings of the 7th Machine Learning for Healthcare Conference*, volume 182 of *Proceedings of Machine Learning Research*, 673–690. PMLR.

Venkataraman, G. R.; Pineda, A. L.; Bear Don’t Walk IV, O. J.; Zehnder, A. M.; Ayyar, S.; Page, R. L.; Bustamante,C. D.; and Rivas, M. A. 2020. FasTag: Automatic text classification of unstructured medical narratives. PLoS one, 15(6):e0234647.

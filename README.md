# KeyClass: Text Classification with Label-Descriptions Only

This repository is an attempt at reproducing the paper [Classifying Unstructured Clinical Notes via Automatic Weak Supervision](https://arxiv.org/pdf/2206.12088) for the CS598 - Deep Learning for Healthcare class at UIUC Spring 2025. 

The original code for the paper was obtained from the authors' Github repository: https://github.com/autonlab/KeyClass

# Repository Structure

- `assets`: images used in this README file.
- `config_files`: contains the `.yaml` configuration files used for each reproducibility test.
- `keyclass`: the original KeyClass implementation with some adaptions (code comments and better messages).
- `keyclass_multilabel`: an attempt at modifying KeyClass to handle MIMIC-3, a multiclass multialabel classification problem.
- `models`: 
- `original_data`: contains the original datasets used for training. A subfolder for each dataset.
- `pretrained_models`: the pre-trained models provided by the researchers.
- `results`: 
- `scripts`: helpful scripts to run the provided code. Adapted from the original source code by the authors.
- `pyproject.toml`: the project configuration file used by the `uv` python package manager. 
- `README.md`: this README file.

# Requirements

The project was tested on Linux machines running Debian/Ubuntu, with an RTX 4090 GPU, and at least 64GB of RAM.

Make sure to have **CUDA 12.4** installed on the system. A guideline is available on the [Nvidia website](https://developer.nvidia.com/cuda-12-4-0-download-archive). 

Install the `uv` python package manager, available at https://github.com/astral-sh/uv.

In addition, the project uses [Weights & Biases](https://wandb.ai/) for model training tracking. Make sure to create your account and authenticate with the API key using `uvx wandb login`. 

On Debian/Ubuntu, run this command to install the necessary dependencies for compiling the `slycot` package:
``` python
sudo apt install gfortran liblapack-dev libopenblas-dev
```

After installing the above packages, on the project folder, run `uv sync` to install the necessary packages.

If you want to use another python environment (conda or pip), the necessary packages are listed in the `pyproject.toml` file.

# Training

To train the model(s) in the paper, run this command:

``` python 
uv run ???
```

# Evaluation

To evaluate the model on ??, run:
``` python 
uv run ???
```

# Pre-trained Models

You can download pretrained models here:

# Results

Here are the reproducibility results in comparison to the original paper's:


# License and Contributing 

MIT License

Copyright (c) 2022 Carnegie Mellon University, Auton Lab.

Permission is hereby granted, free of charge, to any person obtaining a copy of this software and associated documentation files (the "Software"), to deal in the Software without restriction, including without limitation the rights to use, copy, modify, merge, publish, distribute, sublicense, and/or sell copies of the Software, and to permit persons to whom the Software is furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.

See [MIT LICENSE](https://github.com/autonlab/KeyClass/blob/main/LICENSE) in the original repository for details.
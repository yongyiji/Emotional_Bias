# Emotional Scenario Generation Project

This repository is designed for **evaluating the impact of emotion in code-LLMs performance on code generation** and running experiments on different LLMs and datasets.

## Project Structure
```text
project-root/
.
├── script/ # Contains code for code generation for different LLMs and datasets
├── Scenario/ # Contains scripts to generate emotional scenario descriptions
├── run.slurm # Slurm example to run experiments
├── emotional.yml # Conda environment file for dependencies
└── README.md # Project documentation
```


## Installation

Install dependencies using the provided Conda environment:

```bash
conda env create -f emotional.yml
conda activate emotional
```
This will install all necessary packages to generate scenarios and run experiments.

##  Scenario Generation

All emotional scenario generation scripts are located in the Scenario/ folder. These scripts generate descriptive scenarios that can be added to prompts for downstream tasks.


## Running Experiments

Experiments can be run using the run.slurm script. You can modify the script to:

Replace the model you want to use

Replace the dataset you want to run the experiment on

Example command to submit a job:

```bash
sbatch run.slurm
```

Make sure to update the model and dataset parameters in run.slurm as needed.


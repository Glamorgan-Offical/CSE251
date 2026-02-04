# CSE 251A Assignment 1

This project implements a prototype selection strategies to reduce the size of the MNIST training dataset while maintaining classification performance using a 1-Nearest Neighbor (1-NN) classifier. The project compares a baseline **Random Selection** strategy, and **Full Training Set** strategy against a proposed **Cluster-based Selection** strategy (using PCA and K-Means).

## Project Structure

```text
PROJECT1/
├── data/                   # MNIST dataset files (.ubyte) and loader utility
│   └── loader.py           # Data loading script
├── selector/               # Selection strategies
│   ├── base.py             # Abstract base class for selectors
│   ├── baseline.py         # Full dataset selector (Upper bound)
│   ├── cluster.py          # PCA + K-Means selector (Proposed Method)
│   └── random.py           # Stratified random selector (Lower bound)
├── results/                # Experiment outputs
├── main.py                 # Entry point for running experiments
├── run.sh                  # Shell script to reproduce all reported results
└── requirements.txt        # Dependencies
```

## Usage

This project requires Python 3.10. It is recommended to run it on Linux System.

### Setup

Install the dependencies by executing following command:

```bash
pip install -r requirements.txt
```

### Reproduce All Results
Run the provided shell script to execute the full suite of experiments, including PCA sensitivity analysis, comparisons across different $M$ values, and error bar calculations.

```bash
chmod +x run.sh
./run.sh
```

### Run Individual Experiments
Run default configurations using the following command

```bash
# Full dataset 
python main.py --selector baseline

# Random Selecting Prototypes (Default M=1000)
python main.py --selector random

# Proposed methods (Default M=1000, PCA=100)
python main.py --selector cluster
```

> **Note:** When executing the baseline experiment, the console output may display a default prototype count (e.g., `M=1000`). This is merely a display artifact of the default argument values. In practice, the `BaselineSelector` ignores this parameter and strictly utilizes the **full training dataset** ($N=60,000$).

If you need to run specific experimental configurations, using the command line interface

```bash
# Run Cluster Method (M=1000 total, i.e., 100 per class, PCA=100, repeat 5 times)
python main.py --selector cluster --num_prototypes 100 --pca_components 100 --n_runs 5

# Run Random Method (M=5000 total, i.e., 500 per class)
python main.py --selector random --num_prototypes 500
```

**Arguments:**
- `--selector`: Selection strategy to use (`baseline`, `random`, `cluster`).
- `--num_prototypes`: Number of prototypes selected per class (total $M = 10 \times$ this value).
- `--pca_components`: Number of principal components used for dimensionality reduction (applicable only to `cluster`).
- `--n_runs`: Number of experimental repetitions used to compute confidence intervals.

### Results

Upon completion, detailed accuracy metrics and timing statistics are saved in the `results/` directory.


# ECM3401 Individual Project and Literature Review

## Requirements

- Python 3.13.5
- A copy of the [dataset](https://www.kaggle.com/competitions/smartphone-decimeter-2023/data), downloaded and extracted to a location of your choice
- The Python dependencies listed in `requirements.txt`

The dataset location you choose becomes the base path for the project. When running the scripts below, point `--base_path`/`-b` at the `train` folder inside that extracted dataset.

## Setup

1. Create and activate a virtual environment (recommended).
2. Install dependencies from `requirements.txt`.

Windows (powershell):

```bash
python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install -r requirements.txt
```

Windows (cmd):

```bash
python -m venv .venv
.\.venv\Scripts\activate.bat
python -m pip install -r requirements.txt
```

Linux:

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.txt
```

## Important Workflow Rule

Run preprocessing first. The training and evaluation scripts depend on files created by the preprocessing step.

## Command-Line Usage

All commands below are run from the repository root.

### 1. Preprocess the dataset

Run preprocessing before anything else:

```bash
python preprocess.py -b <path-to-extracted-dataset>/sdc2023/train
```

Replace `<path-to-extracted-dataset>` with the location where you downloaded and extracted the dataset.

### 2. Train a model

Train the single-epoch network:

```bash
python train_single_epoch.py -b <path-to-extracted-dataset>/sdc2023/train -o single_epoch_network.pt
```

Train the multi-epoch network:

```bash
python train_multi_epoch.py -b <path-to-extracted-dataset>/sdc2023/train -o multi_epoch_network.pt
```

### 3. Evaluate a model

Evaluate the single-epoch network:

```bash
python evaluate_single_epoch.py -b <path-to-extracted-dataset>/sdc2023/train -m single_epoch_network.pt -o single_epoch_evaluation
```

Evaluate the multi-epoch network:

```bash
python evaluate_multi_epoch.py -b <path-to-extracted-dataset>/sdc2023/train -m multi_epoch_network.pt -o multi_epoch_evaluation
```

Use the `--kf` flag to enable Kalman filter trajectory estimation, and use `-d <drive>` to evaluate a single drive if required.

## Outputs

Running the project creates model checkpoints, evaluation folders, and preprocessing artifacts.

## Notes

- The preprocessing step is required before training or evaluation because later scripts load artifacts created there.
- The exact list of command-line arguments for each script is documented in the script files themselves.

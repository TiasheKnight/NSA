# Network Intrusion Detection System (IDS)

A flow-based machine-learning IDS prototype using the **CSE-CIC-IDS2018** dataset. The project trains and compares classifiers, extracts flow features from PCAP files with CICFlowMeter, and predicts whether network flows are benign or malicious.

> **Current scope:** This is a batch/offline analysis pipeline. It does not yet provide continuous live monitoring, block traffic, or replace a production IDS.

## Project contents

- `train_binary_ids.py` — trains a binary XGBoost detector.
- `train_compare_models.py` — trains and compares KNN, Random Forest, and XGBoost pipelines.
- `predict.py` — runs predictions on CSV or Parquet flow-feature data. Run `python predict.py --help` to see the exact options supported by your checkout.
- `training.ipynb` and `CSE_CIC_IDS2018_Data_Preparation_Model_Training.ipynb` — data preparation and training notebooks.
- `workflow.md` and `workflow.png` — workflow documentation and diagram.
- `pcaps/` — local dataset files; they may not be included in Git.
- `output/` — model artifacts, feature lists, metrics, and prediction outputs.

## Requirements

- Linux/Kali Linux or another compatible environment
- Python and `venv`
- Git
- CICFlowMeter for converting raw PCAP captures into flow features
- CSE-CIC-IDS2018 CICFlowMeter flow-feature files for training

The project has been used in a Python virtual environment. If a dependency is incompatible with your installed Python version, use a supported Python version and create a fresh environment.

## Setup

### 1. Clone the repository

```bash
git clone https://github.com/TiasheKnight/NSA.git
cd NSA
```

### 2. Create and activate a virtual environment

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
```

### 3. Install Python dependencies

```bash
python -m pip install pandas numpy scikit-learn xgboost joblib pyarrow
```

These packages support data handling, preprocessing, model training/inference, model serialization, and Parquet input.

### 4. Prepare the dataset

Place the CSE-CIC-IDS2018 **flow-feature Parquet files** in `pcaps/`, or update the paths in the training scripts. The experiments used files covering these scenarios:

- Botnet — Friday, 2 March 2018
- Brute force — Wednesday, 14 February 2018
- DDoS — Tuesday, 20 February 2018
- DDoS — Wednesday, 21 February 2018
- DoS — Thursday, 15 February 2018
- DoS — Friday, 16 February 2018
- Infiltration — Wednesday, 28 February 2018
- Infiltration — Thursday, 1 March 2018
- Web attacks — Thursday, 22 February 2018
- Web attacks — Friday, 23 February 2018

These files contain extracted flow features; they are **not raw PCAP captures**. Raw captures must be processed with CICFlowMeter before model inference.

### 5. Train the models

Run from the repository root with the virtual environment activated:

```bash
python train_binary_ids.py
python train_compare_models.py
```

Review each script's configuration and input/output paths before running. The scripts write model artifacts, feature lists, metrics, and prediction outputs under `output/`. Training time and memory use depend on the data size.

### 6. Convert a PCAP capture into flow features

Install CICFlowMeter and confirm it is available:

```bash
cicflowmeter --help
```

Example command:

```bash
mkdir -p output/ids
cicflowmeter -f path/to/capture.pcap -c output/ids/capture_flows.csv
```

The output CSV contains extracted flows and their features. To test a capture, generate traffic while the capture is running; a short capture with little traffic may produce very few flows.

### 7. Run predictions

Check the arguments supported by the current prediction script:

```bash
python predict.py --help
```

The script accepts a model, feature list, input CSV/Parquet file, and output path. Use the exact model/scaler combination expected by the script. The comparison artifacts such as `xgboost_pipeline.joblib` are complete preprocessing pipelines; do not apply a separate scaler to a complete pipeline. If `predict.py` expects a standalone model and scaler, use matching standalone artifacts instead.

Example command structure (adjust to the options and artifacts available in your checkout):

```bash
python predict.py \
  --model PATH_TO_MODEL \
  --features-file PATH_TO_FEATURES_TXT \
  --input output/ids/capture_flows.csv \
  --output output/ids/predictions.csv
```

If the script requires `--scaler`, supply the scaler trained for that specific standalone model. Never combine a model and scaler from different experiments.

## Results

Two separate binary-classification experiments were run. They used different sampling/training setups and **their scores should not be compared directly**.

### Experiment A — KNN, Random Forest, and XGBoost

This comparison used a random stratified train/test split of 54,882 sampled flows. The following are the recorded results with tuned classification thresholds.

| Model | Threshold | Accuracy | Balanced accuracy | Precision | Recall | F1 | False-positive rate |
|---|---:|---:|---:|---:|---:|---:|---:|
| KNN | 0.7491 | 0.8799 | 0.8720 | 0.9372 | 0.7878 | 0.8561 | 0.0438 |
| Random Forest | 0.5211 | 0.8839 | 0.8771 | 0.9303 | 0.8042 | 0.8626 | 0.0500 |
| XGBoost | 0.3904 | 0.8868 | 0.8833 | 0.8984 | 0.8460 | 0.8714 | 0.0793 |

Within this experiment, XGBoost had the highest balanced accuracy, recall, and F1 at the selected thresholds. KNN had the lowest false-positive rate, while Random Forest had slightly higher accuracy than KNN. Threshold selection trades off missed attacks against false alarms.

For reference, XGBoost at its default threshold scored accuracy **0.8897**, balanced accuracy **0.8823**, precision **0.9457**, recall **0.8028**, F1 **0.8684**, and false-positive rate **0.0382**.

### Experiment B — Binary XGBoost

A separate experiment sampled up to 100,000 benign and 100,000 attack flows per source file. Its recorded results were:

| Threshold | Accuracy | Balanced accuracy | Attack precision | Attack recall | Attack F1 | False-positive rate |
|---|---:|---:|---:|---:|---:|---:|
| Default | 0.9318 | 0.9157 | 0.9918 | 0.8360 | 0.9073 | 0.0046 |
| Tuned (0.4237) | 0.9314 | 0.9169 | 0.9803 | 0.8452 | 0.9077 | 0.0113 |

At the tuned threshold, balanced accuracy, attack recall, and F1 improved slightly, while precision decreased and the false-positive rate increased. This is a different experiment from Experiment A, so its scores are reported separately rather than used as a direct model-to-model comparison.

### Evaluation limitations

- The reported evaluations use random stratified splits. Flows from the same source capture or day can share patterns, so random splitting may overestimate performance on genuinely unseen traffic.
- A stronger next step is to train on selected days/files and evaluate on a held-out day or attack scenario.
- These are **binary benign-versus-attack** results; they do not establish reliable classification of every individual attack type.
- Accuracy alone is not sufficient for an IDS. Balanced accuracy, attack recall, precision, F1, and false-positive rate also matter.
- Test the pipeline on independent lab captures before drawing conclusions about real-world performance.

## Pipeline overview

```text
Raw PCAP capture
      |
      v
CICFlowMeter feature extraction
      |
      v
Flow-feature CSV
      |
      v
Feature alignment and preprocessing
      |
      v
Trained classifier
      |
      v
Benign / attack predictions
```

The current workflow performs batch inference on extracted flows. Continuous capture, live alerting, traffic blocking, and production IDS integration are not implemented by this workflow unless added separately.

## Large files and Git

Model files and prediction CSVs can be large, and may be excluded by `.gitignore`. If an expected model or dataset is missing after cloning, recreate it by running the training scripts or retrieve it from the project's artifact storage. GitHub recommends keeping files below 50 MB and blocks individual files over 100 MiB in regular Git storage. Consider Git LFS or external artifact storage for model files that must be shared.

## Team

Group 6:

- Jasper de Vries (`s6095321`)
- Tiashe Wu (`s6505465`)
- Fabian Cuza (`s4655818`)

## Disclaimer

This is an academic prototype for network traffic analysis and intrusion-detection experimentation. Model predictions can be wrong and should not be the sole basis for security decisions.

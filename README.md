# When Outliers Matter
### Investigating Gradient-Based Data Selection Under Spurious Correlations

**UCLA COM SCI 260D — Large-Scale Machine Learning | Fall 2026**

## 1. Project Overview

Deep learning models can achieve high overall accuracy while performing poorly on underrepresented groups because of **spurious correlations**—features that correlate with the target label but are not reliable predictors.

Our project investigates whether gradient-based data filtering removes informative minority examples and how different robustness methods affect worst-group generalization.

We compare six methods:

| Method | Description | Requires Group Labels During Training? |
|---|---|---|
| ERM | Standard Empirical Risk Minimization | No |
| EPIC-inspired | Gradient-space isolation and filtering | No |
| JTT | Upweight initially misclassified examples | No |
| Random Filtering | Randomly remove examples at matched rates | No |
| Group DRO | Optimize performance on high-loss groups | Yes |
| DFR | Retrain the classifier using group-balanced data | Yes |

**Research hypothesis:** Gradient-based isolation filtering may disproportionately remove rare but informative training examples, potentially harming worst-group accuracy even when overall accuracy remains high.

We use a shared dataset, feature extractor, and evaluation protocol to compare these methods.

---

## 2. Repository Setup

### Step 1 — Clone the repository

Open a terminal and run:

```bash
git clone https://github.com/alexljt888/When-Outliers-Matter-Investigating-Gradient-Based-Data-Selection-Under-Spurious-Correlations.git

cd When-Outliers-Matter-Investigating-Gradient-Based-Data-Selection-Under-Spurious-Correlations
```

Open the folder in VS Code.

### Step 2 — Create a Python environment

Python 3.12 is recommended.

**macOS / Linux:**

```bash
python3 -m venv .venv
source .venv/bin/activate
```

**Windows PowerShell:**

```powershell
py -3.12 -m venv .venv
.venv\Scripts\Activate.ps1
```

Install dependencies:

```bash
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
```

The project requires PyTorch, Torchvision, NumPy, pandas, Matplotlib, Pillow, and Jupyter/IPython kernel support.

If a dependency is missing, install it in the active environment and update `requirements.txt`.

### Step 3 — Configure Jupyter

In VS Code:

1. Install the Python and Jupyter extensions if necessary.
2. Open a project `.ipynb` file.
3. Click **Select Kernel** in the upper-right corner.
4. Select the Python interpreter inside `.venv`.

Run notebooks from the repository root so relative paths such as `data/celeba/` resolve correctly.

---

## 3. Download the CelebA Dataset

### Dataset Description

We use **CelebA (CelebFaces Attributes)**, containing 202,599 aligned facial images with 40 binary attributes.

- **Target label:** `Blond_Hair`
- **Spurious attribute:** `Male`
- **Training:** 162,770 images
- **Validation:** 19,867 images
- **Test:** 19,962 images

We preserve the official dataset partitions.

### Step 1 — Download the dataset files

Dataset source: [CelebA on Hugging Face](https://huggingface.co/datasets/Yuehao/celeba)

From the repository root, run:

```bash
mkdir -p data/celeba

BASE="https://huggingface.co/datasets/Yuehao/celeba/resolve/main"

curl -L --fail --retry 3 \
  "$BASE/img_align_celeba.zip" \
  -o data/celeba/img_align_celeba.zip

curl -L --fail --retry 3 \
  "$BASE/list_attr_celeba.txt" \
  -o data/celeba/list_attr_celeba.txt

curl -L --fail --retry 3 \
  "$BASE/list_eval_partition.txt" \
  -o data/celeba/list_eval_partition.txt
```

The image archive is approximately 1.44 GB compressed. Allow additional disk space for extraction and generated features.

### Step 2 — Verify and extract

```bash
unzip -tq data/celeba/img_align_celeba.zip

unzip -q data/celeba/img_align_celeba.zip \
  -d data/celeba/
```

Count the extracted images:

```bash
find data/celeba/img_align_celeba \
  -type f -name "*.jpg" | wc -l
```

Expected output:

```text
202599
```

After successful extraction and verification, the ZIP may be deleted to free storage:

```bash
rm data/celeba/img_align_celeba.zip
```

The extracted dataset should contain:

```text
data/
└── celeba/
    ├── img_align_celeba/
    │   ├── 000001.jpg
    │   ├── 000002.jpg
    │   └── ...
    ├── list_attr_celeba.txt
    └── list_eval_partition.txt
```

**Important:** Do not commit the downloaded dataset to GitHub.

---

## 4. Dataset Preparation

### Step 1 — Run `explore_celeba.ipynb`

Execute the notebook from top to bottom.

It performs:

- Attribute and partition loading
- Conversion of attribute labels from `{-1, 1}` to `{0, 1}`
- Definition of the four attribute groups
- Dataset distribution analysis
- Visualization of representative images
- Export of processed metadata

The notebook generates:

```text
data/processed/celeba_metadata.csv
```

This file contains:

| Column | Description |
|---|---|
| `filename` | Original image filename |
| `label` | Blond_Hair, encoded 0 or 1 |
| `spurious_attribute` | Male, encoded 0 or 1 |
| `group` | Group ID, 0–3 |
| `split` | Train = 0, validation = 1, test = 2 |

**Group definitions:**

| Group | Target | Spurious Attribute | Description |
|---|---|---|---|
| G0 | 0 | 0 | Non-blond / Non-male |
| G1 | 0 | 1 | Non-blond / Male |
| G2 | 1 | 0 | Blond / Non-male |
| G3 | 1 | 1 | Blond / Male |

These labels follow the dataset's annotations.

Training group counts:

| Group | Number of Examples |
|---|---:|
| G0 | 71,629 |
| G1 | 66,874 |
| G2 | 22,880 |
| G3 | 1,387 |

Group 3 accounts for only approximately 0.85% of training examples.

### Step 2 — Generate ResNet-18 Features

Open `erm_celeba.ipynb`.

The notebook loads an ImageNet-pretrained ResNet-18, removes its original classification layer, and freezes its parameters.

Each image is transformed into a **512-dimensional feature vector** using Torchvision's pretrained-model preprocessing.

Run the dataset-loading and feature-extraction sections to generate:

```text
data/processed/features/
├── train_features.npy
├── train_labels.npy
├── train_groups.npy
├── val_features.npy
├── val_labels.npy
├── val_groups.npy
├── test_features.npy
├── test_labels.npy
└── test_groups.npy
```

Expected shapes:

| Split | Feature Matrix |
|---|---|
| Training | (162770, 512) |
| Validation | (19867, 512) |
| Test | (19962, 512) |

Feature extraction took approximately eight minutes for the training split on the original development MacBook. Performance will vary by hardware.

**Important:** Feature extraction only needs to be completed once. Subsequent experiments load these cached arrays rather than processing every image again.

If the feature files already exist and have the correct dimensions, skip the extraction cells.

---

## 5. Shared Experimental Protocol

All team members should use the same configuration unless a method specifically requires a documented change.

### Architecture

- ImageNet-pretrained ResNet-18
- Frozen feature extractor
- 512-dimensional feature vectors
- Linear classification head: `Linear(512, 2)`
- Cross-entropy classification loss

This is a frozen-feature experimental setup, not full end-to-end ResNet-18 fine-tuning.

### Baseline Hyperparameters

| Parameter | Value |
|---|---|
| Random seed | 42 |
| Additional evaluation seeds | 123, 2026 |
| Optimizer | Adam |
| Learning rate | 0.001 |
| Weight decay | 0.0001 |
| Batch size | 512 |
| Training epochs | 20 |
| Checkpoint selection | Minimum validation loss |

These are the ERM reference settings. Method-specific settings, such as JTT upweighting strength or Group DRO adjustment parameters, should be selected using validation data and documented.

### Evaluation Metrics

Every method should report:

1. **Overall Accuracy:** Accuracy across all validation examples.
2. **Worst-Group Accuracy (WGA):** Minimum accuracy among G0, G1, G2, and G3.
3. **Per-Group Accuracy:** Separate accuracy for each group.
4. **Training Cost:** Training time and relevant preprocessing overhead.

Filtering experiments should additionally report:

- Global filtering rate
- Removal and retention counts for each group
- Group-specific removal percentages

For multi-seed experiments, report mean and standard deviation.

**Important:** Use the validation set for model selection. Reserve the test set for the final evaluation after the methodology and hyperparameters have been finalized.

Do not use validation examples to train a model unless explicitly implementing and documenting a separate calibration protocol.

---

## 6. Experiments and Team Responsibilities

### Experiment 1 — ERM Baseline

**Notebook:** `erm_celeba.ipynb`

**Status:** Completed

Train the linear classification head on all training examples without group weighting or filtering.

Initial measured validation results:

| Metric | Result |
|---|---:|
| Overall accuracy | 93.87% |
| Worst-group accuracy | 17.03% |
| Best-group accuracy | 99.60% |

The worst-performing group was G3 (Blond / Male).

![ERM Group Accuracy](results/figures/erm_celeba_group_accuracy.png)

The model exhibits a substantial group-performance disparity despite high overall accuracy.

### Experiment 2 — EPIC-Inspired Gradient Filtering

**Notebook:** `epic_celeba.ipynb`

**Status:** Preliminary experiments completed

Investigate whether gradient-space isolation disproportionately identifies rare examples.

The current approach:

1. Load the ERM classifier checkpoint.
2. Compute per-example gradients with respect to the linear head.
3. Calculate gradient-space isolation.
4. Compare group-specific filtering rates.
5. Retrain on retained data and compare against random filtering.
6. Analyze representative gradient medoids and small-cluster membership.

This is an **EPIC-inspired adaptation**, not a complete reproduction of EPIC's original iterative poisoning-defense algorithm.

Existing experiments evaluate 5%, 10%, and 20% filtering rates, including multi-seed comparisons.

### Experiment 3 — JTT (Just Train Twice)

**Notebook:** `jtt_celeba.ipynb`

**Task:** Implement two-stage error-based upweighting.

Procedure:

1. Train an initial ERM classifier for a limited number of epochs.
2. Identify its misclassified training examples.
3. Assign higher training weights to those examples.
4. Initialize a new classifier from scratch.
5. Retrain and evaluate worst-group accuracy.

Suggested initial configuration: two identification epochs, followed by validation-based selection of the upweighting factor.

Group annotations should not be used to determine the error-based weights.

### Experiment 4 — Random Filtering

**Notebook:** `random_filtering_celeba.ipynb` (to be created if a separate notebook is needed)

**Task:** Establish matched-rate random data-removal controls.

Procedure:

1. Randomly select 5%, 10%, and 20% of training examples for removal.
2. Retrain the classifier on the retained data.
3. Use the same training configuration as the isolation-filtering experiments.
4. Repeat across seeds.
5. Compare overall accuracy, worst-group accuracy, and per-group retention.

Random-filtering experiments already appear in `epic_celeba.ipynb`. Reuse these results and avoid duplicating experiments unnecessarily.

### Experiment 5 — Group DRO

**Notebook:** `groupdro_celeba.ipynb` (to be created)

**Task:** Implement Group Distributionally Robust Optimization.

Procedure:

1. Load cached features, target labels, and group IDs.
2. Compute training losses separately for G0–G3.
3. Maintain adaptive group weights that emphasize high-loss groups.
4. Optimize the weighted objective.
5. Tune method-specific hyperparameters using validation data.
6. Evaluate average and worst-group accuracy.

Unlike JTT, Group DRO uses group annotations during training.

### Experiment 6 — DFR (Deep Feature Reweighting)

**Notebook:** `dfr_celeba.ipynb` (to be created)

**Task:** Evaluate group-balanced last-layer retraining.

Procedure:

1. Reuse the same frozen ResNet-18 features.
2. Construct a group-balanced retraining/calibration subset from the training partition.
3. Train a new linear classification head on that subset.
4. Evaluate overall and worst-group accuracy.
5. Compare against ERM and Group DRO.

Do not balance or modify the validation/test sets.

Because our representation is already frozen, this is a simplified, feature-based DFR experiment. Document any differences from the original published method.

---

## 7. Final Method Comparison

Every method should produce a results table using the following format:

| Method | Overall Accuracy | Worst-Group Accuracy | Training Cost |
|---|---|---|---|
| ERM | Measured | Measured | Measured |
| EPIC-inspired | Measured | Measured | Measured |
| Random Filtering | Measured | Measured | Measured |
| JTT | Measured | Measured | Measured |
| Group DRO | Measured | Measured | Measured |
| DFR | Measured | Measured | Measured |

The final report should also distinguish **group-aware methods** (Group DRO, DFR) from methods that do not require group annotations during training.

Differences in supervision requirements must be considered when interpreting results.

---

## 8. Saving Experimental Results

Save generated files using consistent naming conventions:

```text
results/
├── figures/
│   ├── erm_celeba_*.png
│   ├── epic_*.png
│   ├── jtt_*.png
│   ├── groupdro_*.png
│   └── dfr_*.png
├── metrics/
│   ├── erm_celeba_*.csv
│   ├── epic_*.csv
│   ├── jtt_*.csv
│   ├── groupdro_*.csv
│   └── dfr_*.csv
└── models/
    └── *.pt
```

Each experiment should save:

- A training history CSV
- Final validation metrics
- Per-group accuracy results
- At least one useful visualization
- Random seeds and experimental hyperparameters

Large model checkpoints, datasets, and cached features should remain local.

**Reproducibility requirement:** If a notebook uses a checkpoint that is excluded from Git, document how to regenerate it. The EPIC notebook currently depends on `results/models/erm_celeba.pt`, which is generated by the ERM notebook.

---

## 9. GitHub Collaboration

Each teammate should work on a separate branch to avoid conflicts.

### Before starting work

Update the local repository:

```bash
git checkout main
git pull origin main
```

Create a personal branch:

```bash
git checkout -b feature/jtt
```

Replace `jtt` with the assigned experiment name, such as `groupdro`, `dfr`, or `random-filtering`.

### After finishing work

Check modified files:

```bash
git status
```

Stage and commit:

```bash
git add .
git commit -m "Implement JTT training and group robustness evaluation"
```

Push the branch:

```bash
git push -u origin feature/jtt
```

Open GitHub and create a **Pull Request** to merge the branch into `main`.

Before merging, review code, notebook outputs, experimental configuration, and generated files.

Avoid committing:

- `data/`
- `.venv/`
- `.pt` or `.pth` checkpoints
- Large temporary files

Do not force-push to the shared `main` branch.

---

## 10. Troubleshooting

**`ModuleNotFoundError`**

Check that the virtual environment is active and that the notebook uses the `.venv` kernel.

```bash
python -m pip install -r requirements.txt
```

**Missing `celeba_metadata.csv`**

Run `explore_celeba.ipynb` from the beginning.

**Missing cached features**

Run the feature-extraction sections in `erm_celeba.ipynb`.

**Missing ERM checkpoint**

Run the ERM training and checkpoint-saving cells before starting EPIC. The checkpoint is not stored in Git.

**No CUDA or Apple MPS device**

PyTorch can fall back to CPU, although image feature extraction will be slower.

**Git push rejected**

Fetch and reconcile teammates' changes before pushing. Avoid force-pushing shared work.

**Image download or extraction failure**

Verify that `img_align_celeba.zip` is a valid archive and that extraction produces exactly 202,599 JPEG images.

---

## 11. References

- Yang et al. (2022). *Not All Poisons are Created Equal: Robust Training against Data Poisoning.* ICML.
- Liu et al. (2021). *Just Train Twice: Improving Group Robustness without Training Group Information.* ICML.
- Sagawa et al. (2020). *Distributionally Robust Neural Networks for Group Shifts: On the Importance of Regularization for Worst-Case Generalization.* ICLR.
- Kirichenko et al. (2023). *Last Layer Re-Training is Sufficient for Robustness to Spurious Correlations.* ICLR.
- Liu et al. (2015). *Deep Learning Face Attributes in the Wild.* ICCV.

**Dataset:** [CelebA on Hugging Face](https://huggingface.co/datasets/Yuehao/celeba)

---

**Project goal:** Understand how gradient-based filtering and alternative robustness methods affect underrepresented groups, identify their limitations, and provide reproducible experimental comparisons on CelebA.
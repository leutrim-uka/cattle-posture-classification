# Pipeline Configuration Parameters Reference

This document provides comprehensive explanations for all configuration parameters used in the moomotion pipeline YAML config files.

## Overview

The moomotion project has **4 main training pipelines**, each with its own configuration file in `configs/pipelines/`:

1. **train_classifier.yaml** — Grouped and randomized splits with classical ML classifiers (XGBoost, RandomForest, LogReg)
2. **train_logo.yaml** — Leave-One-Group-Out evaluation for  (per-cow cross-validation)
3. **train_lstm.yaml** — LSTM sequence models with optional attention
4. **train_on_all_data.yaml** — Train on 2024, evaluate on 2025 (temporal generalization)

---

## Common Parameters (Shared Across Multiple Pipelines)

### Data Loading

#### `data_path` / `train_data_path` / `test_data_path`

**What it does:** Specifies which CSV file to load for training/testing.

**Available datasets:**
- `./data/CWB_2024.csv` — Collar + Weather + Bolus 2024
- `./data/CWB_2025.csv` — Collar + Weather + Bolus 2025
- `./data/CW_FULL_COHORT_2024.csv` — Full cohort 2024
- `./data/CW_FULL_COHORT_2025.csv` — Full cohort 2025
- `./data/FINAL_merged_collar_herde_weather_bolus.csv` — Complete merged 2024 data
- `./data/2025_merged_collar_herde_bolus_weather.csv` — Complete merged 2025 data

**Used in:** All 4 pipelines

**Example:**
```yaml
data_path: ./data/CWB_2024.csv
```

---

### Feature Selection

#### `experiment_no`

**What it does:** Selects a predefined feature set/combination from `core/experiments.py`.

**Available experiments:**
- `1`: IMU features (40-240 mG)
- `3`: Collar + Weather (movement + weather features)
- `4`: Collar + Bolus (movement + bolus features)
- `5`: Collar + Weather + Bolus (movement + weather + bolus features)

Each experiment is a carefully curated set designed to test different feature combinations and their impact on prediction performance.

**Used in:** All 4 pipelines

**Example:**
```yaml
experiment_no: 5
```

---

### Data Preprocessing

#### `missing_strategy`

**What it does:** Controls how missing values (NaN) in the dataset are handled.

**Options:**
- `"keep"` — Retain rows with NaN values; model training/inference handles missing values implicitly
- `"drop"` — Remove entire rows if any column contains NaN

**Rationale:**
- `keep`: Preserves data volume; useful when missingness is sparse
- `drop`: Ensures complete data; useful when missingness is random and small

**Used in:** All 4 pipelines

**Example:**
```yaml
missing_strategy: keep
```

#### `remove_outliers`

**What it does:** Detects and removes anomalous observations using the IQR (Interquartile Range) method.

**How it works:**
1. For each class (standing/lying) separately:
   - Compute Q1 (25th percentile) and Q3 (75th percentile)
   - Define outlier threshold: outside [Q1 - 1.5×IQR, Q3 + 1.5×IQR]
2. Remove rows with feature values outside these thresholds

**Options:** `true` or `false`

**Caution:** Removing outliers artificially makes the task easier by eliminating hard-to-classify samples.

**Used in:** All 4 pipelines

**Example:**
```yaml
remove_outliers: false
```

---

### Feature Engineering

#### `feature_engineering`

**What it does:** Enables automatic computation of derived features from raw inputs.

**Derived features computed when enabled:**
- `Age` — Computed from animal birthday and datetime
- `IMU_total` — Weighted sum of all IMU levels (40-240 mG)
- `IMU_*_diff` — Per-level IMU differences (captures acceleration)
- `IMU_*_diff_rolling_sum` — Rolling window sums of IMU differences

**Options:** `true` or `false`

**Used in:** All 4 pipelines

**Example:**
```yaml
feature_engineering: true
```

#### `drop_original_features`

**What it does:** When `feature_engineering: true`, removes the original raw features after computing engineered ones.

**Rationale:** Creates a cleaner feature space with only derived features.

**Options:** `true` or `false`

**Only meaningful if:** `feature_engineering: true` (ignored otherwise)

**Used in:** All 4 pipelines

**Example:**
```yaml
feature_engineering: true
drop_original_features: true
```

---

### Advanced Preprocessing

#### `use_markov`

**What it does:** Applies first-order Markov chain (HMM) smoothing to model predictions.

**How it works:**
1. After raw model predictions, models state transitions
2. Uses transitions learned from training data to smooth predictions
3. Forces predictions to follow believable state transitions

**Rationale:** Posture doesn't change randomly; transitions are constrained (e.g., lying→standing is rare mid-second).

**Options:** `true` or `false`

**Used in:** train_classifier, train_logo, train_lstm

**Example:**
```yaml
use_markov: false
```

---

## Pipeline-Specific Parameters

### train_classifier.yaml

#### `test_size`

**What it does:** Fraction of data used for testing during train/test splits.

**Options:** Float between 0.0 and 1.0 (e.g., 0.2 = 80% train, 20% test)

**Used when:** `split_strategy: grouped` (ignored for `random`)

**Example:**
```yaml
test_size: 0.2
```

---

#### `split_strategy`

**What it does:** Defines the evaluation methodology for assessing model performance.

**Options:**

- **`"random"`** — k-Fold Stratified Cross-Validation
  - Splits data into k random folds
  - Each observation appears exactly once in a test fold
  - Evaluates generalization ability to new observations
  - Output: multiple fold results aggregated by mean±std
  - Controlled by: `random_cv_folds`, `random_cv_seed`

- **`"grouped"`** — Repeated Holdout with Multiple Random Splits
  - Creates multiple random train/test splits with different seeds
  - Each seed produces a different split
  - Evaluates robustness across various data configurations
  - Output: multiple run results aggregated by mean±std
  - Controlled by: `grouped_seed_start`, `grouped_seed_end`

**Used to study:** Impact of random seed variation on model performance

**Example:**
```yaml
split_strategy: random
```

---

#### Stratified k-Fold Parameters (for `split_strategy: random`)

##### `random_cv_folds`

Number of folds in stratified k-fold CV.

**Options:** Integer ≥ 2 (typical: 3, 5, 10)

```yaml
random_cv_folds: 5
```

##### `random_cv_seed`

Random seed controlling fold splitting (ensures reproducibility).

**Options:** Integer

```yaml
random_cv_seed: 42
```

---

#### Repeated Holdout Parameters (for `split_strategy: grouped`)

##### `grouped_seed_start` and `grouped_seed_end`

Defines the range `[grouped_seed_start, grouped_seed_end)` of seeds for repeated holdout splits.

**How it works:**
- Each integer seed in the range generates one random split
- Suggests: use range of size 10-20 for robust estimates

**Example:**
```yaml
grouped_seed_start: 42
grouped_seed_end: 52  # Generates 10 splits: seeds 42, 43, ..., 51
```

---

#### `models_to_train`

**What it does:** Specifies which ML models to train.

**Available models:**
- `xgboost` — Gradient Boosting (usually best performance)
- `random_forest` — Random Forest
- `logistic_regression` — Logistic Regression
- `svm` — Support Vector Machine
- `naive_bayes` — Naive Bayes
- `knn` — k-Nearest Neighbors
- `decision_tree` — Decision Tree

**Options:** List of model names (can train multiple simultaneously)

**Example:**
```yaml
models_to_train:
  - xgboost
  - random_forest
```

---

#### `param_grid_path`

**What it does:** Path to YAML file defining hyperparameter grids for GridSearchCV.

**Default:** `./configs/parameter_grids.yaml`

**Format:** Each model has a nested parameter dict with lists of values to search.

**Example:**
```yaml
param_grid_path: ./configs/parameter_grids.yaml
```

---

#### `cv_folds`

**What it does:** Number of cross-validation folds during hyperparameter tuning (GridSearchCV).

**Note:** Separate from evaluation folds. Used only for hyperparameter optimization.

**Options:** Integer ≥ 2

**Example:**
```yaml
cv_folds: 5
```

---

#### `use_smote`

**What it does:** Applies SMOTE (Synthetic Minority Over-sampling Technique) to training data.

**How it works:**
1. Detects imbalanced class distribution
2. Creates synthetic minority class samples via interpolation
3. Balances training set for fair classifier learning

**Applied to:** Only training folds/splits (test set remains unmodified)

**Options:** `true` or `false`

**Example:**
```yaml
use_smote: false
```

---

### train_logo.yaml

#### `cow_normalize_imu`

**What it does:** Normalizes IMU (Inertial Measurement Unit) features within each individual cow's data.

**Why it matters:** Different cows' sensors may have different calibration or mounting angles, causing systematic differences in raw IMU readings.

**How it works:**
- For each cow separately:
  - Compute robust normalization (median and IQR-based scaling) of IMU features
  - Apply normalization to that cow's data only
- Corrects per-cow sensor bias while preserving within-cow patterns

**Effect on output directory:**
- `cow_normalize_imu: true` → `logo_cownorm/<suffix>/`
- `cow_normalize_imu: false` → `logo/<suffix>/`

**Options:** `true` or `false`

**Example:**
```yaml
cow_normalize_imu: false
```

---

#### `best_params` (LOGO-specific)

XGBoost hyperparameters used for LOGO evaluation. These typically come from prior hyperparameter tuning.

**Key parameters:**

##### `objective`

Loss function for training.

**Options:**
- `"binary:logistic"` — Outputs class probabilities (0-1)
- `"binary:logitraw"` — Outputs raw scores (unbounded)

**Default:** `"binary:logistic"`

##### `max_depth`

Maximum depth of each decision tree.

- Deeper trees: capture complex patterns but risk overfitting
- Shallow trees: simpler, more generalizable

**Options:** Integer ≥ 1 (typical: 1-10)

**Default:** 3

##### `learning_rate`

Step size for gradient descent (shrinkage parameter).

- Lower rates: slower learning, better regularization, more robust
- Higher rates: faster learning, risk of overfitting

**Options:** Float typically 0.01-0.5

**Default:** 0.1

##### `n_estimators`

Number of boosting rounds (trees sequentially built).

- More trees: generally better performance but slower
- Fewer trees: faster training but may underfit

**Options:** Integer ≥ 1 (typical: 100-1000)

**Default:** 300

##### `subsample`

Fraction of training samples used to build each tree.

- Reduces overfitting by introducing randomness
- Recommended: 0.7-0.9

**Options:** Float 0.0-1.0

**Default:** 0.9

##### `colsample_bytree`

Fraction of features used to build each tree.

- Reduces overfitting by introducing feature randomness
- Recommended: 0.7-0.9

**Options:** Float 0.0-1.0

**Default:** 0.9

##### `eval_metric`

Metric used for model evaluation during training.

**Options:**
- `"logloss"` — Binary cross-entropy (more informative)
- `"error"` — Classification error rate (0/1 loss)

**Default:** `"logloss"`

##### `random_state`

Seed for reproducibility.

**Options:** Integer

**Default:** 42

---

### train_lstm.yaml

#### `attention`

**What it does:** Adds an attention mechanism layer to the LSTM.

**Standard LSTM:**
- Processes entire sequence
- Outputs final hidden state for prediction
- All timesteps equally weighted

**LSTM with Attention:**
- Learns which timesteps are most important
- Assigns attention weights to each timestep
- Weighted sum used for prediction
- Better interpretability and often better performance

**Options:** `true` or `false`

**Example:**
```yaml
attention: false
```

---

#### `is_lstm`

**What it does:** Internal flag indicating LSTM pipeline (must be `true`).

**Used for:** Preprocessing pipeline selection (sequence vs. static features)

**Options:** `true`

**Example:**
```yaml
is_lstm: true
```

---

#### Data Splitting Parameters

##### `test_size`

Fraction of data held out for final evaluation.

**Options:** Float 0.0-1.0

**Example:**
```yaml
test_size: 0.2  # 20% test, 80% becomes train+validation
```

##### `val_size`

Fraction of remaining data (after test split) used for validation.

**How it works:**
```
Total data = 100%
Test = test_size × 100% = 20%
Remaining = 80%
Validation = val_size × 80% = 8%
Training = (1 - val_size) × 80% = 72%
```

**Used for:** Early stopping (monitors model on validation set during training)

**Example:**
```yaml
val_size: 0.1
```

---

#### Multi-Seed Training Parameters

##### `seed_start` and `seed_end`

Defines range `[seed_start, seed_end)` of random seeds for independent model runs.

**Why multiple seeds:** LSTM weights initialized randomly; different seeds explore different local optima.

**How it works:**
- Each seed trains a completely independent LSTM model
- Results are aggregated (mean ± std uncertainty estimates)
- Typical range size: 10-20 runs

**Example:**
```yaml
seed_start: 42
seed_end: 52  # Trains 10 models: seeds 42-51
```

---

### train_on_all_data.yaml

#### `train_data_path` and `test_data_path`

**What it does:** Specifies separate training and test datasets.

**Typical usage:**
- `train_data_path`: 2024 data (all available for training)
- `test_data_path`: 2025 data (held-out future data)

**Tests:** Temporal generalization (can model trained on past predict future?)

**Example:**
```yaml
train_data_path: ./data/FINAL_merged_collar_herde_weather_bolus.csv
test_data_path: ./data/2025_merged_collar_herde_bolus_weather.csv
```

---

#### `test_cow_selection_strategy`

**What it does:** Controls which animals' 2025 data is used for testing.

**Options:**

1. **`"new_cows_test"`** — Only cows NEW in 2025 (not in 2024)
   - Filters: `test_data.animal_id NOT IN train_data.animal_id`
   - Tests: Generalization to completely unseen individuals
   - Use case: Can model handle novel animals with same underlying behavior?

2. **`"recurring_cows_test"`** — Only cows that REPEAT in both 2024 and 2025
   - Filters: `test_data.animal_id IN train_data.animal_id`
   - Tests: Within-subject temporal generalization
   - Use case: Does model maintain accuracy for same animals over time?

3. **`"recurring_cows_to_training"`** — Move recurring 2025 data to training set
   - Takes: 2025's recurring animals and adds them to training
   - Tests only: Completely new animals from 2025
   - Rationale: Maximizes training data by including repeated subjects over time

**Example:**
```yaml
test_cow_selection_strategy: new_cows_test
```

---

#### `out_dir`

**What it does:** Root directory for all output artifacts (models, predictions, metrics).

**Directory structure created:**
```
<out_dir>/<suffix>/
├── train/               # Training set predictions and metrics
├── test/                # Test set predictions and metrics
├── preds/               # Model predictions
└── aggregate_metrics.json  # Summary statistics
```

**Example:**
```yaml
out_dir: ./data/final_results/final_experiment/FINAL/
```

---

## Quick Reference Table

| Parameter | Type | Pipelines | Options/Notes |
|-----------|------|-----------|---------------|
| `data_path` | str | all | CSV file path |
| `experiment_no` | int | all | 1-8 (defines feature set) |
| `missing_strategy` | str | all | "keep", "drop" |
| `remove_outliers` | bool | all | true, false |
| `feature_engineering` | bool | all | true, false |
| `drop_original_features` | bool | all | true, false |
| `use_markov` | bool | classifier, logo, lstm | true, false |
| `split_strategy` | str | classifier | "random", "grouped" |
| `random_cv_folds` | int | classifier | ≥ 2 |
| `random_cv_seed` | int | classifier | any integer |
| `grouped_seed_start` | int | classifier | starting seed |
| `grouped_seed_end` | int | classifier | ending seed (exclusive) |
| `models_to_train` | list | classifier | list of model names |
| `param_grid_path` | str | classifier | YAML file path |
| `cv_folds` | int | classifier | ≥ 2 |
| `use_smote` | bool | classifier | true, false |
| `test_size` | float | classifier, lstm | 0.0-1.0 |
| `cow_normalize_imu` | bool | logo | true, false |
| `attention` | bool | lstm | true, false |
| `is_lstm` | bool | lstm | true (fixed) |
| `val_size` | float | lstm | 0.0-1.0 |
| `seed_start` | int | lstm | starting seed |
| `seed_end` | int | lstm | ending seed (exclusive) |
| `train_data_path` | str | train_on_all | CSV file path |
| `test_data_path` | str | train_on_all | CSV file path |
| `test_cow_selection_strategy` | str | train_on_all | "new_cows_test", "recurring_cows_test", "recurring_cows_to_training" |
| `out_dir` | str | train_on_all | output directory path |
| `best_params` | dict | logo, train_on_all | XGBoost hyperparameters |

---

## Tips for Configuration

### For First-Time Users

Start with defaults in each YAML file. They are designed for reasonable baseline performance.

### For Experimenting with Features

1. Change `experiment_no` to try different feature combinations
2. Toggle `feature_engineering` to add/remove derived features
3. Toggle `use_markov` to test temporal smoothing

### For Addressing Class Imbalance

Use `use_smote: true` in classifier config if you notice class imbalance warnings.

### For Reducing Noise

Use `remove_outliers: true` to clean data, but verify performance impact (outliers sometimes contain important signals).

### For Temporal Validation

Use `train_on_all_data` pipeline with:
- `train_data_path: 2024 data`
- `test_data_path: 2025 data`
- Different `test_cow_selection_strategy` options to measure generalization

### For Understanding Seed Sensitivity

Compare results with different `grouped_seed_start`/`grouped_seed_end` ranges or `seed_start`/`seed_end` ranges to measure stability.

---

## Advanced Configuration Tuning

### Preventing Overfitting

```yaml
# Reduce model complexity
max_depth: 2                # Shallower trees
learning_rate: 0.05        # Slower learning
subsample: 0.7             # Fewer samples per tree
colsample_bytree: 0.7      # Fewer features per tree

# Increase regularization
remove_outliers: true      # Remove noise
use_markov: true           # Enforce transitions
```

### Improving Minority Class Performance

```yaml
use_smote: true            # Balance training data
eval_metric: logloss       # More sensitive to probabilities
```

### Speeding Up Training

```yaml
max_depth: 3               # Faster trees
n_estimators: 100          # Fewer trees
random_cv_folds: 3         # Fewer CV folds
```

---

For further questions, refer to:
- `core/experiments.py` — Feature set definitions
- `core/features.py` — Feature enums and data types
- `pipelines/config.py` — Configuration class definitions
- `steps/` — Individual preprocessing/training steps

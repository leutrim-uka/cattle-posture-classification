# Configuration Setup Guide

Quick recipes for common configuration scenarios.

## Scenario 1: First-Time Setup (Baseline)

**Goal:** Get the pipeline running with sensible defaults to understand performance baseline.

### train_classifier.yaml
```yaml
data_path: ./data/CWB_2024.csv
experiment_no: 5
feature_engineering: false
missing_strategy: keep
remove_outliers: false
split_strategy: random        # Standard cross-validation
random_cv_folds: 5
models_to_train:
  - xgboost
use_smote: false
```

**Run:**
```bash
python run.py --pipeline train_classifier
```

---

## Scenario 2: Measure Cow-Specific Generalization

**Goal:** Understand if model generalizes across different cows (Leave-One-Group-Out).

### train_logo.yaml
```yaml
data_path: ./data/CWB_2024.csv
experiment_no: 1
feature_engineering: false
missing_strategy: keep
remove_outliers: false
cow_normalize_imu: false     # Start without normalization
use_markov: false
```

**Run:**
```bash
python run.py --pipeline train_logo
```

**Then experiment:** Toggle `cow_normalize_imu: true` to compare with/without sensor normalization.

---

## Scenario 3: Temporal Generalization (2024 → 2025)

**Goal:** Test if model trained on 2024 can predict 2025 behavior (real-world validation).

### train_on_all_data.yaml
```yaml
train_data_path: ./data/FINAL_merged_collar_herde_weather_bolus.csv     # All 2024
test_data_path: ./data/2025_merged_collar_herde_bolus_weather.csv       # All 2025
experiment_no: 8
test_cow_selection_strategy: new_cows_test    # Test on completely new animals
feature_engineering: false
missing_strategy: keep
remove_outliers: false
```

**Run:**
```bash
python run.py --pipeline train_on_all_data
```

**Variations:**
- Use `recurring_cows_test` to test on same animals over time
- Use `recurring_cows_to_training` to maximize training data from repeated subjects

---

## Scenario 4: Measure Robustness to Seed Variation

**Goal:** Understand how sensitive the model is to random initialization.

### train_classifier.yaml
```yaml
data_path: ./data/CWB_2024.csv
experiment_no: 5
split_strategy: grouped       # Multiple random train/test splits
grouped_seed_start: 42
grouped_seed_end: 62          # 20 different splits
random_cv_folds: 5
models_to_train:
  - xgboost
use_smote: false
```

**Run:**
```bash
python run.py --pipeline train_classifier
```

**Compare output:** Results will show mean±std across 20 runs, indicating robustness.

---

## Scenario 5: Handle Class Imbalance

**Goal:** Improve performance when standing/lying classes are imbalanced.

### train_classifier.yaml
```yaml
data_path: ./data/CWB_2024.csv
experiment_no: 5
use_smote: true              # Enable synthetic minority oversampling
split_strategy: random
random_cv_folds: 5
remove_outliers: false
models_to_train:
  - xgboost
```

**Run:**
```bash
python run.py --pipeline train_classifier
```

---

## Scenario 6: Experiment with Feature Engineering

**Goal:** Test impact of derived features (Age, IMU_total, differences, rolling sums).

### train_classifier.yaml (Version A: Without Engineering)
```yaml
feature_engineering: false
experiment_no: 5
```

### train_classifier.yaml (Version B: With Engineering)
```yaml
feature_engineering: true
drop_original_features: false  # Keep both original + engineered
experiment_no: 5
```

### train_classifier.yaml (Version C: Engineering Only)
```yaml
feature_engineering: true
drop_original_features: true   # Remove originals, keep only engineered
experiment_no: 5
```

**Run all three, compare:**
```bash
# Run each version and compare aggregate_metrics.json output
python run.py --pipeline train_classifier --config ./configs/pipelines/train_classifier.yaml
```

---

## Scenario 7: Deep Neural Network (LSTM)

**Goal:** Train sequence models with optional attention.

### train_lstm.yaml (Standard LSTM)
```yaml
data_path: ./data/FINAL_merged_collar_herde_weather_bolus.csv
experiment_no: 4
attention: false
seed_start: 42
seed_end: 52               # Train 10 independent models
test_size: 0.2
val_size: 0.1
```

### train_lstm.yaml (LSTM with Attention)
```yaml
data_path: ./data/FINAL_merged_collar_herde_weather_bolus.csv
experiment_no: 4
attention: true            # Enable attention mechanism
seed_start: 42
seed_end: 52
```

**Run:**
```bash
python run.py --pipeline train_lstm
```

**Compare:** attention=true often gives better performance and interpretability.

---

## Scenario 8: Clean Data (Remove Noise)

**Goal:** Test on cleaner dataset to maximize performance.

### train_classifier.yaml
```yaml
data_path: ./data/CWB_2024.csv
experiment_no: 5
missing_strategy: drop       # Remove rows with missing data
remove_outliers: true        # Remove statistical outliers
feature_engineering: false
split_strategy: random
use_smote: false
```

**Warning:** Removing data artificially improves headline metrics but may hide real-world challenges.

---

## Scenario 9: Compare Evaluation Strategies

**Goal:** Understand difference between CV (random) and repeated holdout (grouped).

### train_classifier.yaml (Version A: Random CV)
```yaml
split_strategy: random
random_cv_folds: 5
random_cv_seed: 42
```

### train_classifier.yaml (Version B: Grouped Seeds)
```yaml
split_strategy: grouped
grouped_seed_start: 42
grouped_seed_end: 52
test_size: 0.2
```

**Run both:**
```bash
# Version A: CV splits data multiple ways, each observation tested once per fold
# Version B: Multiple random train/test splits, evaluates robustness

python run.py --pipeline train_classifier
```

**What to observe:**
- If results similar: model is stable to different splits
- If Version B has higher std: model is sensitive to train/test configuration

---

## Scenario 10: Production-Ready Training

**Goal:** Train final model for deployment on both 2024 and 2025 data.

### train_on_all_data.yaml
```yaml
train_data_path: ./data/FINAL_merged_collar_herde_weather_bolus.csv     # All 2024
test_data_path: ./data/2025_merged_collar_herde_bolus_weather.csv       # All 2025
experiment_no: 8
test_cow_selection_strategy: recurring_cows_to_training    # Maximize training data
feature_engineering: true       # Use all available signals
drop_original_features: false   # Keep both raw + engineered
remove_outliers: true           # Clean data
missing_strategy: drop          # Ensure complete cases
out_dir: ./data/final_results/production_v1/
best_params:
  max_depth: 4
  learning_rate: 0.05
  n_estimators: 500
  subsample: 0.8
  colsample_bytree: 0.8
```

**Run:**
```bash
python run.py --pipeline train_on_all_data
```

---

## Configuration Comparison Workflow

To systematically compare configurations:

1. **Create a new config file:** `cp configs/pipelines/train_classifier.yaml configs/pipelines/train_classifier_EXPERIMENT.yaml`
2. **Modify one parameter** (e.g., `feature_engineering: true`)
3. **Run:** `python run.py --pipeline train_classifier --config configs/pipelines/train_classifier_EXPERIMENT.yaml`
4. **Compare metrics:** Check `data/final_results/*/aggregate_metrics.json`
5. **Record results** in a tracking spreadsheet

---

## Debugging Configuration Issues

### Problem: Model performance is poor

**Try:**
1. Check `experiment_no` — might be missing important features
2. Enable `feature_engineering: true` — adds derived signals
3. Check `missing_strategy` — ensure you're not losing too much data with "drop"
4. Reduce `max_depth` — might be overfitting

### Problem: Training is very slow

**Try:**
1. Reduce `random_cv_folds` or `grouped_seed_end - grouped_seed_start`
2. Reduce `n_estimators` in `best_params`
3. Use `random_cv_folds: 3` instead of 5

### Problem: Model shows high variance across seeds

**Try:**
1. Increase `grouped_seed_end - grouped_seed_start` (20+ runs)
2. Reduce `learning_rate` (more stable learning)
3. Increase `subsample` and `colsample_bytree` (less randomness)

### Problem: Results are not reproducible

**Check:**
1. All `random_state` and `seed` parameters are set to fixed values
2. Using `random_cv_seed: 42` (or other fixed seed)
3. Python's PYTHONHASHSEED environment variable is set

---

## Common Pitfalls

| Pitfall | Symptom | Fix |
|---------|---------|-----|
| Wrong experiment_no | Missing important features | Check core/experiments.py for feature name |
| remove_outliers: true | Performance looks too good | Outliers might be important signals; use carefully |
| use_smote: true with balanced data | Training time increases | Only enable if you have class imbalance |
| High seed/fold count | Training never finishes | Reduce grouped_seed_end or cv_folds |
| Different results each run | Non-reproducible output | Ensure all *_seed parameters are fixed values |

---

## Next Steps

1. **Start with Scenario 1** (Baseline) to verify setup
2. **Run Scenarios 2-3** to understand generalization
3. **Experiment within each scenario** by changing one parameter at a time
4. **Document your findings** in a results tracking spreadsheet
5. **Commit winning configurations** to git with descriptive commit messages

---

For detailed parameter explanations, see [PARAMETERS_REFERENCE.md](PARAMETERS_REFERENCE.md).

# Moomotion

Moomotion is a machine learning project for posture classification in cattle using collar, weather, and bolus-derived signals.

## Project Structure

- `pipelines/`: Entry points for training and evaluation workflows.
- `steps/`: Reusable processing, splitting, and model-training units.
- `core/`: Shared domain logic (feature definitions, experiment feature sets, utilities).
- `configs/pipelines/`: Pipeline runtime configs. This is the primary place to change experiments.
- `configs/parameter_grids.yaml`: Hyperparameter search definitions for classical ML models.
- `data/`: Input datasets and generated artifacts.

## Quick Start

1. Install dependencies with Poetry.

```bash
poetry install
```

2. Activate the environment.

```bash
poetry shell
```

3. Run a pipeline via the unified runner.

```bash
python run.py --pipeline train_classifier
```

## Unified Runner

All pipelines are executable through `run.py`:

```bash
python run.py --pipeline <pipeline_name> --config <optional_yaml_path>
```

Supported pipeline names:

- `train_classifier`
- `train_logo`
- `train_lstm`
- `train_on_all_data`

If `--config` is omitted, the runner uses defaults from `configs/pipelines/`.

## Pipeline Configs

Pipelines are now configuration-driven. Runtime parameters should be edited in YAML, not in source files.

Default config files:

- `configs/pipelines/train_classifier.yaml`
- `configs/pipelines/train_logo.yaml`
- `configs/pipelines/train_lstm.yaml`
- `configs/pipelines/train_on_all_data.yaml`

### Example: Classifier Config

```yaml
data_path: ./data/CWB_2024.csv
experiment_no: 5
feature_engineering: false
use_markov: false
split_strategy: random
random_cv_folds: 5
random_cv_seed: 42
models_to_train:
	- xgboost
```

## Evaluation Modes (Classifier)

`train_classifier` supports two split strategies:

- `grouped`: subject-aware splitting (prevents train/test subject overlap).
- `random`: observation-level randomized splitting with stratified outer CV.

For `random`, the pipeline executes stratified K-fold cross-validation and prints aggregate mean ± std metrics across folds.

## Running Pipelines Directly

Each pipeline can also be run directly with a config file:

```bash
python -m pipelines.train_classifier --config ./configs/pipelines/train_classifier.yaml
python -m pipelines.train_logo --config ./configs/pipelines/train_logo.yaml
python -m pipelines.train_lstm --config ./configs/pipelines/train_lstm.yaml
python -m pipelines.train_on_all_data --config ./configs/pipelines/train_on_all_data.yaml
```

## Reproducibility Guidelines

- Keep experiment settings in committed YAML configs.
- Prefer creating a new config file per experiment rather than modifying defaults in place.
- Keep seeds explicit in config files.
- Compare evaluation strategies using the same model, features, and metric definitions.

## Common Workflow for New Contributors

1. Choose a pipeline objective (`train_classifier`, `train_logo`, etc.).
2. Copy the corresponding file from `configs/pipelines/` and edit only the copy.
3. Run through `run.py` with `--config` pointing to the new file.
4. Inspect outputs under `data/final_results/`.
5. Commit both code changes and the config used to produce results.
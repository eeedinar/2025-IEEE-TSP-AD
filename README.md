# ML Classification Framework

Configuration-driven machine learning framework for binary classification.

## Quick Start

```bash
# From scripts directory
cd scripts
python train.py ../configs/config.yaml

# With debug logging
python train.py ../configs/config.yaml --debug
```

## Installation

```bash
conda create -n ml-framework python=3.9
conda activate ml-framework
pip install -r requirements.txt
```

## Configuration

Edit `configs/config.yaml`:

```yaml
data:
  train_files:
    0: 'data/class0.txt'
    1: 'data/class1.txt'
  test_size: 0.2
  validation_size: 0.2

models:
  logistic_regression:
    enabled: true
  
  neural_network:
    enabled: true
    epochs: 1000
    batch_size: 128

training:
  cross_validation:
    enabled: true
    n_folds: 5
```

## Output

Results saved to `results/experiment_timestamp/`:
- `models/` - Trained models
- `results/config.csv` - Metrics summary
- `plots/` - Visualizations
- `logs/` - Training logs

## View Results

```bash
# Check metrics
cat results/config_20251030_120000/results/config.csv

# View log
tail -f results/logs/config_20251030_120000.log
```

# Experiment: config_0.991_20250718_180804

## Overview
- **Date**: 2025-07-18 18:48:41
- **Config**: config_0.991.yaml
- **Random Seed**: 30
- **Model Selection**: Best model chosen by max val_f1

## Directory Structure
- `models/` - Trained model files
- `preprocessing/` - Preprocessing pipeline state and components
  - `preprocessing_state.pkl` - Complete pipeline state
  - `scaler.joblib` - Feature scaler (if used)
  - `nmf_model.joblib` - NMF model (if used)
  - `preprocessing_metadata.json` - Preprocessing configuration and indices
- `metadata/` - Experiment metadata and tracking information
- `results/` - Training results and metrics
- `plots/` - Visualizations and analysis plots
  - `preprocessing/` - Preprocessing visualizations
- `configs/` - Configuration files used

## Results Summary
- **Models Trained**: 4
- **Best Model**: neural_network_weighted_cross_entropy (by val_f1)
- **Final Features**: 14

## Preprocessing Pipeline
- feature_range: 690 → 211 features
- multicollinearity: 211 → 14 features
- scaling: 14 → 14 features

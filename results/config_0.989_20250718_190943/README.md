# Experiment: config_0.989_20250718_190943

## Overview
- **Date**: 2025-07-18 19:29:25
- **Config**: config_0.989.yaml
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
- **Best Model**: neural_network_combined_focal_dice (by val_f1)
- **Final Features**: 9

## Preprocessing Pipeline
- feature_range: 690 → 211 features
- multicollinearity: 211 → 9 features
- scaling: 9 → 9 features

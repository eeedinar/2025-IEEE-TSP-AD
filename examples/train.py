#!/usr/bin/env python3
"""
Machine Learning Training Framework
===================================

A clean, professional training system that reads all configuration from YAML.

Author: ML Engineering Team
Version: 10.0.0
"""
import argparse
import json
import time
import warnings
from pathlib import Path
from datetime import datetime
from typing import Dict, Tuple, Optional, Any, List
import logging
import numpy as np
import pandas as pd
import yaml
import joblib
import traceback
from sklearn.metrics import confusion_matrix
from sklearn.model_selection import train_test_split, StratifiedKFold, KFold
import random
import torch
from torchsummary import summary

# Suppress warnings
warnings.filterwarnings("ignore")

# Suppress matplotlib font manager debug messages
logging.getLogger('matplotlib.font_manager').setLevel(logging.WARNING)
logging.getLogger('matplotlib').setLevel(logging.WARNING)

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Add parent directory to path
import sys
sys.path.append(str(Path(__file__).parent.parent))

# Framework imports
from src.data import DataLoader, PreprocessingPipeline
from src.models import ModelFactory
from src.metrics import MetricsWrapper
from src.utils.model_analysis import ModelAnalyzer
from src.visualization.report_generator import ReportGenerator

class MLTrainer:
    """Main training orchestrator that reads all configuration from YAML."""
    
    def __init__(self, config_path: str):
        """Initialize trainer with configuration file."""
        self.config_path = Path(config_path)
        self.project_root = Path(__file__).parent.parent
        
        # Load configuration
        with open(self.config_path, 'r') as f:
            self.config = yaml.safe_load(f)
        
        # Set up experiment
        self.experiment_name = self._create_experiment_name()
        self.random_seed = self.config['data'].get('random_state', 42)
        self._set_random_seeds()

        # Initialize storage
        self.results = {}
        self.trained_models = {}
        self.preprocessing_state = None
        
        logger.info(f"Initialized trainer - Experiment: {self.experiment_name}")
    
    def _create_experiment_name(self) -> str:
        """Create unique experiment identifier."""
        config_name = self.config_path.stem
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        return f"{config_name}_{timestamp}"
    
    def _set_random_seeds(self):
        """Set random seeds for reproducibility."""    
        random.seed(self.random_seed)
        np.random.seed(self.random_seed)
        
        try:
            torch.manual_seed(self.random_seed)
            torch.cuda.manual_seed_all(self.random_seed)
            torch.backends.cudnn.deterministic = True
            torch.backends.cudnn.benchmark = False
        except ImportError:
            pass
    
    def run(self):
        """Execute the complete training pipeline."""
        logger.info("="*80)
        logger.info("MACHINE LEARNING TRAINING PIPELINE")
        logger.info("="*80)
        logger.info(f"Configuration: {self.config_path.name}")
        logger.info(f"Experiment: {self.experiment_name}")
        logger.info(f"Random seed: {self.random_seed}")
        logger.info("="*80)

        try:
            # Execute pipeline
            data_splits = self._load_and_preprocess_data()

            # saving to check sanity
            # np.save('pipeline_X_train.npy', data_splits['X_train'])
            # np.save('pipeline_y_train.npy', data_splits['y_train'])
            # if data_splits.get('X_val') is not None:
            #     np.save('pipeline_X_val.npy', data_splits['X_val'])
            #     np.save('pipeline_y_val.npy', data_splits['y_val'])
            # if data_splits.get('X_test') is not None:
            #     np.save('pipeline_X_test.npy', data_splits['X_test'])
            #     np.save('pipeline_y_test.npy', data_splits['y_test'])

            self._train_models(data_splits)
            self._save_results()
            self._generate_reports()
            self._print_summary()

        except Exception as e:
            logger.error(f"Pipeline failed: {str(e)}", exc_info=True)
            raise
 
    def _load_and_preprocess_data(self) -> Dict[str, np.ndarray]:
        """Load data and apply preprocessing pipeline.
        processed_data contains:
        {
            'X_train': raw_train_features,           # Raw features after preprocessing
            'X_val': raw_val_features,               # Raw features  
            'X_test': raw_test_features,             # Raw features
            'X_train_nmf': nmf_train_features,       # NMF features (if NMF enabled)
            'X_val_nmf': nmf_val_features,           # NMF features
            'X_test_nmf': nmf_test_features,         # NMF features
            'y_train': train_labels,
            'y_val': val_labels,
            'y_test': test_labels
        }
        """
        logger.info("\n" + "="*60)
        logger.info("DATA LOADING AND PREPROCESSING")
        logger.info("="*60)

        # Create output directory structure early
        output_config = self.config.get('output', {})
        base_output_dir = self.project_root / output_config.get('output_dir', 'results') / self.experiment_name
        preprocessing_plots_dir = base_output_dir / 'plots' / 'preprocessing'
        preprocessing_plots_dir.mkdir(parents=True, exist_ok=True)
        
        # Load config file
        data_config = self.config['data']

        # Load raw data
        loader = DataLoader()
        load_config = data_config.get('train_files', {}) or data_config.get('inference_files', {})
        if not load_config:
            raise ValueError("Empty configuration provided. Expected 'train_files' or 'inference_files' config.")
        X, y = loader.load_from_config(load_config, self.project_root)

        # Load feature metadata if available
        feature_metadata = None
        if data_config.get('feature_indices_file'):
            metadata_path = self.project_root / data_config['feature_indices_file']
            if metadata_path.exists():
                feature_metadata = np.loadtxt(metadata_path)
                logger.info(f"Loaded feature metadata: {len(feature_metadata)} values")
        
        # Create train/val/test splits
        splits = self._create_data_splits(X, y)

        # Apply preprocessing pipeline to the splits
        pipeline = PreprocessingPipeline(data_config)
        processed_data, self.preprocessing_state = pipeline.execute_pipeline(
            X_train=splits['X_train'],
            X_val=splits.get('X_val'),
            X_test=splits.get('X_test'),
            y_train=splits['y_train'],
            y_val=splits.get('y_val'),
            y_test=splits.get('y_test'),
            feature_metadata=feature_metadata,
            mode='train',
            plots_dir=preprocessing_plots_dir
        )

        # Store output directory for later use
        self._output_dir = base_output_dir
        
        # Log preprocessing summary
        logger.info("\nPreprocessing stages:")
        for stage in self.preprocessing_state.stages:
            logger.info(f"  {stage['name']}: {stage['n_features_in']} → {stage['n_features_out']}")
        
        # Return all processed data (includes both raw and NMF features)
        return processed_data 

    def _create_data_splits(self, X: np.ndarray, y: np.ndarray) -> Dict[str, np.ndarray]:
        """Create train/validation/test splits."""
        data_config = self.config['data']

        # Get split parameters
        test_size = data_config.get('test_size', 0.2)
        val_size = data_config.get('validation_size', 0.2)
        stratify = data_config.get('stratify', False)
        split_strategy = data_config.get('split_strategy', 'train_val_test')
        
        # Create splits based on strategy
        if split_strategy == 'train_val_test':
            # First split: train+val vs test
            X_temp, X_test, y_temp, y_test = train_test_split(
                X, y,
                test_size=test_size,
                random_state=self.random_seed,
                stratify=y if stratify else None
            )
            
            # Second split: train vs val
            val_size_adjusted = val_size / (1 - test_size)
            X_train, X_val, y_train, y_val = train_test_split(
                X_temp, y_temp,
                test_size=val_size_adjusted,
                random_state=self.random_seed,
                stratify=y_temp if stratify else None
            )
            
            logger.info(f"\nData splits (before preprocessing):")
            logger.info(f"  Train: {len(X_train)} samples ({len(X_train)/len(X)*100:.1f}%)")
            logger.info(f"  Val: {len(X_val)} samples ({len(X_val)/len(X)*100:.1f}%)")
            logger.info(f"  Test: {len(X_test)} samples ({len(X_test)/len(X)*100:.1f}%)")
            
            # Show class distribution
            for split_name, y_split in [('Train', y_train), ('Val', y_val), ('Test', y_test)]:
                unique, counts = np.unique(y_split, return_counts=True)
                logger.info(f"  {split_name} classes: {dict(zip(unique, counts))}")
            
            return {
                'X_train': X_train, 'y_train': y_train,
                'X_val': X_val, 'y_val': y_val,
                'X_test': X_test, 'y_test': y_test
            }
        else:
            # Simple train/test split
            X_train, X_test, y_train, y_test = train_test_split(
                X, y,
                test_size=test_size,
                random_state=self.random_seed,
                stratify=y if stratify else None
            )
            
            logger.info(f"\nData splits (before preprocessing):")
            logger.info(f"  Train: {len(X_train)} samples ({len(X_train)/len(X)*100:.1f}%)")
            logger.info(f"  Test: {len(X_test)} samples ({len(X_test)/len(X)*100:.1f}%)")
            
            # Show class distribution
            for split_name, y_split in [('Train', y_train), ('Test', y_test)]:
                unique, counts = np.unique(y_split, return_counts=True)
                logger.info(f"  {split_name} classes: {dict(zip(unique, counts))}")
            
            return {
                'X_train': X_train, 'y_train': y_train,
                'X_val': None, 'y_val': None,  # No validation set
                'X_test': X_test, 'y_test': y_test
            }

    def _train_models(self, data_splits: Dict[str, np.ndarray]):
            """Train all enabled models from configuration."""
            logger.info("\n" + "="*60)
            logger.info("MODEL TRAINING")
            logger.info("="*60)
            
            # Store data splits for plotting
            self._last_data_splits = data_splits
            
            # Get configurations
            models_config = self.config.get('models', {})
            training_global = self.config.get('training', {})
            
            # Get evaluation metrics to compute
            self.eval_metrics = self.config.get('evaluation', {}).get('metrics', [
                'accuracy', 'precision', 'recall', 'f1', 'auc', 'mcc', 
                'balanced_accuracy', 'auprc'
            ])

            global_features = ['early_stopping', 'gradient_clipping', 'model_selection', 
                             'optimize_threshold', 'cross_validation']

            # Get enabled models
            enabled_models = [(name, cfg) for name, cfg in models_config.items() if cfg.get('enabled', False)]
            
            # Log models to train
            logger.info(f"Training {len(enabled_models)} models:")
            for name, _ in enabled_models:
                logger.info(f" - {name}")

            # Train each model
            for idx, (model_name, model_config) in enumerate(enabled_models, 1):
                logger.info(f"\n[{idx}/{len(enabled_models)}] Training {model_name}...")

                self._set_random_seeds()  # Reset seeds before each model
                try:
                    # Prepare configuration
                    config = model_config.copy()

                    # Apply global defaults (only if not already in model config)
                    for feature in global_features:
                        if feature not in config and training_global.get(feature, {}).get('enabled'):
                            config[feature] = training_global[feature]

                    # Add dimensions if neural network
                    if any(param in config for param in ['epochs', 'batch_size', 'hidden_dims', 'd_model']):
                        config['input_dim'] = data_splits['X_train'].shape[1]
                        config['n_classes'] = len(np.unique(data_splits['y_train']))

                    # Handle NMF dimensions
                    if 'nmf' in model_name.lower():
                        
                        # Check if NMF features exist
                        if 'X_train_nmf' not in data_splits:
                            raise ValueError(
                                f"Model '{model_name}' requires NMF features, but NMF preprocessing is not enabled. "
                                f"Please enable NMF in your config:\n"
                                f"data:\n"
                                f"  nmf:\n"
                                f"    enabled: true\n"
                                f"    n_components: auto"
                            )
                        raw_dim = data_splits['X_train'].shape[1]
                        nmf_dim = data_splits['X_train_nmf'].shape[1]

                        config.update({
                            'raw_dim': raw_dim,
                            'nmf_dim': nmf_dim,
                        })

                        logger.info(f"NMF model '{model_name}' dimensions: raw={raw_dim}, nmf={nmf_dim}")

                    # Add random seed if model supports it
                    base_name = ModelFactory.resolve_model_name(model_name)
                    if 'random_state' not in config and base_name not in ['naive_bayes', 'knn']:
                        config['random_state'] = self.random_seed
                    
                    # Resolve string references to actual configs
                    for key in ['scheduler', 'optimizer', 'loss']:
                        if key in config and isinstance(config[key], str):
                            section = f"{key}s" if key != 'loss' else 'losses'

                            logger.info(f"Looking for {key}={config[key]} in section {section}")
                            logger.info(f"Available in {section}: {list(self.config.get(section, {}).keys())}")

                            if section in self.config and config[key] in self.config[section]:
                                logger.info(f"Found {config[key]} in {section}")
                                config[key] = self.config[section][config[key]]
                            else:
                                logger.info(f"NOT FOUND: {config[key]} in {section}")[config[key]]
                    
                    # Train model with prepared config
                    logger.info(f"Config for {model_name}:\n" + "\n".join(f"  {k}: {v if not isinstance(v, dict) else '{' + ', '.join(f'{dk}={dv}' for dk, dv in v.items()) + '}'}" for k, v in config.items()))

                    model, metrics = self._train_single_model(model_name, config, data_splits)
                    self.trained_models[model_name] = model
                    self.results[model_name] = metrics

                    logger.info(f"  ✓ Complete - Val F1: {metrics.get('val_f1', 0):.4f}, "
                              f"Test F1: {metrics.get('test_f1', 0):.4f}, "
                              f"Parameters: {metrics.get('total_parameters', 'N/A'):,}")
                    
                except Exception as e:
                    logger.error(f"  ✗ Failed: {str(e)}")
                    logger.debug(f"Traceback:\n{traceback.format_exc()}")
                    self.results[model_name] = {'error': str(e)}
    
    def _train_single_model(self, model_name: str, config: Dict, 
                          data_splits: Dict) -> Tuple[Any, Dict]:
        """Train a single model and evaluate it."""
        start_time = time.time()

        # Create and train model
        base_name = ModelFactory.resolve_model_name(model_name)
        model = ModelFactory.create_model(base_name, config=config)
        
        # Print model summary
        if hasattr(model, 'model') and hasattr(model.model, 'parameters'):

            # Print detailed summary (requires knowing input shape)
            input_shape = (config['input_dim'],)  # Adjust based on your model
            summary(model.model, input_shape)
            logger.info(f"\n{model_name} Detailed Parameter Breakdown:")
            logger.info("="*60)
            
            # List all parameters with their shapes
            total_params = 0
            for name, param in model.model.named_parameters():
                param_count = param.numel()
                total_params += param_count
                logger.info(f"{name}: {list(param.shape)} = {param_count} params")
            
            logger.info(f"\nTotal parameters (manual count): {total_params:,}")
            
            # Also show the architecture
            logger.info("\nModel architecture:")
            print(model.model)


        # Set NMF flag on the model
        model.uses_nmf = 'nmf' in model_name.lower() and 'X_train_nmf' in data_splits

        # Train with appropriate method
        if hasattr(model, 'train'):
            # Check if this is an NMF model
            if model.uses_nmf:
                # Pass NMF features for NMF models
                if 'X_val' in model.train.__code__.co_varnames:
                    model.train(
                        data_splits['X_train'], data_splits['y_train'],
                        data_splits.get('X_val'), data_splits.get('y_val'),
                        X_train_nmf=data_splits['X_train_nmf'],
                        X_val_nmf=data_splits.get('X_val_nmf')
                    )
                else:
                    model.train(
                        data_splits['X_train'], data_splits['y_train'],
                        X_train_nmf=data_splits['X_train_nmf']
                    )
            else:
                # Regular models without NMF
                if 'X_val' in model.train.__code__.co_varnames:
                    model.train(data_splits['X_train'], data_splits['y_train'],
                               data_splits.get('X_val'), data_splits.get('y_val'))
                else:
                    model.train(data_splits['X_train'], data_splits['y_train'])


        # Evaluate model
        metrics = self._evaluate_model(model, data_splits)
        metrics['total_parameters'] = ModelAnalyzer.count_parameters(model)
        metrics['training_time'] = time.time() - start_time
        
        # Optimize threshold
        if config.get('optimize_threshold', {}).get('enabled'):
            if data_splits.get('X_val') is not None and hasattr(model, 'predict_proba'):
                # Find threshold of output probability for best F1 on validation data
                optimize_config = config.get('optimize_threshold', {})
                metric = optimize_config.get('metric', 'f1')
                thresholds = optimize_config.get('thresholds', None)
                threshold = self._optimize_threshold(model, data_splits['X_val'], data_splits['y_val'], 
                                   metric, thresholds, 
                                   X_val_nmf=data_splits.get('X_val_nmf') if model.uses_nmf else None)
                metrics['optimal_threshold'] = threshold

                # Add optimized test metrics
                if model.uses_nmf:
                    y_proba = model.predict_proba(data_splits['X_test'], X_nmf=data_splits.get('X_test_nmf'))
                else:
                    y_proba = model.predict_proba(data_splits['X_test'])

                metrics['test_f1_optimal'] = MetricsWrapper.get_eval_metrics(
                    metrics_names='f1',
                    y_true=data_splits['y_test'],
                    y_pred=y_proba,
                    prob_thr=threshold
                )

        # Cross-validation
        if config.get('cross_validation', {}).get('enabled'):
            if data_splits.get('X_val') is not None:
                cv_results = self._run_cross_validation(model_name, config, data_splits)
                # Merge CV results directly into metrics (flat structure)
                metrics.update(cv_results)

        return model, metrics
    

    def _evaluate_model(self, model, data_splits: Dict) -> Dict:
        """Evaluate model on all splits."""
        metrics = {}

        # Evaluate on each split
        for split in ['train', 'val', 'test']:
            X = data_splits.get(f'X_{split}', None)
            if X is None:  # continue if X is None ex. X_val is not present
                continue
            
            y = data_splits.get(f'y_{split}', None)
    
            # Get predictions and probabilities
            nmf_X = data_splits.get(f'X_{split}_nmf', None)
            if hasattr(model, 'uses_nmf') and model.uses_nmf:
                nmf_X = data_splits.get(f'X_{split}_nmf')
                # For NMF models, pass NMF features to predict
                y_pred = model.predict(X, X_nmf=nmf_X)
                y_proba = None
                if hasattr(model, 'predict_proba'):
                    y_proba = model.predict_proba(X, X_nmf=nmf_X)
            else:
                # Regular prediction
                y_pred = model.predict(X)
                y_proba = None
                if hasattr(model, 'predict_proba'):
                    y_proba = model.predict_proba(X)

            # Compute all metrics using MetricsWrapper
            split_scores = MetricsWrapper.get_eval_metrics(
                metrics_names=self.eval_metrics,
                y_true=y, 
                y_pred=y_proba if y_proba is not None else y_pred
            )
            
            # Add split prefix to metric names
            for metric_name, score in split_scores.items():
                if score is not None and not np.isnan(score):
                    metrics[f'{split}_{metric_name}'] = score

        # Confusion matrix
        if data_splits.get('X_test') is not None:
            if hasattr(model, 'uses_nmf') and model.uses_nmf:
                cm = confusion_matrix(data_splits['y_test'], model.predict(data_splits['X_test'], X_nmf=data_splits.get('X_test_nmf')))
            else:
                cm = confusion_matrix(data_splits['y_test'], model.predict(data_splits['X_test']))
            metrics['confusion_matrix'] = cm.tolist()

        return metrics


    def _optimize_threshold(self, model, X_val: np.ndarray, y_val: np.ndarray, 
                          metric: str = 'f1', thresholds: Optional[List[float]] = None,
                          X_val_nmf: Optional[np.ndarray] = None) -> float:
        """Find optimal classification threshold using MetricsWrapper."""
        if not hasattr(model, 'predict_proba'):
            return 0.5

        if X_val_nmf is not None:
            y_proba = model.predict_proba(X_val, X_nmf=X_val_nmf)[:, 1]
        else:
            y_proba = model.predict_proba(X_val)[:, 1]
        
        # Use provided thresholds or default
        if thresholds is None:
            thresholds = np.linspace(0.1, 0.9, 17)

        best_score = -np.inf
        best_threshold = 0.5
        
        for threshold in thresholds:
            y_pred = (y_proba >= threshold).astype(int)
                
            # Get score using MetricsWrapper
            score = MetricsWrapper.get_eval_metrics(
                metrics_names=metric,
                y_true=y_val,
                y_pred=y_proba,
                prob_thr=threshold
            )
            
            if score is not None and not np.isnan(score) and score > best_score:
                best_score = score
                best_threshold = threshold
        
        return best_threshold
    
    def _run_cross_validation(self, model_name: str, config: Dict, 
                             data_splits: Dict) -> Dict:
        """Run cross-validation for any model type."""
        # Extract parameters
        cv_config = config.get('cross_validation', {})
        n_folds   = cv_config.get('n_folds', 5)
        stratify  = cv_config.get('stratify', False)

        # Check if threshold optimization is enabled
        optimize_threshold_enabled = config.get('optimize_threshold', {}).get('enabled', False)

        # For CV, combine train and val if val exists
        if data_splits.get('X_val') is not None:
            X_cv = np.vstack([data_splits['X_train'], data_splits['X_val']])
            y_cv = np.hstack([data_splits['y_train'], data_splits['y_val']])
            
            # Also combine NMF features if this is an NMF model
            if 'nmf' in model_name.lower() and 'X_train_nmf' in data_splits and 'X_val_nmf' in data_splits:
                X_cv_nmf = np.vstack([data_splits['X_train_nmf'], data_splits['X_val_nmf']])
            else:
                X_cv_nmf = None
        else:
            X_cv = data_splits['X_train']
            y_cv = data_splits['y_train']
            X_cv_nmf = data_splits.get('X_train_nmf') if 'nmf' in model_name.lower() else None

        # Create splitter
        if stratify and len(np.unique(y_cv)) < 20:
            splitter = StratifiedKFold(n_splits=n_folds, shuffle=True, random_state=self.random_seed)
            logger.info(f'Using Stratified KFold of {n_folds} folds')
        else:
            splitter = KFold(n_splits=n_folds, shuffle=True, random_state=self.random_seed)
            logger.info(f'Using KFold of {n_folds} folds')

        # Storage for scores - organized by split
        fold_scores = {
            'train': {metric: [] for metric in self.eval_metrics},
            'val': {metric: [] for metric in self.eval_metrics},
            'test': {metric: [] for metric in self.eval_metrics}
        }
        
        # Add storage for optimal f1 if threshold optimization is enabled
        if optimize_threshold_enabled:
            fold_scores['test']['f1_optimal'] = []

        # Disable CV for fold training
        fold_config = config.copy()
        fold_config['cross_validation'] = {'enabled': False}
        
        # Run k-fold CV
        for fold, (train_idx, val_idx) in enumerate(splitter.split(X_cv, y_cv), 1):
            # Prepare fold data
            fold_data_splits = {
                'X_train': X_cv[train_idx],
                'y_train': y_cv[train_idx],
                'X_val': X_cv[val_idx],
                'y_val': y_cv[val_idx],
                'X_test': data_splits['X_test'],  # reporting on test set
                'y_test': data_splits['y_test']
            }
            
            # Add NMF features if this is an NMF model
            if X_cv_nmf is not None:
                fold_data_splits['X_train_nmf'] = X_cv_nmf[train_idx]
                fold_data_splits['X_val_nmf']   = X_cv_nmf[val_idx]
                fold_data_splits['X_test_nmf']  = data_splits.get('X_test_nmf')

            try:
                # Use _train_single_model for consistency
                model, fold_metrics = self._train_single_model(model_name, fold_config, fold_data_splits)
                            
                # Extract metrics for each split
                for split in ['train', 'val', 'test']:
                    for metric in self.eval_metrics:
                        metric_key = f'{split}_{metric}'
                        if metric_key in fold_metrics:
                            fold_scores[split][metric].append(fold_metrics[metric_key])

                # Extract optimal f1 if available
                if optimize_threshold_enabled and 'test_f1_optimal' in fold_metrics:
                    fold_scores['test']['f1_optimal'].append(fold_metrics['test_f1_optimal'])

                # Display fold results
                val_f1 = fold_metrics.get('val_f1', 0)
                test_f1 = fold_metrics.get('test_f1', 0)
                params = fold_metrics.get('total_parameters', 'N/A')
                if isinstance(params, (int, float)):
                    params = f"{params:,}"
                logger.info(f"  Fold {fold} ✓ Complete - Val F1: {val_f1:.4f}, Test F1: {test_f1:.4f}, Parameters: {params}")
                    
            except Exception as e:
                logger.warning(f"Fold {fold} failed for {model_name}: {str(e)}")
                continue
        
        # Aggregate results
        results = {'cv_n_folds': n_folds}
        
        # Calculate mean and std for each split and metric
        for split in ['train', 'val', 'test']:
            for metric, scores in fold_scores[split].items():
                if scores:
                    results[f'cv_mean_{split}_{metric}'] = float(np.mean(scores))
                    results[f'cv_std_{split}_{metric}'] = float(np.std(scores))

         # Log summary
        logger.info(f"  ✓ Cross-validation complete for {model_name}: {n_folds} folds")
        if results.get('cv_mean_val_f1') is not None:
            logger.info(f"    Val F1: {results.get('cv_mean_val_f1', 0):.4f}±{results.get('cv_std_val_f1', 0):.4f}")
        if results.get('cv_mean_test_f1') is not None:
            logger.info(f"    Test F1: {results.get('cv_mean_test_f1', 0):.4f}±{results.get('cv_std_test_f1', 0):.4f}")

        return results

    def _save_results(self):
        """Save all training results and models in organized structure."""
        logger.info("\n" + "="*60)
        logger.info("SAVING RESULTS")
        logger.info("="*60)
        
        output_config = self.config.get('output', {})
        base_output_dir = self._output_dir  # Already set in _load_and_preprocess_data
        
        # Create organized directory structure
        dirs = {
            'models': base_output_dir / 'models',
            'preprocessing': base_output_dir / 'preprocessing',
            'metadata': base_output_dir / 'metadata',
            'results': base_output_dir / 'results',
            'plots': base_output_dir / 'plots',
            'configs': base_output_dir / 'configs'
        }
        
        for dir_path in dirs.values():
            dir_path.mkdir(parents=True, exist_ok=True)
        
        # Save models
        if output_config.get('save_models', True):
            for model_name, model in self.trained_models.items():
                model_path = dirs['models'] / f"{model_name}.joblib"
                try:
                    if hasattr(model, 'save'):
                        model.save(str(model_path))
                    else:
                        joblib.dump(model, str(model_path))
                    logger.info(f"  ✓ Saved model: {model_name}")
                except Exception as e:
                    logger.error(f"  ✗ Failed to save {model_name}: {e}")
        
        # Save preprocessing state and components
        # Create preprocessing pipeline instance to access save methods
        pipeline = PreprocessingPipeline(self.config['data'])
        pipeline.state = self.preprocessing_state
        
        # Save complete state
        state_path = dirs['preprocessing'] / "preprocessing_state.pkl"
        pipeline.save_state(state_path)
        
        # Save individual components
        saved_components = pipeline.save_components(dirs['preprocessing'])
        
        logger.info("  ✓ Saved preprocessing components")
        
        # Save results
        if output_config.get('save_results', True):
            results_format = output_config.get('results_format', 'both')
            
            # Save detailed results
            if results_format in ['json', 'both']:
                with open(dirs['results'] / 'detailed_results.json', 'w') as f:
                    # Filter out non-serializable data
                    serializable_results = {}
                    for model_name, metrics in self.results.items():
                        serializable_results[model_name] = {
                            k: v if not isinstance(v, np.ndarray) else v.tolist()
                            for k, v in metrics.items()
                        }
                    json.dump(serializable_results, f, indent=2)
            
            # Save results summary as CSV
            if results_format in ['csv', 'both']:
                # Create DataFrame only with successful models
                successful_results = {k: v for k, v in self.results.items() if 'error' not in v}
                if successful_results:
                    results_data = []

                    for model_name, metrics in successful_results.items():
                        row = {'model': model_name}
                        
                        # Add all regular metrics (non-CV, non-list/dict)
                        for k, v in metrics.items():
                            if not k.startswith('cv_') and not isinstance(v, (list, dict, np.ndarray)):
                                row[k] = v
                        # Handle confusion matrix
                        if 'confusion_matrix' in metrics:
                            row['confusion_matrix'] = metrics['confusion_matrix']

                        # Add CV metrics as mean±std if CV was performed
                        if 'cv_n_folds' in metrics:
                            # Find all CV metrics that were actually calculated
                            cv_metrics_by_split = {
                                'train': set(),
                                'val': set(),
                                'test': set()
                            }
                            
                            # Collect all CV metrics organized by split
                            for key in metrics:
                                if key.startswith('cv_mean_'):
                                    # Extract split and metric name
                                    parts = key.split('_', 3)  # ['cv', 'mean', 'split', 'metric']
                                    if len(parts) == 4:
                                        split = parts[2]
                                        metric_name = parts[3]
                                        if split in cv_metrics_by_split:
                                            cv_metrics_by_split[split].add(metric_name)
                            
                            # Add CV metrics to row
                            for split, metric_set in cv_metrics_by_split.items():
                                for metric in metric_set:
                                    mean_key = f'cv_mean_{split}_{metric}'
                                    std_key = f'cv_std_{split}_{metric}'
                                    if mean_key in metrics and std_key in metrics:
                                        mean = metrics[mean_key]
                                        std = metrics[std_key]
                                        row[f'cv_{split}_{metric}'] = f"{mean:.4f}±{std:.4f}"

                        results_data.append(row)

                    # Create and save DataFrame
                    results_df = pd.DataFrame(results_data).set_index('model')
                    config_name = self.config_path.stem  # Get filename without extension
                    csv_filename = f'{config_name}.csv'
                    results_df.to_csv(dirs['results'] / csv_filename)
                    logger.info(f"  ✓ Saved results to {csv_filename}")
        
        # Save experiment metadata
        metadata = {
            'experiment': {
                'name': self.experiment_name,
                'timestamp': datetime.now().isoformat(),
                'config_file': self.config_path.name,
                'random_seed': self.random_seed,
                'framework_version': '10.0.0'
            },
            'data': {
                'preprocessing_stages': [
                    {'name': s['name'], 'features_in': s['n_features_in'], 'features_out': s['n_features_out']}
                    for s in self.preprocessing_state.stages
                ],
                'final_features': self.preprocessing_state.stages[-1]['n_features_out'] if self.preprocessing_state.stages else 0
            },
            'models': {
                'trained': list(self.trained_models.keys()),
                'best_model': self._get_best_model(),
                'failed': [k for k, v in self.results.items() if 'error' in v]
            },
            'model_selection': self.config.get('evaluation', {}).get('model_selection', {
                'metric': 'val_f1',
                'mode': 'max'
            }),
            'preprocessing': {
                'components': saved_components if output_config.get('save_preprocessing_artifacts', True) else {},
                'state_file': 'preprocessing_state.pkl'
            },
            'paths': {
                'models': 'models/',
                'preprocessing': 'preprocessing/',
                'results': 'results/',
                'plots': 'plots/',
                'configs': 'configs/'
            }
        }
        
        with open(dirs['metadata'] / 'experiment_metadata.json', 'w') as f:
            json.dump(metadata, f, indent=2)
        
        # Save configuration copy
        import shutil
        shutil.copy2(self.config_path, dirs['configs'] / 'config.yaml')
        
        # Create README
        selection_config = self.config.get('evaluation', {}).get('model_selection', {})
        metric = selection_config.get('metric', 'val_f1')
        mode = selection_config.get('mode', 'max')
        
        readme_content = f"""# Experiment: {self.experiment_name}

        ## Overview
        - **Date**: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}
        - **Config**: {self.config_path.name}
        - **Random Seed**: {self.random_seed}
        - **Model Selection**: Best model chosen by {mode} {metric}

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
        - **Models Trained**: {len(self.trained_models)}
        - **Best Model**: {self._get_best_model()} (by {metric})
        - **Final Features**: {self.preprocessing_state.stages[-1]['n_features_out'] if self.preprocessing_state.stages else 'N/A'}

        ## Preprocessing Pipeline
        {chr(10).join(f"- {s['name']}: {s['n_features_in']} → {s['n_features_out']} features" for s in self.preprocessing_state.stages)}
        """
        
        with open(base_output_dir / 'README.md', 'w') as f:
            f.write(readme_content)
        
        logger.info(f"\n  All results saved to: {base_output_dir}")
    
    def _generate_reports(self):
        """Generate comprehensive visualization reports."""
        report_generator = ReportGenerator(self.config)
        report_generator.generate_all_reports(
            results=self.results,
            trained_models=self.trained_models,
            data_splits=self._last_data_splits,
            preprocessing_state=self.preprocessing_state,
            output_dir=self._output_dir
        )
    
    def _get_best_model(self) -> Optional[str]:
        """Get the best performing model based on configured metric."""
        # Get selection criteria from config
        selection_config = self.config.get('evaluation', {}).get('model_selection', {})
        metric = selection_config.get('metric', 'val_f1')
        mode = selection_config.get('mode', 'max')  # 'max' or 'min'
        
        # Filter successful models that have the metric
        successful_models = {
            k: v for k, v in self.results.items() 
            if 'error' not in v and metric in v
        }

        if not successful_models:
            return None
        
        # Select based on mode
        if mode == 'min':
            return min(successful_models.keys(), 
                      key=lambda k: successful_models[k][metric])
        else:
            return max(successful_models.keys(), 
                      key=lambda k: successful_models[k][metric])
    
    def _print_summary(self):
        """Print training summary."""
        logger.info("\n" + "="*80)
        logger.info("TRAINING SUMMARY")
        logger.info("="*80)
        
        # Count successful models
        successful = [k for k, v in self.results.items() if 'error' not in v]
        failed = [k for k, v in self.results.items() if 'error' in v]
        
        logger.info(f"\nModels trained: {len(successful)}/{len(self.results)}")
        
        # Get selection criteria
        selection_config = self.config.get('evaluation', {}).get('model_selection', {})
        metric = selection_config.get('metric', 'val_f1')
        mode = selection_config.get('mode', 'max')
        
        if successful:
            # Sort by the configured metric
            reverse = (mode == 'max')
            sorted_models = sorted(
                successful,
                key=lambda k: self.results[k].get(metric, float('-inf') if reverse else float('inf')),
                reverse=reverse
            )
            
            logger.info(f"\nTop models by {metric} ({mode}):")
            for i, model in enumerate(sorted_models[:5], 1):
                metrics = self.results[model]
                param_str = f"{metrics.get('total_parameters', 0):,}" if metrics.get('total_parameters', 0) > 0 else "N/A"
                metric_value = metrics.get(metric, 'N/A')
                
                # Format metric value
                if isinstance(metric_value, (int, float)):
                    metric_str = f"{metric_value:.4f}"
                else:
                    metric_str = str(metric_value)
                
                # Get test F1 for comparison
                test_f1 = metrics.get('test_f1', 'N/A')
                test_f1_str = f"{test_f1:.4f}" if isinstance(test_f1, (int, float)) else str(test_f1)
                
                logger.info(
                    f"  {i}. {model:<30} {metric}: {metric_str}, "
                    f"Test F1: {test_f1_str}, "
                    f"Params: {param_str}"
                )
        
        if failed:
            logger.info(f"\nFailed models: {', '.join(failed)}")
        
        logger.info("\n" + "="*80)

class TeeOutput:
    def __init__(self, file):
        self.terminal = sys.__stdout__
        self.log = file
        
    def write(self, message):
        self.terminal.write(message)
        self.log.write(message)
        self.log.flush()
        
    def flush(self):
        pass

def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(
        description="Machine Learning Training Framework"
    )
    
    parser.add_argument(
        'config',
        type=str,
        help='Path to configuration file (YAML)'
    )
    
    parser.add_argument(
        '--debug',
        action='store_true',
        help='Enable debug logging'
    )
    
    args = parser.parse_args()
    
    # Validate config file
    config_path = Path(args.config)
    if not config_path.exists():
        logger.error(f"Configuration file not found: {args.config}")
        return 1

    # Read config to get output directory
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)

    # Get output directory from config, default to 'results'
    output_dir = config.get('output', {}).get('output_dir', 'results')

    # Setup log file using config's output directory
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    log_dir = Path(__file__).parent.parent / output_dir / "logs"
    log_dir.mkdir(parents=True, exist_ok=True)
    log_file = log_dir / f"{config_path.stem}_{timestamp}.log"
    
    # Open log file for writing
    with open(log_file, 'w') as f:
        # Redirect stdout and stderr to both console and file
        sys.stdout = TeeOutput(f)
        sys.stderr = sys.stdout
        
        # Configure logging
        logging.basicConfig(
            level=logging.DEBUG if args.debug else logging.INFO,
            format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
            force=True  # Override any existing config
        )
        
        print(f"Log file: {log_file}")
        print("="*80)
        
        # Run training
        try:
            trainer = MLTrainer(args.config)
            trainer.run()
            return 0
        except Exception as e:
            logging.error(f"Training failed: {e}", exc_info=True)
            return 1


if __name__ == "__main__":
    sys.exit(main())
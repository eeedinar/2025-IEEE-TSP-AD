"""
Preprocessing Pipeline Implementation
=========================================================

A modular preprocessing pipeline for machine learning workflows.
Single execute_pipeline method with mode parameter for simplicity.

Pipeline Stages:
1. Feature range selection
2. Multicollinearity removal (fit on X_train only)
3. NMF transformation (fit on X_train only)
4. Feature scaling (fit on X_train only)

"""

import numpy as np
from pathlib import Path
from typing import Dict, Tuple, Optional, List, Any, Union
from dataclasses import dataclass, field
import joblib
import json
import logging
from sklearn.preprocessing import StandardScaler, MinMaxScaler, RobustScaler
from sklearn.decomposition import NMF

try:
    # Try relative imports first (when used as a module)
    from .feature_range_selector import FeatureRangeSelector, FeatureSelectionResult
    from .multicollinearity_analyzer import MulticollinearityAnalyzer
    from .nmf_processor import NMFProcessor
    from .feature_scaler import FeatureScaler
except ImportError:
    # Fall back to direct imports (when run directly)
    from feature_range_selector import FeatureRangeSelector, FeatureSelectionResult
    from multicollinearity_analyzer import MulticollinearityAnalyzer
    from nmf_processor import NMFProcessor
    from feature_scaler import FeatureScaler

logger = logging.getLogger(__name__)


@dataclass
class PreprocessingState:
    """Stores preprocessing state for inference."""
    stages: List[Dict[str, Any]] = field(default_factory=list)
    transformers: Dict[str, Any] = field(default_factory=dict)
    feature_indices: Dict[str, List[int]] = field(default_factory=dict)
    config: Dict[str, Any] = field(default_factory=dict)

class PreprocessingPipeline:
    """
    A preprocessing pipeline with single execute method.

    Modes:
    - 'train': Fit transformers on X_train, transform all splits
    - 'inference': Transform new data using saved state
    """
    
    def __init__(self, config: Dict[str, Any]):
        """
        Initialize with configuration dictionary.
        
        Args:
            config: Configuration dictionary containing settings for each stage
        """
        self.config = config
        self.state = PreprocessingState(config=config)
        

    def execute_pipeline(
        self,
        X_train: Optional[np.ndarray] = None,
        X_val: Optional[np.ndarray] = None,
        X_test: Optional[np.ndarray] = None,
        X_inference: Optional[np.ndarray] = None,
        y_train: Optional[np.ndarray] = None,
        y_val: Optional[np.ndarray] = None,
        y_test: Optional[np.ndarray] = None,
        feature_metadata: Optional[np.ndarray] = None,
        mode: str = 'train',
        saved_state: Optional[PreprocessingState] = None,
        plots_dir: Optional[Path] = None
    ) -> Union[Tuple[Dict[str, np.ndarray], PreprocessingState], np.ndarray]:
        """
        Execute preprocessing with specified mode.
        
        Args:
            X_train, X_val, X_test: Training data splits (for mode='train')
            X_inference: New data (for mode='inference')
            y_train, y_val, y_test: Optional labels for each split
            feature_metadata: Optional feature metadata (e.g., q-values for scattering data)
            mode: 'train' or 'inference'
            saved_state: Required for mode='inference'
            plots_dir: Directory to save preprocessing plots (optional)
            
        Returns:
            For 'train': (dict of processed splits, preprocessing state)
            For 'inference': processed array
        """
        if mode == 'train':
            if X_train is None:
                raise ValueError("X_train required for training mode")
            return self._train_mode(
                X_train, X_val, X_test, 
                y_train, y_val, y_test,  # Pass all labels
                feature_metadata, plots_dir
            )

        elif mode == 'inference':
            if X_inference is None or saved_state is None:
                raise ValueError("X_inference and saved_state required for inference mode")

            # Validate state compatibility
            self.validate_inference_state(saved_state, X_inference.shape)
            return self._inference_mode(X_inference, saved_state)

        else:
            raise ValueError(f"Invalid mode: {mode}. Must be 'train' or 'inference'")

    def _train_mode(
        self,
        X_train: np.ndarray,
        X_val: Optional[np.ndarray],
        X_test: Optional[np.ndarray],
        y_train: Optional[np.ndarray],
        y_val: Optional[np.ndarray],
        y_test: Optional[np.ndarray],
        feature_metadata: Optional[np.ndarray],
        plots_dir: Optional[Path] = None
    ) -> Tuple[Dict[str, np.ndarray], PreprocessingState]:
        """
        Execute pipeline in training mode.

        Fits all transformers on X_train and applies to all splits.

        TRAINING OUTPUT:
        {
        'X_train': raw_train_features,           # Raw features
        'X_val': raw_val_features,               # Raw features  
        'X_test': raw_test_features,             # Raw features
        'X_train_nmf': nmf_train_features,       # NMF features
        'X_val_nmf': nmf_val_features,           # NMF features
        'X_test_nmf': nmf_test_features,         # NMF features
        'y_train': train_labels,
        'y_val': val_labels,
        'y_test': test_labels
        }
        """
        # Package data for consistent handling
        data = {'train': X_train}
        if X_val is not None:
            data['val'] = X_val
        if X_test is not None:
            data['test'] = X_test

        n_features_start = data['train'].shape[1]

        # Apply each stage in sequence
        if self.config.get('feature_range', {}).get('enabled', False):
            data = self._apply_feature_range(data, feature_metadata, plots_dir)

        if self.config.get('multicollinearity', {}).get('enabled', False):
            data = self._apply_multicollinearity(data, y_train, plots_dir)

        if self.config.get('nmf', {}).get('enabled', False):
            data = self._apply_nmf(data, plots_dir)

        if self.config.get('scaling', {}).get('enabled', False):
            data = self._apply_scaling(data)

        # Log transformation summary
        n_features_end = data['train'].shape[1]
        logger.info(f"Preprocessing complete: {n_features_start} → {n_features_end} features")

        # Prepare output with consistent naming
        output = {f'X_{k}': v for k, v in data.items()} # ['train', 'val', 'test'] and ['train_nmf', 'val_nmf', 'test_nmf']

        # Add labels
        if y_train is not None:
            output['y_train'] = y_train
        if y_val is not None:
            output['y_val'] = y_val
        if y_test is not None:
            output['y_test'] = y_test

        return output, self.state

    def _inference_mode(
        self,
        X: np.ndarray,
        saved_state: PreprocessingState
    ) -> Dict[str, np.ndarray]:
        """
        Execute pipeline in inference mode.
        
        Returns both raw and NMF features after processing.

        INFERENCE OUTPUT:
        {
            'raw': processed_raw_features,           # Raw features
            'nmf': processed_nmf_features            # NMF features  
        }
        """
        X_raw = X.copy()  # Keep original for NMF processing
        X_nmf = None  # Explicitly initialize

        # Apply each transformation in sequence
        for stage in saved_state.stages:
            stage_name = stage['name']
            
            if stage_name == 'feature_range':
                # Apply feature selection
                indices = saved_state.feature_indices.get('feature_range', [])
                if indices:
                    X_raw = X_raw[:, indices]
                    
            elif stage_name == 'multicollinearity':
                # Apply multicollinearity selection (relative to current features)
                indices = saved_state.feature_indices.get('multicollinearity_relative', [])
                if indices:
                    X_raw = X_raw[:, indices]
                    
            elif stage_name == 'nmf':
                # Apply NMF transformation to get NMF features
                transformer = saved_state.transformers.get('nmf')
                if transformer:
                    X_nmf = transformer.transform(X_raw)
                else:
                    X_nmf = None
                    
            elif stage_name == 'scaling':
                # Apply scaling to both raw and NMF features
                raw_scaler = saved_state.transformers.get('raw_scaler')
                if raw_scaler:
                    X_raw = raw_scaler.transform(X_raw)
                
                # Apply NMF scaler if NMF features exist
                if X_nmf is not None:
                    nmf_scaler = saved_state.transformers.get('nmf_scaler')
                    if nmf_scaler:
                        X_nmf = nmf_scaler.transform(X_nmf)
        
        # Return both raw and NMF features
        result = {'raw': X_raw}
        if X_nmf is not None:
            result['nmf'] = X_nmf
    
        return result

    def _apply_feature_range(
        self,
        data: Dict[str, np.ndarray],
        feature_metadata: Optional[np.ndarray],
        plots_dir: Optional[Path] = None
    ) -> Dict[str, np.ndarray]:
        """
        Apply feature range selection using FeatureRangeSelector.
        
        Args:
            data: Dictionary of data splits
            feature_metadata: Optional feature values (e.g., q-values)
            plots_dir: Directory to save plots
            
        Returns:
            Dictionary of filtered data splits
        """
        config = self.config['feature_range']
        
        # Initialize selector
        selector = FeatureRangeSelector(config)
        
        # Get number of features from training data
        if 'train' not in data:
            raise ValueError("'train' key must be present in data dictionary")

        # Select features and filter all data splits
        selected_indices, result = selector.select_features(
            X_train=data['train'],
            feature_values=feature_metadata
        )
        
        # Apply selection and update state
        filtered_data = self._apply_selection(data, selected_indices)

        # Generate plot if plots directory provided
        if plots_dir:
            plot_path = plots_dir / 'feature_range_selection.png'
            selector.plot_selection(data, filtered_data, result, feature_metadata, plot_path)
        
        # Update state
        self._update_state('feature_range', result)

        return filtered_data

    def _apply_multicollinearity(self, data: Dict[str, np.ndarray], y_train: Optional[np.ndarray] = None, plots_dir: Optional[Path] = None) -> Dict[str, np.ndarray]:
        """
        Apply multicollinearity removal based on X_train.
        
        Removes highly correlated features to reduce redundancy.
        """    
        config = self.config['multicollinearity']

        # Initialize analyzer
        analyzer = MulticollinearityAnalyzer(
            correlation_threshold=config.get('correlation_threshold', 0.95),
            min_features=config.get('min_features', 1),
            combination_method=config.get('combination_method', 'union')
        )

        # Get feature names from previous stage if available
        feature_names = None
        for stage in self.state.stages:
            if stage['name'] == 'feature_range' and 'selected_values' in stage:
                feature_names = [f"{v:.4f}" for v in stage['selected_values']]
                break

        # Prepare data - either split by class or use all
        X_input = data['train']
        if y_train is not None and len(np.unique(y_train)) > 1:
            X_input = [data['train'][y_train == label] for label in np.unique(y_train)]

        # Analyze and select features
        _, indices_result, _ = analyzer.analyze_and_select_features(
            X_input, 
            feature_names=feature_names,
            verbose=False,
            plot=plots_dir is not None,
            save_path=str(plots_dir / 'multicollinearity') if plots_dir else None
        )

        # Get selected indices - handle both single array and multi-array results
        selected_indices = indices_result if isinstance(indices_result, np.ndarray) else indices_result['combined']

        # Apply selection and update state
        filtered_data = self._apply_selection(data, selected_indices)

        result = FeatureSelectionResult(
            selected_indices=selected_indices,
            selected_values=None,
            n_features_before=data['train'].shape[1],
            n_features_after=len(selected_indices),
            value_range=None
        )
        
        # Prepare extra info with per-class indices if available
        extra_info = {'correlation_threshold': config.get('correlation_threshold', 0.95)}
        if isinstance(indices_result, dict):
            # Store per-class indices
            extra_info['per_class_indices_relative'] = {}
            for key, indices in indices_result.items():
                if key != 'combined':  # Skip combined, already stored
                    extra_info['per_class_indices_relative'][key] = indices.tolist()

        self._update_state('multicollinearity', result, extra_info=extra_info)

        return filtered_data
    
    def _apply_nmf(self, data: Dict[str, np.ndarray], plots_dir: Optional[Path] = None) -> Dict[str, np.ndarray]:
        """
        Apply Non-negative Matrix Factorization transformation.
        
        Reduces dimensionality while preserving non-negative structure.
        """

        # Create NMF processor with config
        nmf_config = {
            **self.config.get('nmf', {}),
            'random_state': self.config.get('random_state', 42)
        }
        nmf_processor = NMFProcessor(nmf_config)

        # Determine plot path
        plot_path = None
        if plots_dir:
            plots_dir.mkdir(parents=True, exist_ok=True)
            plot_path = plots_dir / 'nmf_rank_selection.png'

        # Create separate outputs instead of concatenated
        transformed_data = {}
        
        # Keep raw features in main keys (X_train stays as raw features)
        for key in data.keys():
            transformed_data[key] = data[key].copy()  # Raw features

        # NMF fit on train data
        transformed_data['train_nmf'] = nmf_processor.fit_transform(data['train'], plot_path)

        # Transform other splits
        for key in ['val', 'test']:
            if key in data:
                transformed_data[f'{key}_nmf'] = nmf_processor.transform(data[key])
        
        # Update state
        self.state.transformers['nmf'] = nmf_processor.nmf
        self.state.stages.append({
            'name': 'nmf',
            'n_features_in': data['train'].shape[1],
            'n_features_out': nmf_processor.selected_rank,
        })

        return transformed_data

    def _apply_scaling(self, data: Dict[str, np.ndarray]) -> Dict[str, np.ndarray]:
        """
        Apply feature scaling transformation using dictionary-driven approach with explicit fit/transform logic.
        """
        config = self.config['scaling']
        scaler_type = config.get('scaler_type', 'standard')
        
        scaled_data = {}
        
        # Define scaling configuration
        scaling_config = {
            'raw_scaler': {
                'fit': 'train',
                'transform': ['val', 'test']
            },
            'nmf_scaler': {
                'fit': 'train_nmf', 
                'transform': ['val_nmf', 'test_nmf']
            }
        }

        for scaler_name, config_dict in scaling_config.items():
            fit_key = config_dict['fit']
            transform_keys = config_dict['transform']
            
            if fit_key in data:
                # Create and fit scaler
                scaler = FeatureScaler(scaler_type)
                scaled_data[fit_key] = scaler.fit_transform(data[fit_key])
                
                # Transform other splits
                for key in transform_keys:
                    if key in data:
                        scaled_data[key] = scaler.transform(data[key])
                
                # Store scaler
                self.state.transformers[scaler_name] = scaler.scaler
        
        # Update state
        self.state.stages.append({
            'name': 'scaling',
            'type': scaler_type,
            'n_features_in': data['train'].shape[1],
            'n_features_out': data['train'].shape[1]
        })
        
        nmf_part = f", nmf {data['train_nmf'].shape[1]} → {data['train_nmf'].shape[1]}" if 'train_nmf' in data else ""
        logger.info(f"  scaling: raw {data['train'].shape[1]} → {data['train'].shape[1]}{nmf_part}")        
        return scaled_data

    def _apply_selection(
        self, 
        data: Dict[str, np.ndarray], 
        indices: np.ndarray
    ) -> Dict[str, np.ndarray]:
        """Apply index selection to all data splits."""
        return {key: value[:, indices] for key, value in data.items()}
    
    def _update_state(
        self, 
        stage_name: str, 
        result: FeatureSelectionResult,
        extra_info: Optional[Dict[str, Any]] = None
    ) -> None:
        """Update pipeline state with stage information."""
        # Store absolute indices for feature range
        if stage_name == 'feature_range':
            self.state.feature_indices[stage_name] = result.selected_indices.tolist()
            
        # Store relative indices for multicollinearity
        elif stage_name == 'multicollinearity':
            self.state.feature_indices['multicollinearity_relative'] = result.selected_indices.tolist()
            
            # Also store global indices for reference
            prev_indices = self.state.feature_indices.get('feature_range')
            if prev_indices:
                global_indices = [prev_indices[i] for i in result.selected_indices]
                self.state.feature_indices['multicollinearity_global'] = global_indices
        
        # Store stage info
        stage_info = {
            'name': stage_name,
            'n_features_in': result.n_features_before,
            'n_features_out': result.n_features_after,
        }
        
        if result.value_range:
            stage_info['value_range'] = result.value_range
            
        # Store selected values (q-values) if available
        if result.selected_values is not None:
            stage_info['selected_values'] = result.selected_values.tolist() if isinstance(result.selected_values, np.ndarray) else result.selected_values
            
        if extra_info:
            stage_info.update(extra_info)
            
        self.state.stages.append(stage_info)
    
    def validate_inference_state(self, state: PreprocessingState, X_shape: Tuple[int, int]) -> None:
        """
        Validate that saved state is compatible with inference data.
        
        Args:
            state: Saved preprocessing state
            X_shape: Shape of inference data
            
        Raises:
            ValueError: If state is incompatible with data
        """
        expected_features = X_shape[1]
        
        # Check first stage expects correct number of features
        if state.stages:
            first_stage = state.stages[0]
            if first_stage['n_features_in'] != expected_features:
                raise ValueError(
                    f"Feature mismatch: inference data has {expected_features} features, "
                    f"but pipeline expects {first_stage['n_features_in']} features"
                )
        
        # Validate required components are present
        for stage in state.stages:
            stage_name = stage['name']
            
            if stage_name in ['feature_range', 'multicollinearity']:
                # Check indices exist
                if stage_name == 'feature_range' and 'feature_range' not in state.feature_indices:
                    raise ValueError(f"Missing feature indices for {stage_name}")
                elif stage_name == 'multicollinearity' and 'multicollinearity_relative' not in state.feature_indices:
                    raise ValueError(f"Missing relative indices for {stage_name}")

            elif stage_name == 'nmf':
                # Check NMF transformer exists
                if 'nmf' not in state.transformers:
                    raise ValueError(f"Missing NMF transformer")

            elif stage_name == 'scaling':
                # Check required scalers exist - need at least raw_scaler
                if 'raw_scaler' not in state.transformers:
                    raise ValueError(f"Missing raw_scaler for scaling stage")

                # If NMF stage exists, should also have nmf_scaler
                has_nmf_stage = any(s['name'] == 'nmf' for s in state.stages)
                if has_nmf_stage and 'nmf_scaler' not in state.transformers:
                    raise ValueError(f"Missing nmf_scaler for scaling stage (NMF pipeline detected)")

    
    def get_feature_names(self, original_names: Optional[List[str]] = None) -> List[str]:
        """
        Get feature names after preprocessing.
        
        Args:
            original_names: Original feature names (optional)
            
        Returns:
            List of feature names after all transformations
        """
        if not self.state.stages:
            return original_names or []
        
        # Get final stage info
        final_stage = self.state.stages[-1]
        n_features = final_stage['n_features_out']
        
        # Generate names based on final transformation
        if final_stage['name'] == 'nmf':
            return [f'nmf_component_{i}' for i in range(n_features)]
        elif original_names:
            # Track which original features remain
            indices = self._get_final_feature_indices()
            if indices and len(indices) == n_features:
                return [original_names[i] for i in indices if i < len(original_names)]

        # Default names
        return [f'feature_{i}' for i in range(n_features)]
    
    def _get_final_feature_indices(self) -> Optional[List[int]]:
        """Get the final feature indices after all selections."""
        # Only applicable if no dimensionality reduction was applied
        if any(stage['name'] == 'nmf' for stage in self.state.stages):
            return None
            
        # Start with all features
        indices = None
        
        for stage in self.state.stages:
            if stage['name'] == 'feature_range':
                indices = self.state.feature_indices.get('feature_range')
            elif stage['name'] == 'multicollinearity' and indices:
                mc_indices = self.state.feature_indices.get('multicollinearity_relative', [])
                indices = [indices[i] for i in mc_indices]
                
        return indices

    def save_state(self, filepath: Union[str, Path]) -> None:
        """
        Save preprocessing state to disk.
        
        Args:
            filepath: Path to save the state file
        """
        filepath = Path(filepath)
        filepath.parent.mkdir(parents=True, exist_ok=True)
        joblib.dump(self.state, filepath)
        logger.info(f"Saved preprocessing state to {filepath}")
    
    def save_components(self, output_dir: Union[str, Path]) -> Dict[str, str]:
        """
        Save individual preprocessing components.
        
        Args:
            output_dir: Directory to save components
            
        Returns:
            Dictionary mapping component names to their saved paths
        """
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        
        saved_paths = {}
        
        # Save scaler if exists
        if 'scaler' in self.state.transformers:
            scaler_path = output_dir / 'scaler.joblib'
            joblib.dump(self.state.transformers['scaler'], scaler_path)
            saved_paths['scaler'] = 'scaler.joblib'
            logger.info(f"  Saved scaler to {scaler_path}")
        
        # Save NMF if exists
        if 'nmf' in self.state.transformers:
            nmf_path = output_dir / 'nmf_model.joblib'
            joblib.dump(self.state.transformers['nmf'], nmf_path)
            saved_paths['nmf'] = 'nmf_model.joblib'
            logger.info(f"  Saved NMF model to {nmf_path}")
        
        # Save preprocessing metadata
        metadata = {
            'stages': self.state.stages,
            'feature_indices': self.state.feature_indices,
            'config': self.state.config,
            'final_features': self.state.stages[-1]['n_features_out'] if self.state.stages else 0
        }
        
        metadata_path = output_dir / 'preprocessing_metadata.json'
        with open(metadata_path, 'w') as f:
            json.dump(metadata, f, indent=2)
        saved_paths['metadata'] = 'preprocessing_metadata.json'
        logger.info(f"  Saved preprocessing metadata to {metadata_path}")
        
        return saved_paths
    
    @staticmethod
    def load_state(filepath: Union[str, Path]) -> PreprocessingState:
        """
        Load preprocessing state from disk.
        
        Args:
            filepath: Path to the state file
            
        Returns:
            Loaded preprocessing state
        """
        filepath = Path(filepath)
        state = joblib.load(filepath)
        logger.info(f"Loaded preprocessing state from {filepath}")
        return state


# Example usage
if __name__ == "__main__":
    # Example configuration
    config = {
        'feature_range': {
            'enabled': True,
            'min_value': 0.4,
            'max_value': 1.45
        },
        'multicollinearity': {
            'enabled': True,
            'correlation_threshold': 0.95,
            'min_features': 10
        },
        'nmf': {
            'enabled': True,
            'n_components': 'auto',
            'max_iter': 1000000,
            'tolerance': 1e-4,
            'init': 'nndsvd'
        },
        'scaling': {
            'enabled': True,
            'scaler_type': 'standard'
        },
        'random_state': 42,
        'visualization': {
            'save_plots': True
        }
    }
    
    # Create pipeline
    pipeline = PreprocessingPipeline(config)
    
    # Example data
    n_samples, n_features = 100, 50 # specify samples

    X_train = np.random.rand(n_samples, n_features)
    X_val = np.random.rand(n_samples // 2, n_features)
    X_test = np.random.rand(n_samples // 2, n_features)
    
    # Create labels for demonstration (binary classification)
    y_train = np.array([0] * (n_samples // 2) + [1] * (n_samples // 2))
    y_val = np.array([0] * (n_samples // 4) + [1] * (n_samples // 4))
    y_test = np.array([0] * (n_samples // 4) + [1] * (n_samples // 4))
    
    q_values = np.linspace(0.4, 1.45, n_features)
    
    # Execute in training mode
    print("=" * 60)
    print("TRAINING MODE")
    print("=" * 60)
    
    # Create plots directory
    plots_dir = Path("preprocessing_plots")
    plots_dir.mkdir(exist_ok=True)
    
    processed_data, state = pipeline.execute_pipeline(
        X_train=X_train,
        X_val=X_val,
        X_test=X_test,
        y_train=y_train,
        y_val=y_val,
        y_test=y_test,
        feature_metadata=q_values,
        mode='train',
        plots_dir=plots_dir
    )
    
    print(f"\nOriginal shape: {X_train.shape}")
    print(f"Processed shapes:")
    for key, value in processed_data.items():
        print(f"  {key}: {value.shape}")
    
    print(f"\nPipeline stages:")
    for stage in state.stages:
        print(f"  {stage['name']}: {stage['n_features_in']} → {stage['n_features_out']} features")
    
    # Save state
    pipeline.save_state('preprocessing_state.pkl')
    
    # Simulate inference on new data
    print("\n" + "=" * 60)
    print("INFERENCE MODE")
    print("=" * 60)
    
    # Load state (simulating a new session)
    loaded_state = PreprocessingPipeline.load_state('preprocessing_state.pkl')
    
    # New data with same original feature dimension
    X_new = np.random.rand(10, n_features)
    print(f"\nNew data shape: {X_new.shape}")
    
    # Create new pipeline instance and run inference
    inference_pipeline = PreprocessingPipeline(config)
    X_processed = inference_pipeline.execute_pipeline(
        X_inference=X_new,
        mode='inference',
        saved_state=loaded_state
    )
    
    # Handle dictionary output
    print("Processed data contents:")
    for key, value in X_processed.items():
        print(f"  {key}: {value.shape}")

    # Verify dimensions match training
    assert X_processed['raw'].shape[1] == processed_data['X_train'].shape[1], \
        "Raw feature dimensions should match between training and inference!"

    if 'nmf' in X_processed:
        assert X_processed['nmf'].shape[1] == processed_data['X_train_nmf'].shape[1], \
            "NMF feature dimensions should match between training and inference!"

    print("\n✓ Inference validation passed!")

    # Get feature names
    original_names = [f'q_{q:.3f}' for q in q_values]
    final_names = pipeline.get_feature_names(original_names)
    print(f"\nFinal feature names (first 5): {final_names[:5]}")

    # Show how to use the inference results
    print("\n" + "=" * 40)
    print("USING INFERENCE RESULTS")
    print("=" * 40)

    print("For NMF-SimCLR model:")
    print(f"  Raw features: {X_processed['raw'].shape}")
    if 'nmf' in X_processed:
        print(f"  NMF features: {X_processed['nmf'].shape}")
        print("  Usage: model.forward(raw_features=X_processed['raw'], nmf_features=X_processed['nmf'])")
    else:
        print("  No NMF features (pipeline without NMF)")
    
    # Clean up
    import os
    if os.path.exists('preprocessing_state.pkl'):
        os.remove('preprocessing_state.pkl')
    
    # Optional: Clean up plots directory
    # import shutil
    # if os.path.exists('preprocessing_plots'):
    #     shutil.rmtree('preprocessing_plots')
"""Data preprocessing utilities with validation split support."""

import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, MinMaxScaler, RobustScaler
from typing import Tuple, Optional, Dict, Any, Union
from loguru import logger

try:
    from imblearn.over_sampling import SMOTE, ADASYN
    from imblearn.under_sampling import RandomUnderSampler
    from imblearn.combine import SMOTETomek
    IMBALANCED_LEARN_AVAILABLE = True
except ImportError:
    IMBALANCED_LEARN_AVAILABLE = False
    logger.warning("imbalanced-learn not available. Install with: pip install imbalanced-learn")


class DataPreprocessor:
    """Handles data preprocessing and transformation with validation split support."""
    
    def __init__(self, config: Optional[Dict[str, Any]] = None):
        """
        Initialize the DataPreprocessor.
        
        Args:
            config: Configuration dictionary
        """
        self.config = config or {}
        self.scaler = None
        self.fitted = False
        
    def prepare_data_with_validation(
        self,
        X: np.ndarray,
        y: np.ndarray,
        test_size: Optional[float] = None,
        validation_size: Optional[float] = None,
        random_state: Optional[int] = None,
        stratify: Optional[bool] = None,
        scale: Optional[bool] = None,
        scaler_type: Optional[str] = None,
        handle_imbalance: Optional[str] = None
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """
        Prepare data with train/validation/test split.
        
        Args:
            X: Feature matrix
            y: Label vector
            test_size: Proportion of data for testing
            validation_size: Proportion of training data for validation
            random_state: Random seed
            stratify: Whether to stratify the splits
            scale: Whether to scale features
            scaler_type: Type of scaler ('standard', 'minmax', 'robust')
            handle_imbalance: Method to handle imbalance
            
        Returns:
            X_train, X_val, X_test, y_train, y_val, y_test
        """
        # Get parameters from config if not provided
        test_size = test_size if test_size is not None else self.config.get('test_size', 0.3)
        validation_size = validation_size if validation_size is not None else self.config.get('validation_size', 0.2)
        random_state = random_state if random_state is not None else self.config.get('random_state', 42)
        stratify = stratify if stratify is not None else self.config.get('stratify', True)
        scale = scale if scale is not None else self.config.get('scale', True)
        scaler_type = scaler_type if scaler_type is not None else self.config.get('scaler_type', 'standard')
        handle_imbalance = handle_imbalance if handle_imbalance is not None else self.config.get('handle_imbalance')
        
        # First split: train+val vs test
        X_temp, X_test, y_temp, y_test = train_test_split(
            X, y,
            test_size=test_size,
            random_state=random_state,
            stratify=y if stratify else None
        )
        
        # Second split: train vs val
        # Calculate validation size relative to temp set
        val_size_adjusted = validation_size / (1 - test_size)
        
        X_train, X_val, y_train, y_val = train_test_split(
            X_temp, y_temp,
            test_size=val_size_adjusted,
            random_state=random_state,
            stratify=y_temp if stratify else None
        )
        
        logger.info(f"Train/val/test split: {len(X_train)}/{len(X_val)}/{len(X_test)} samples")
        logger.info(f"Split proportions: {len(X_train)/len(X):.1%}/{len(X_val)/len(X):.1%}/{len(X_test)/len(X):.1%}")
        
        # Handle class imbalance (only on training data)
        if handle_imbalance and IMBALANCED_LEARN_AVAILABLE:
            X_train, y_train = self.handle_class_imbalance(
                X_train, y_train, method=handle_imbalance, random_state=random_state
            )
        
        # Scale features
        if scale:
            X_train, X_val, X_test = self.scale_features_with_validation(
                X_train, X_val, X_test, scaler_type=scaler_type
            )
        
        return X_train, X_val, X_test, y_train, y_val, y_test
        
    def prepare_data(
        self,
        X: np.ndarray,
        y: np.ndarray,
        test_size: float = 0.3,
        random_state: int = 42,
        stratify: bool = True,
        scale: bool = True,
        scaler_type: str = 'standard',
        handle_imbalance: Optional[str] = None
    ) -> Union[Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray],
               Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]]:
        """
        Prepare data for training and testing.
        
        This method now checks the split_strategy in config:
        - 'train_test': Returns (X_train, X_test, y_train, y_test)
        - 'train_val_test': Returns (X_train, X_val, X_test, y_train, y_val, y_test)
        
        Args:
            X: Feature matrix
            y: Label vector
            test_size: Proportion of data for testing
            random_state: Random seed
            stratify: Whether to stratify the split
            scale: Whether to scale features
            scaler_type: Type of scaler ('standard', 'minmax', 'robust')
            handle_imbalance: Method to handle imbalance
            
        Returns:
            Depending on split_strategy:
            - train_test: (X_train, X_test, y_train, y_test)
            - train_val_test: (X_train, X_val, X_test, y_train, y_val, y_test)
        """
        # Check split strategy
        split_strategy = self.config.get('split_strategy', 'train_test')
        
        if split_strategy == 'train_val_test':
            # Use validation split
            return self.prepare_data_with_validation(
                X, y, test_size, None, random_state, stratify, scale, scaler_type, handle_imbalance
            )
        
        # Original train/test split
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, 
            test_size=test_size, 
            random_state=random_state,
            stratify=y if stratify else None
        )
        
        logger.info(f"Train/test split: {len(X_train)}/{len(X_test)} samples")
        
        # Scale features
        if scale:
            X_train, X_test = self.scale_features(
                X_train, X_test, scaler_type=scaler_type
            )
        
        # Handle class imbalance
        if handle_imbalance and IMBALANCED_LEARN_AVAILABLE:
            X_train, y_train = self.handle_class_imbalance(
                X_train, y_train, method=handle_imbalance
            )
        
        return X_train, X_test, y_train, y_test
    
    def scale_features(
        self,
        X_train: np.ndarray,
        X_test: np.ndarray,
        scaler_type: str = 'standard'
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Scale features using specified scaler.
        
        Args:
            X_train: Training features
            X_test: Test features
            scaler_type: Type of scaler to use
            
        Returns:
            Scaled X_train and X_test
        """
        if scaler_type == 'standard':
            self.scaler = StandardScaler()
        elif scaler_type == 'minmax':
            self.scaler = MinMaxScaler()
        elif scaler_type == 'robust':
            self.scaler = RobustScaler()
        else:
            raise ValueError(f"Unknown scaler type: {scaler_type}")
        
        X_train_scaled = self.scaler.fit_transform(X_train)
        X_test_scaled = self.scaler.transform(X_test)
        
        self.fitted = True
        logger.info(f"Features scaled using {scaler_type} scaler")
        
        return X_train_scaled, X_test_scaled
    
    def scale_features_with_validation(
        self,
        X_train: np.ndarray,
        X_val: np.ndarray,
        X_test: np.ndarray,
        scaler_type: str = 'standard'
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Scale features for train/val/test split.
        
        Args:
            X_train: Training features
            X_val: Validation features
            X_test: Test features
            scaler_type: Type of scaler to use
            
        Returns:
            Scaled X_train, X_val, and X_test
        """
        if scaler_type == 'standard':
            self.scaler = StandardScaler()
        elif scaler_type == 'minmax':
            self.scaler = MinMaxScaler()
        elif scaler_type == 'robust':
            self.scaler = RobustScaler()
        else:
            raise ValueError(f"Unknown scaler type: {scaler_type}")
        
        # Fit only on training data
        X_train_scaled = self.scaler.fit_transform(X_train)
        X_val_scaled = self.scaler.transform(X_val)
        X_test_scaled = self.scaler.transform(X_test)
        
        self.fitted = True
        logger.info(f"Features scaled using {scaler_type} scaler")
        
        return X_train_scaled, X_val_scaled, X_test_scaled
    
    def handle_class_imbalance(
        self,
        X: np.ndarray,
        y: np.ndarray,
        method: str = 'smote',
        random_state: int = 42
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Handle class imbalance using various techniques.
        
        Args:
            X: Feature matrix
            y: Label vector
            method: Method to use ('smote', 'adasyn', 'undersample', 'smote_tomek')
            random_state: Random seed
            
        Returns:
            Resampled X and y
        """
        if not IMBALANCED_LEARN_AVAILABLE:
            logger.warning("imbalanced-learn not available, skipping resampling")
            return X, y
        
        original_shape = X.shape
        
        if method == 'smote':
            sampler = SMOTE(random_state=random_state)
        elif method == 'adasyn':
            sampler = ADASYN(random_state=random_state)
        elif method == 'undersample':
            sampler = RandomUnderSampler(random_state=random_state)
        elif method == 'smote_tomek':
            sampler = SMOTETomek(random_state=random_state)
        else:
            raise ValueError(f"Unknown resampling method: {method}")
        
        X_resampled, y_resampled = sampler.fit_resample(X, y)
        
        logger.info(
            f"Resampling with {method}: {original_shape} -> {X_resampled.shape}"
        )
        logger.info(
            f"Class distribution after resampling: "
            f"Class 0: {(y_resampled == 0).sum()}, "
            f"Class 1: {(y_resampled == 1).sum()}"
        )
        
        return X_resampled, y_resampled
    
    def transform(self, X: np.ndarray) -> np.ndarray:
        """
        Transform new data using fitted scaler.
        
        Args:
            X: Feature matrix
            
        Returns:
            Transformed features
        """
        if not self.fitted or self.scaler is None:
            raise ValueError("Preprocessor not fitted. Call prepare_data first.")
        
        return self.scaler.transform(X)
    
    def get_feature_statistics(
        self, 
        X: np.ndarray, 
        y: np.ndarray
    ) -> Dict[str, Any]:
        """
        Get statistics about the features.
        
        Args:
            X: Feature matrix
            y: Label vector
            
        Returns:
            Dictionary of statistics
        """
        stats = {
            'n_samples': len(X),
            'n_features': X.shape[1],
            'class_distribution': {
                'class_0': int((y == 0).sum()),
                'class_1': int((y == 1).sum())
            },
            'imbalance_ratio': float((y == 1).sum() / (y == 0).sum()),
            'feature_stats': {}
        }
        
        # Per-feature statistics
        for i in range(X.shape[1]):
            stats['feature_stats'][f'feature_{i}'] = {
                'mean': float(X[:, i].mean()),
                'std': float(X[:, i].std()),
                'min': float(X[:, i].min()),
                'max': float(X[:, i].max()),
                'nan_count': int(np.isnan(X[:, i]).sum()),
                'inf_count': int(np.isinf(X[:, i]).sum())
            }
        
        # Per-class statistics
        stats['class_stats'] = {}
        for class_label in [0, 1]:
            X_class = X[y == class_label]
            stats['class_stats'][f'class_{class_label}'] = {
                'n_samples': len(X_class),
                'mean_per_feature': X_class.mean(axis=0).tolist(),
                'std_per_feature': X_class.std(axis=0).tolist()
            }
        
        return stats
    
    def _plot_feature_selection(
        self,
        I_class0: np.ndarray,
        I_class1: np.ndarray,
        selected_indices_0: np.ndarray,
        selected_indices_1: np.ndarray,
        combined_indices: np.ndarray,
        feature_indices: np.ndarray
    ):
        """
        Plot mean intensity profiles with selected features highlighted.
        
        This visualization shows:
        - Mean intensity profiles for each class with standard deviation bands
        - Class-specific selected features
        - Combined selected features
        """
        import matplotlib.pyplot as plt
        
        # Calculate statistics
        mean_class0 = I_class0.mean(axis=0)
        std_class0 = I_class0.std(axis=0)
        mean_class1 = I_class1.mean(axis=0)
        std_class1 = I_class1.std(axis=0)
        
        # Feature indices (x-axis)
        x = np.arange(len(mean_class0))
        
        # Create figure with two subplots
        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 10))
        
        # Plot 1: Mean intensity with class-specific features
        ax1.plot(x, mean_class0, 'b-', label="Class 0 Mean", linewidth=1.5)
        ax1.fill_between(x, mean_class0 - std_class0, mean_class0 + std_class0, 
                         color='b', alpha=0.2, label="Class 0 Std")
        ax1.plot(x, mean_class1, 'r-', label="Class 1 Mean", linewidth=1.5)
        ax1.fill_between(x, mean_class1 - std_class1, mean_class1 + std_class1, 
                         color='r', alpha=0.2, label="Class 1 Std")
        
        # Highlight class-specific selected features
        ax1.plot(selected_indices_0, mean_class0[selected_indices_0], 'bo', 
                markersize=8, label=f"Selected (Class 0): {len(selected_indices_0)}")
        ax1.plot(selected_indices_1, mean_class1[selected_indices_1], 'ro', 
                markersize=8, label=f"Selected (Class 1): {len(selected_indices_1)}")
        
        ax1.set_title("Mean Intensity with Class-Specific Selected Features", fontsize=14)
        ax1.set_xlabel("Feature Index")
        ax1.set_ylabel("Intensity")
        ax1.legend(loc='best')
        ax1.grid(True, alpha=0.3)
        
        # Plot 2: Mean intensity with combined features
        ax2.plot(x, mean_class0, 'b-', label="Class 0 Mean", linewidth=1.5)
        ax2.fill_between(x, mean_class0 - std_class0, mean_class0 + std_class0, 
                         color='b', alpha=0.2)
        ax2.plot(x, mean_class1, 'r-', label="Class 1 Mean", linewidth=1.5)
        ax2.fill_between(x, mean_class1 - std_class1, mean_class1 + std_class1, 
                         color='r', alpha=0.2)
        
        # Highlight combined selected features
        ax2.plot(combined_indices, mean_class0[combined_indices], 'go', 
                markersize=8, label=f"Combined Selected: {len(combined_indices)}")
        ax2.plot(combined_indices, mean_class1[combined_indices], 'go', 
                markersize=8)
        
        ax2.set_title("Mean Intensity with Combined Selected Features", fontsize=14)
        ax2.set_xlabel("Feature Index")
        ax2.set_ylabel("Intensity")
        ax2.legend(loc='best')
        ax2.grid(True, alpha=0.3)
        
        plt.tight_layout()
        
        # Save plot if output directory is configured
        if hasattr(self, 'config') and 'output_dir' in self.config:
            output_dir = Path(self.config['output_dir']) / 'plots'
            output_dir.mkdir(parents=True, exist_ok=True)
            plt.savefig(output_dir / 'feature_selection.png', dpi=300, bbox_inches='tight')
            logger.info(f"Feature selection plot saved to {output_dir / 'feature_selection.png'}")
        
        # Check if we should show plots
        try:
            from models_config import OUTPUT_CONFIG
            if OUTPUT_CONFIG.get('show_plots', False):
                plt.show()
            else:
                plt.close()
        except ImportError:
            plt.close()
        
        # Additional plot with q-values if available
        if self.tracker and self.tracker.feature_values is not None:
            self._plot_feature_selection_with_qvalues(
                I_class0, I_class1, 
                selected_indices_0, selected_indices_1,
                combined_indices, feature_indices
            )
    
    def _plot_feature_selection_with_qvalues(
        self,
        I_class0: np.ndarray,
        I_class1: np.ndarray,
        selected_indices_0: np.ndarray,
        selected_indices_1: np.ndarray,
        combined_indices: np.ndarray,
        feature_indices: np.ndarray
    ):
        """Plot mean intensity vs q-values with selected features."""
        import matplotlib.pyplot as plt
        
        # Get q-values for current features
        q_values = np.array(self.tracker.feature_values)[feature_indices]
        
        # Calculate statistics
        mean_class0 = I_class0.mean(axis=0)
        std_class0 = I_class0.std(axis=0)
        mean_class1 = I_class1.mean(axis=0)
        std_class1 = I_class1.std(axis=0)
        
        # Create plot
        plt.figure(figsize=(12, 6))
        
        # Plot mean intensity vs q-values
        plt.plot(q_values, mean_class0, 'b-', label="Class 0 Mean", linewidth=1.5)
        plt.fill_between(q_values, mean_class0 - std_class0, mean_class0 + std_class0, 
                         color='b', alpha=0.2)
        plt.plot(q_values, mean_class1, 'r-', label="Class 1 Mean", linewidth=1.5)
        plt.fill_between(q_values, mean_class1 - std_class1, mean_class1 + std_class1, 
                         color='r', alpha=0.2)
        
        # Highlight selected features
        q_selected = q_values[combined_indices]
        plt.plot(q_selected, mean_class0[combined_indices], 'go', 
                markersize=8, label=f"Selected Features: {len(combined_indices)}")
        plt.plot(q_selected, mean_class1[combined_indices], 'go', markersize=8)
        
        plt.title("Mean Intensity vs Q-values with Selected Features", fontsize=14)
        plt.xlabel("Q-value (Å⁻¹)")
        plt.ylabel("Intensity")
        plt.legend(loc='best')
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        
        # Save plot
        if hasattr(self, 'config') and 'output_dir' in self.config:
            output_dir = Path(self.config['output_dir']) / 'plots'
            output_dir.mkdir(parents=True, exist_ok=True)
            plt.savefig(output_dir / 'feature_selection_qvalues.png', dpi=300, bbox_inches='tight')
            logger.info(f"Q-value feature selection plot saved")
        
        # Check if we should show plots
        try:
            from models_config import OUTPUT_CONFIG
            if OUTPUT_CONFIG.get('show_plots', False):
                plt.show()
            else:
                plt.close()
        except ImportError:
            plt.close()

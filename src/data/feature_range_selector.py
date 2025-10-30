"""
Feature Range Selector
======================

An implementation for selecting features based on value ranges or indices.
Supports both continuous value-based selection (e.g., q-values) and index-based selection.

"""

import numpy as np
from typing import Dict, Tuple, Optional, List, Any, Union
from dataclasses import dataclass
from pathlib import Path
import logging
import matplotlib.pyplot as plt

logger = logging.getLogger(__name__)


@dataclass
class FeatureSelectionResult:
    """Container for feature selection results."""
    selected_indices: np.ndarray
    selected_values: Optional[np.ndarray]  # Changed from List[str] to np.ndarray
    n_features_before: int
    n_features_after: int
    value_range: Optional[Tuple[float, float]] = None  # Added for consistency

class FeatureRangeSelector:
    """
    Selects features within a specified range.
    
    The selector operates in two modes:
    - Value-based: When feature metadata (e.g., q-values) is provided
    - Index-based: When only the number of features is known
    """

    def __init__(self, config: Dict[str, float]):
        """
        Initialize with selection configuration.
        
        Args:
            config: Dictionary containing 'min_value' and 'max_value'
        """
        self.config = config
        self.min_value = config.get('min_value', 0)
        self.max_value = config.get('max_value', float('inf'))

    def select_features(
        self,
        X_train: np.ndarray,
        feature_values: Optional[np.ndarray] = None  # Renamed from feature_metadata
    ) -> Tuple[Dict[str, np.ndarray], FeatureSelectionResult]:
        """
        Select features from data based on configuration.
        
        Args:
            X_train: Array of np.ndarray
            feature_values: Optional array of feature values for value-based selection
            
        Returns:
            Tuple of (selected_indices, selection_result)
        """
        
        n_features = X_train.shape[1]
        
        # Determine selection indices
        if feature_values is not None:
            # Value-based selection
            feature_values = np.round(feature_values, 4)  # Handle numerical precision
            mask = (feature_values >= self.min_value) & (feature_values <= self.max_value)
            selected_indices = np.where(mask)[0]
            selected_values = feature_values[selected_indices]
            value_range = (self.min_value, self.max_value)
        else:
            # Index-based selection
            min_idx = max(0, int(self.min_value))
            max_idx = min(n_features - 1, int(self.max_value))
            selected_indices = np.arange(min_idx, max_idx + 1)
            selected_values = None
            value_range = None

        # Create result
        result = FeatureSelectionResult(
            selected_indices=selected_indices,
            selected_values=selected_values,
            n_features_before=n_features,
            n_features_after=len(selected_indices),  # Fixed: was len(indices)
            value_range=value_range
        )

        return selected_indices, result

    def plot_selection(
        self,
        original_data: Dict[str, np.ndarray],
        filtered_data: Dict[str, np.ndarray],
        result: FeatureSelectionResult,
        feature_metadata: Optional[np.ndarray],
        save_path: Union[str, Path]
    ) -> None:
        """
        Generate feature range selection visualization.
        
        Args:
            original_data: Dictionary of original data arrays
            filtered_data: Dictionary of filtered data arrays
            result: Feature selection result object
            feature_metadata: Optional feature values (e.g., q-values)
            save_path: Path to save the plot
        """        
        save_path = Path(save_path)
        save_path.parent.mkdir(parents=True, exist_ok=True)
        
        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 10))
        
        # Get train data for visualization
        X_train = original_data['train']
        n_samples = X_train.shape[0]
        
        # Find class labels if we have both classes in training
        # Simple heuristic: assume roughly balanced classes
        mid_point = n_samples // 2
        
        # Plot 1: Mean profiles with selected features
        mean_profile = X_train.mean(axis=0)
        std_profile = X_train.std(axis=0)
        
        x_axis = feature_metadata if feature_metadata is not None else np.arange(len(mean_profile))
        
        # Plot full profile
        ax1.plot(x_axis, mean_profile, 'b-', alpha=0.5, label='All features')
        ax1.fill_between(x_axis, mean_profile - std_profile, mean_profile + std_profile, 
                         alpha=0.2, color='blue')
        
        # Highlight selected features
        selected_indices = result.selected_indices
        ax1.plot(x_axis[selected_indices], mean_profile[selected_indices], 'ro', 
                markersize=6, label=f'Selected ({len(selected_indices)} features)')
        
        ax1.set_xlabel('Feature Value' if feature_metadata is not None else 'Feature Index')
        ax1.set_ylabel('Mean Intensity')
        ax1.set_title('Feature Range Selection')
        ax1.legend()
        ax1.grid(True, alpha=0.3)
        
        # Plot 2: Selection range visualization
        if result.value_range:
            min_val, max_val = result.value_range
            ax2.axvspan(min_val, max_val, alpha=0.3, color='green', label='Selected range')
            
        if feature_metadata is not None:
            ax2.hist(feature_metadata, bins=50, alpha=0.5, label='All features', color='gray')
            if result.selected_values is not None:
                ax2.hist(result.selected_values, bins=30, alpha=0.8, label='Selected features', color='green')
        else:
            # Bar plot for index-based selection
            ax2.bar(range(len(mean_profile)), mean_profile, color='gray', alpha=0.5)
            ax2.bar(selected_indices, mean_profile[selected_indices], color='green', alpha=0.8)
        
        ax2.set_xlabel('Feature Value' if feature_metadata is not None else 'Feature Index')
        ax2.set_ylabel('Count' if feature_metadata is not None else 'Mean Intensity')
        ax2.set_title('Selected Feature Distribution')
        ax2.legend()
        
        plt.tight_layout()
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.close()
        
        logger.info(f"Saved feature range selection plot to {save_path}")
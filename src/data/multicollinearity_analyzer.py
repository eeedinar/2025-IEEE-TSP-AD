"""
Multicollinearity Analysis Module
===========================================

Removes highly correlated features while ensuring minimum features are retained.

"""

import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from typing import Tuple, List, Optional, Dict, Any, Union
from dataclasses import dataclass
import logging
from datetime import datetime
try:
    from .feature_range_selector import FeatureSelectionResult
except ImportError:
    from feature_range_selector import FeatureSelectionResult

logger = logging.getLogger(__name__)


class MulticollinearityAnalyzer:
    """
    Analyzes and removes highly correlated features from datasets.
    
    Prioritizes removing features with higher average correlation to others.
    """
    
    def __init__(
        self, 
        correlation_threshold: Union[float, List[float]] = 0.95, 
        min_features: Union[int, List[int]] = 1,
        combination_method: str = 'union'
    ):
        """
        Initialize the analyzer.
        
        Args:
            correlation_threshold: Single value or list of thresholds per array
            min_features: Single value or list of minimum features per array
            combination_method: How to combine results ('union' or 'intersection')
        """
        self.correlation_threshold = correlation_threshold if isinstance(correlation_threshold, list) else [correlation_threshold]
        self.min_features = min_features if isinstance(min_features, list) else [min_features]
        
        if combination_method not in ['union', 'intersection']:
            raise ValueError("combination_method must be 'union' or 'intersection'")
            
        self.combination_method = combination_method
        self.correlation_matrix = None
        self.selected_indices = None
        self.removed_indices = None

    def analyze_and_select_features(
        self, 
        X: Union[np.ndarray, List[np.ndarray]],
        feature_names: Optional[List[str]] = None,
        verbose: bool = True,
        plot: bool = False,
        save_path: Optional[str] = None
    ) -> Union[Tuple[np.ndarray, np.ndarray, np.ndarray], 
               Tuple[Dict[str, np.ndarray], Dict[str, np.ndarray], Dict[str, np.ndarray]]]:
        """
        Select features by removing highly correlated ones.
        
        Returns:
            For single array: (X_filtered, indices, removed)
            For multiple arrays: (X_dict, indices_dict, removed_dict)
        """
        # Prepare arrays and parameters
        arrays = [X] if isinstance(X, np.ndarray) else X
        n_arrays = len(arrays)
        n_features = arrays[0].shape[1]
        
        # Validate dimensions
        for i, arr in enumerate(arrays[1:], 1):
            if arr.shape[1] != n_features:
                raise ValueError(f"Shape mismatch: array {i} has {arr.shape[1]} features, expected {n_features}")
        
        # Extend parameters
        thresholds = self._extend_parameter(self.correlation_threshold, n_arrays)
        min_features_list = self._extend_parameter(self.min_features, n_arrays)
        
        # Single array case
        if n_arrays == 1:
            arr = arrays[0]
            corr_matrix = np.abs(np.corrcoef(arr.T))
            indices = self._select_features(corr_matrix, thresholds[0], min_features_list[0])
            removed = np.setdiff1d(np.arange(n_features), indices)
            
            self.selected_indices = indices
            self.removed_indices = removed
            self.correlation_matrix = np.corrcoef(arr.T)
            
            if verbose:
                logger.info(f"Selected {len(indices)}/{n_features} features (threshold={thresholds[0]})")
                self._print_analysis_summary(n_features, thresholds[0])
            
            if plot:
                base_path = save_path or 'multicollinearity'
                self._plot_correlation_analysis(np.abs(self.correlation_matrix), feature_names, 
                                               f"{base_path}_correlation.png" if save_path else "correlation_matrix.png")
                self._plot_feature_analysis([arr], [indices], feature_names,
                                           f"{base_path}_features.png" if save_path else "feature_analysis.png")
            
            return arr[:, indices], indices, removed
        
        # Multiple arrays case
        results = {}
        for i, arr in enumerate(arrays):
            key = f'array_{i}'
            corr_matrix = np.abs(np.corrcoef(arr.T))
            indices = self._select_features(corr_matrix, thresholds[i], min_features_list[i])
            
            results[key] = {
                'X': arr[:, indices],
                'indices': indices,
                'removed': np.setdiff1d(np.arange(n_features), indices)
            }
            
            if verbose:
                logger.info(f"{key}: {len(indices)}/{n_features} features (threshold={thresholds[i]})")
        
        # Combine indices
        combined_indices = results['array_0']['indices']
        for i in range(1, n_arrays):
            indices = results[f'array_{i}']['indices']
            combined_indices = (np.union1d(combined_indices, indices) if self.combination_method == 'union' 
                               else np.intersect1d(combined_indices, indices))
        
        combined_indices = np.sort(combined_indices)
        
        # Add combined result using stacked arrays
        combined_array = np.vstack(arrays)
        results['combined'] = {
            'X': combined_array[:, combined_indices],
            'indices': combined_indices,
            'removed': np.setdiff1d(np.arange(n_features), combined_indices)
        }
        
        # Update instance state
        self.selected_indices = combined_indices
        self.removed_indices = results['combined']['removed']
        self.correlation_matrix = np.corrcoef(combined_array.T)
        
        if verbose:
            logger.info(f"Combined ({self.combination_method}): {len(combined_indices)}/{n_features} features")
            # Use the first threshold for summary (or could use max/min)
            self._print_analysis_summary(n_features, thresholds[0])
        
        if plot:
            self._generate_plots(arrays, {k: v['indices'] for k, v in results.items()}, feature_names, save_path)

        # Extract dictionaries for return
        return ({k: v['X'] for k, v in results.items()},
                {k: v['indices'] for k, v in results.items()},
                {k: v['removed'] for k, v in results.items()})

    def _select_features(self, corr_matrix: np.ndarray, threshold: float, min_features: int) -> np.ndarray:
        """Select features based on correlation threshold and minimum count."""
        n_features = len(corr_matrix)
        
        # Initialize storage for average correlations
        self.feature_avg_correlations = np.zeros(n_features)

        # Find highly correlated pairs
        high_corr_pairs = []
        for i in range(n_features):
            for j in range(i + 1, n_features):
                if corr_matrix[i, j] > threshold:
                    high_corr_pairs.append((corr_matrix[i, j], i, j))
        
        high_corr_pairs.sort(reverse=True)
        
        # Select features to drop
        to_drop = set()
        for _, i, j in high_corr_pairs:
            if i in to_drop or j in to_drop:
                continue
            
            if n_features - len(to_drop) - 1 >= min_features:
                # Drop feature with higher average correlation
                avg_i = np.mean([corr_matrix[i, k] for k in range(n_features) if k != i and k not in to_drop])
                avg_j = np.mean([corr_matrix[j, k] for k in range(n_features) if k != j and k not in to_drop])
                

                # SAVE the average correlations as they're calculated
                self.feature_avg_correlations[i] = avg_i
                self.feature_avg_correlations[j] = avg_j

                to_drop.add(i if avg_i > avg_j else j)

        # Calculate final averages for all features (for plotting)
        for feature_idx in range(n_features):
            if self.feature_avg_correlations[feature_idx] == 0:  # Not calculated yet
                self.feature_avg_correlations[feature_idx] = np.mean([corr_matrix[feature_idx, k] 
                                                                    for k in range(n_features) 
                                                                    if k != feature_idx and k not in to_drop])

                
        return np.array(sorted([i for i in range(n_features) if i not in to_drop]))

    def _generate_plots(self, arrays: List[np.ndarray], indices_dict: Dict[str, np.ndarray], 
                       feature_names: Optional[List[str]], save_path: Optional[str]):
        """Generate both correlation and feature analysis plots."""
        base_path = save_path or 'multicollinearity'
        
        # Correlation plot
        corr_path = f"{base_path}_correlation.png" if save_path else "correlation_matrix.png"
        self._plot_correlation_analysis(np.abs(self.correlation_matrix), feature_names, corr_path)
        
        # Feature analysis plots - one for each array
        for i, arr in enumerate(arrays):
            # Extract indices for this specific array
            array_indices = [indices_dict[f'array_{i}']]
            array_path = f"{base_path}_array_{i}_features.png" if save_path else f"feature_analysis_array_{i}.png"

            # Use existing _plot_feature_analysis with single array
            self._plot_feature_analysis([arr], array_indices, feature_names, array_path)

        
        # Combined feature analysis plot using all arrays
        if len(arrays) > 1:
            all_indices = [indices_dict[f'array_{i}'] for i in range(len(arrays))]
            feat_path = f"{base_path}_combined_features.png" if save_path else "feature_analysis_combined.png"
            self._plot_feature_analysis(arrays, all_indices, feature_names, feat_path)

        self._plot_correlation_distributions(arrays, feature_names, f"{base_path}_distributions.png" if save_path else "correlation_distributions.png")

    def _extend_parameter(self, param: List[Union[float, int]], n_arrays: int) -> List[Union[float, int]]:
        """Extend parameter list to match number of arrays."""
        if len(param) == 1:
            return param * n_arrays
        elif len(param) == n_arrays:
            return param
        else:
            raise ValueError(f"Parameter list length ({len(param)}) must be 1 or match number of arrays ({n_arrays})")

    def _create_result(
        self, 
        indices: np.ndarray, 
        n_features: int, 
        feature_names: Optional[List[str]]
    ) -> FeatureSelectionResult:
        """Create a FeatureSelectionResult object."""
        selected_names = [feature_names[i] for i in indices] if feature_names else None
        return FeatureSelectionResult(
            selected_indices=indices,
            selected_values=selected_names,
            n_features_before=n_features,
            n_features_after=len(indices),
            value_range=None  # Not applicable for multicollinearity
        )
    
    def _print_analysis_summary(self, n_features: int, threshold: float):
        """Print summary of multicollinearity analysis."""
        logger.info(f"\nMulticollinearity Analysis Summary:")
        logger.info(f"  Original features: {n_features}")
        logger.info(f"  Correlation threshold: {threshold:.2f}")
        logger.info(f"  Features removed: {len(self.removed_indices)}")
        logger.info(f"  Features retained: {len(self.selected_indices)}")
        
        if self.correlation_matrix is not None:
            # Calculate correlation statistics
            upper_triangle = np.triu(self.correlation_matrix, k=1)
            correlations = upper_triangle[upper_triangle != 0]
            
            logger.info(f"\nCorrelation Statistics:")
            logger.info(f"  Mean |correlation|: {np.mean(np.abs(correlations)):.3f}")
            logger.info(f"  Max |correlation|: {np.max(np.abs(correlations)):.3f}")
            logger.info(f"  Correlations > 0.9: {np.sum(np.abs(correlations) > 0.9)}")
            logger.info(f"  Correlations > 0.95: {np.sum(np.abs(correlations) > 0.95)}")
    
    def _plot_correlation_analysis(
        self, 
        corr_matrix: np.ndarray, 
        feature_names: Optional[List[str]] = None,
        save_path: str = 'correlation_matrix.png'
    ):
        """Generate correlation heatmap visualization."""
        plt.figure(figsize=(12, 10))
        
        # Create mask for selected features
        mask = np.ones_like(corr_matrix, dtype=bool)
        mask[np.ix_(self.selected_indices, self.selected_indices)] = False
        
        # Plot correlation matrix
        sns.heatmap(
            corr_matrix,
            mask=mask,
            cmap='coolwarm',
            center=0,
            vmin=-1,
            vmax=1,
            square=True,
            linewidths=0.5,
            cbar_kws={"shrink": 0.8, "label": "Correlation"},
            annot=False
        )
        
        # Highlight selected features
        ax = plt.gca()
        for idx in self.selected_indices:
            ax.axhline(y=idx, color='green', linewidth=2, alpha=0.3)
            ax.axvline(x=idx, color='green', linewidth=2, alpha=0.3)
        
        plt.title(f'Feature Correlation Matrix\n'
                 f'Green lines: Selected features ({len(self.selected_indices)})',
                 fontsize=14, fontweight='bold')
        
        if feature_names:
            plt.xticks(range(len(feature_names)), feature_names, rotation=90)
            plt.yticks(range(len(feature_names)), feature_names)
        
        plt.tight_layout()
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        logger.info(f"Correlation plot saved to {save_path}")
        plt.close()
    
    def _plot_feature_analysis(
        self,
        arrays: List[np.ndarray],
        all_indices: List[np.ndarray],
        feature_names: Optional[List[str]] = None,
        save_path: str = 'feature_analysis.png'
    ):
        """Plot mean intensity with selected features for arrays."""
        plt.figure(figsize=(12, 6))
        
        # Prepare x-axis
        n_features = arrays[0].shape[1]
        x_labels, x_label_text = self._prepare_x_axis(n_features, feature_names)
        
        # Set up colors and labels
        n_arrays = len(arrays)
        if n_arrays == 1:
            colors = ['blue']
            labels = ['Data']
        elif n_arrays == 2:
            colors = ['blue', 'red']
            labels = ['Class 0', 'Class 1']
        else:
            colors = plt.cm.tab10(np.linspace(0, 1, n_arrays))
            labels = [f'Array {i}' for i in range(n_arrays)]
        
        # Plot each array
        for i, (arr, indices) in enumerate(zip(arrays, all_indices)):
            color = colors[i]
            label = labels[i]
            
            # Compute and plot statistics
            mean = arr.mean(axis=0)
            std = arr.std(axis=0)
            
            plt.plot(x_labels, mean, '-', color=color, label=f"{label} Mean", linewidth=2)
            plt.fill_between(x_labels, mean - std, mean + std, color=color, alpha=0.2)
            
            # Highlight selected features
            plt.scatter(x_labels[indices], mean[indices], color=color, s=80, 
                       marker='o', edgecolor='black', linewidth=1, 
                       label=f"Selected ({label})")
        
        # Show combined selection if multiple arrays
        if n_arrays > 1 and hasattr(self, 'selected_indices'):
            mean_combined = np.mean([arr.mean(axis=0) for arr in arrays], axis=0)
            plt.scatter(x_labels[self.selected_indices], mean_combined[self.selected_indices], 
                       color='green', s=100, marker='o',
                       label=f"Combined Selection ({len(self.selected_indices)})", 
                       edgecolor='black', linewidth=1)
        
        # Formatting
        title = f"Feature Intensity Analysis: {labels[0]}" if n_arrays == 1 else "Feature Intensity Analysis with Selected Features"
        plt.title(title, fontsize=14)
        plt.xlabel(x_label_text, fontsize=12)
        plt.ylabel("Intensity", fontsize=12)
        plt.legend()
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        logger.info(f"Feature analysis plot saved to {save_path}")
        plt.close()

    ### the count has nothing to do with _select_features() -> high_corr_pairs -- just for statistical reporting can be used
    def _plot_correlation_distributions(self, arrays: List[np.ndarray], feature_names: Optional[List[str]] = None,
                                           save_path: str = 'correlation_distributions'):
        """Plot correlation analysis and save statistics - optimized single loop."""
        
        if self.correlation_matrix is None:
            return
        
        # Prepare x-axis
        n_features = arrays[0].shape[1]
        x_labels, x_label_text = self._prepare_x_axis(n_features, feature_names)
        
        # Setup
        fig, (ax1, ax2, ax3) = plt.subplots(3, 1, figsize=(20, 20), dpi=400)
        colors = ['blue', 'red']
        labels = [f'Class {i}' for i in range(len(arrays))]

        # Configurable thresholds (hardcoded)
        correlation_thresholds = [
            0.950, 0.955, 0.960, 0.965, 0.970, 0.975, 0.980, 0.985,         # coarse grid
            0.987, 0.988, 0.989, 0.990, 0.991, 0.992, 0.993, 0.994, 0.995,  # fine grid
            0.999
        ]

        # Initialize comprehensive stats
        total_pairs = (n_features * (n_features - 1)) // 2
        stats = [
            "CORRELATION ANALYSIS SUMMARY",
            "=" * 60,
            f"Date: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}",
            f"Arrays: {len(arrays)}, Features: {n_features}",
            f"Total correlation pairs per array: {total_pairs:,}",
            f"Correlation threshold used: {self.correlation_threshold[0]:.3f}",
            f"Features selected: {len(self.selected_indices) if self.selected_indices is not None else 'N/A'}",
            f"Features removed: {len(self.removed_indices) if self.removed_indices is not None else 'N/A'}",
            ""
        ]
        # Storage for threshold analysis
        threshold_data = []
            
        # SINGLE LOOP: Bottom plot + Statistics together
        for i, arr in enumerate(arrays):
            # Compute correlation matrix once
            corr = np.corrcoef(arr.T)

            # Call _select_features - this calculates and saves average correlations
            selected_indices = self._select_features(np.abs(corr), self.correlation_threshold[0], self.min_features[0])
        
            # Top: Feature-wise average correlation
            ax1.plot(x_labels, self.feature_avg_correlations, '-', linewidth=2, color=colors[i], 
                    label=f'{labels[i]} Avg |Corr|')        

            # Bottom plot: Add histogram
            triu_indices = np.triu_indices_from(corr, k=1)
            correlations = corr[triu_indices]
            abs_corr = np.abs(correlations)
            
            # Bottom plot: Add histogram
            ax2.hist(correlations, bins=30, alpha=0.6, density=True, color=colors[i],
                    label=f'{labels[i]} Correlations')
            
            # Collect threshold data for additional plot
            threshold_counts = []
            for threshold in correlation_thresholds:
                count = np.sum(abs_corr > threshold)
                threshold_counts.append(count)
            threshold_data.append(threshold_counts)


           # Comprehensive Statistics for this array
            stats.extend([
                f"{labels[i].upper()} ({arr.shape[0]} samples):",
                "-" * 40,
                f"  Correlation Statistics:",
                f"    Mean correlation: {np.mean(abs_corr):.4f}",
                f"    Std correlation:  {np.std(abs_corr):.4f}",
                f"    Min correlation:  {np.min(abs_corr):.4f}",
                f"    Max correlation:  {np.max(abs_corr):.4f}",
                "",
                f"  Correlation Thresholds:",
            ])
            
            # Add threshold statistics using the list
            for threshold in correlation_thresholds:
                count = np.sum(abs_corr > threshold)
                percentage = count / total_pairs * 100
                stats.append(f"    |Corr| > {threshold}: {count:,} pairs ({percentage:.1f}%)")
            
            stats.extend([
                "",
                f"  Feature-wise Analysis (from _select_features):",
                f"    Mean feature avg |corr|: {np.mean(self.feature_avg_correlations):.4f}",
                f"    Most correlated feature: {np.argmax(self.feature_avg_correlations)} (avg: {np.max(self.feature_avg_correlations):.4f})",
                f"    Least correlated feature: {np.argmin(self.feature_avg_correlations)} (avg: {np.min(self.feature_avg_correlations):.4f})",
                ""
            ])
            
            # Add q-value info if available
            if feature_names is not None and len(x_labels) == n_features:
                most_corr_idx = np.argmax(np.mean(np.abs(corr), axis=1))
                least_corr_idx = np.argmin(np.mean(np.abs(corr), axis=1))
                stats.extend([
                    f"  Q-value Analysis:",
                    f"    Most correlated q-value: {x_labels[most_corr_idx]:.4f} Å⁻¹",
                    f"    Least correlated q-value: {x_labels[least_corr_idx]:.4f} Å⁻¹",
                    ""
                ])
        
        # Finish top plot
        ax1.set_xlabel(x_label_text)
        ax1.set_ylabel('Average |Correlation|')
        ax1.set_title('Feature-wise Average Correlation')
        ax1.legend()
        ax1.grid(True, alpha=0.3)
        
        # Finish bottom plot
        ax2.set_xlabel('Correlation Value')
        ax2.set_ylabel('Density')
        ax2.set_title('Correlation Value Distributions')
        ax2.legend()
        ax2.grid(True, alpha=0.3)
        
        # Number of pairs vs threshold
        ax3.plot(correlation_thresholds, threshold_data[0], 'o-', color='blue', label=f'{labels[0]}', linewidth=2)
        if len(arrays) > 1:
            ax3.plot(correlation_thresholds, threshold_data[1], 's-', color='red', label=f'{labels[1]}', linewidth=2)
        ax3.set_xlabel('Correlation Threshold')
        ax3.set_ylabel('Number of Pairs Above Threshold')
        ax3.set_title('High Correlation Pairs vs Threshold')
        ax3.legend()
        ax3.grid(True)

        plt.tight_layout()
        plt.savefig(f'{save_path}', dpi=300, bbox_inches='tight')
        plt.close()
        
        # Save stats
        with open(f'{save_path}_stats.txt', 'w') as f:
            f.write('\n'.join(stats))
        
        logger.info(f"Correlation analysis saved: {save_path}")

    def _prepare_x_axis(self, n_features: int, feature_names: Optional[List[str]] = None) -> Tuple[np.ndarray, str]:
        """Prepare x-axis labels and determine appropriate label text."""
        x = np.arange(n_features)
        
        if feature_names is not None:
            try:
                # Try to convert to float array (q-values)
                q_values = np.array([float(name) for name in feature_names])
                return q_values, "q-value (Å⁻¹)"
            except (ValueError, TypeError):
                # If conversion fails, they're actual names
                return x, "Feature"
        
        return x, "Feature Index"
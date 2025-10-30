"""
Visualization Utilities
=======================

Core plotting functionality for the ML framework.

"""

import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from typing import List, Optional, Tuple, Union, Dict, Any
from pathlib import Path


class Plotter:
    """Handles core plotting functionality."""
    
    def __init__(self):
        """Initialize plotter with default settings."""
        # Try to use modern seaborn style
        try:
            plt.style.use('seaborn-v0_8-darkgrid')
        except:
            try:
                plt.style.use('seaborn-darkgrid')
            except:
                plt.style.use('default')
        
        # Set default figure parameters
        plt.rcParams['figure.figsize'] = (10, 6)
        plt.rcParams['font.size'] = 10
        plt.rcParams['axes.labelsize'] = 12
        plt.rcParams['axes.titlesize'] = 14
        plt.rcParams['xtick.labelsize'] = 10
        plt.rcParams['ytick.labelsize'] = 10
        plt.rcParams['legend.fontsize'] = 10
        
        # Color palettes
        self.colors = {
            'primary': '#1f77b4',
            'secondary': '#ff7f0e',
            'success': '#2ca02c',
            'danger': '#d62728',
            'warning': '#ff9800',
            'info': '#17a2b8',
            'light': '#f0f0f0',
            'dark': '#333333'
        }
    
    def save_and_close(self, save_path: Optional[Union[str, Path]] = None) -> None:
        """Save figure and close it."""
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight', facecolor='white')
        plt.close()
    
    def plot_confusion_matrix(self, 
                            cm: np.ndarray, 
                            labels: Optional[List[str]] = None,
                            title: str = 'Confusion Matrix',
                            save_path: Optional[Union[str, Path]] = None) -> None:
        """Plot confusion matrix heatmap."""
        if labels is None:
            labels = ['Class 0', 'Class 1']
        
        plt.figure(figsize=(8, 6))
        
        # Calculate percentages
        cm_normalized = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
        
        # Create annotations with both count and percentage
        annotations = np.empty_like(cm).astype(str)
        for i in range(cm.shape[0]):
            for j in range(cm.shape[1]):
                annotations[i, j] = f'{cm[i, j]}\n({cm_normalized[i, j]:.1%})'
        
        # Create heatmap
        sns.heatmap(cm, annot=annotations, fmt='', cmap='Blues', 
                    xticklabels=labels, yticklabels=labels,
                    cbar_kws={'label': 'Count'}, square=True)
        
        plt.title(title, fontsize=16, fontweight='bold', pad=20)
        plt.ylabel('True Label', fontsize=12)
        plt.xlabel('Predicted Label', fontsize=12)
        plt.tight_layout()
        
        self.save_and_close(save_path)
    
    def plot_metrics_bar(self,
                        metrics: Dict[str, float],
                        title: str = 'Model Performance',
                        save_path: Optional[Union[str, Path]] = None) -> None:
        """Plot metrics as a bar chart."""
        if not metrics:
            return
        
        plt.figure(figsize=(10, 6))
        
        # Extract metrics and values
        names = list(metrics.keys())
        values = list(metrics.values())
        
        # Create bars
        x_pos = np.arange(len(names))
        bars = plt.bar(x_pos, values, color=self.colors['primary'], 
                       edgecolor='black', linewidth=1.2)
        
        # Color bars based on performance
        for bar, value in zip(bars, values):
            if value >= 0.9:
                bar.set_facecolor(self.colors['success'])
            elif value >= 0.8:
                bar.set_facecolor(self.colors['info'])
            elif value >= 0.7:
                bar.set_facecolor(self.colors['warning'])
            else:
                bar.set_facecolor(self.colors['danger'])
        
        # Customize plot
        plt.xlabel('Metrics', fontsize=12, fontweight='bold')
        plt.ylabel('Score', fontsize=12, fontweight='bold')
        plt.title(title, fontsize=16, fontweight='bold', pad=20)
        plt.xticks(x_pos, names, rotation=45, ha='right')
        plt.ylim(0, 1.1)
        plt.grid(True, alpha=0.3, axis='y')
        
        # Add value labels
        for bar, value in zip(bars, values):
            plt.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01,
                    f'{value:.3f}', ha='center', va='bottom', fontsize=10, fontweight='bold')
        
        plt.tight_layout()
        self.save_and_close(save_path)
    
    def plot_learning_curves(self,
                           train_scores: List[float],
                           val_scores: List[float],
                           metric_name: str = 'Loss',
                           title: str = 'Learning Curves',
                           save_path: Optional[Union[str, Path]] = None) -> None:
        """Plot training and validation learning curves."""
        epochs = range(1, len(train_scores) + 1)
        
        plt.figure(figsize=(10, 6))
        
        # Plot curves
        plt.plot(epochs, train_scores, 'b-', label=f'Training {metric_name}', 
                 linewidth=2.5, marker='o', markersize=4)
        plt.plot(epochs, val_scores, 'r-', label=f'Validation {metric_name}', 
                 linewidth=2.5, marker='s', markersize=4)
        
        # Fill between curves to show gap
        plt.fill_between(epochs, train_scores, val_scores, alpha=0.2, color='gray')
        
        plt.xlabel('Epoch', fontsize=12, fontweight='bold')
        plt.ylabel(metric_name, fontsize=12, fontweight='bold')
        plt.title(title, fontsize=16, fontweight='bold', pad=20)
        plt.legend(loc='best', frameon=True, shadow=True)
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        
        self.save_and_close(save_path)
    
    def plot_feature_importance(self,
                              importances: np.ndarray,
                              feature_names: Optional[List[str]] = None,
                              top_n: int = 20,
                              title: str = 'Feature Importance',
                              save_path: Optional[Union[str, Path]] = None) -> None:
        """Plot feature importance scores."""
        # Sort features by importance
        indices = np.argsort(importances)[::-1][:top_n]
        
        if feature_names is None:
            feature_names = [f'Feature {i}' for i in range(len(importances))]
        
        # Create figure
        fig_height = max(8, top_n * 0.3)
        plt.figure(figsize=(10, fig_height))
        
        # Create horizontal bar plot
        y_pos = np.arange(len(indices))
        plt.barh(y_pos, importances[indices], 
                 color=self.colors['primary'], 
                 edgecolor='black', linewidth=0.5)
        
        # Customize plot
        plt.yticks(y_pos, [feature_names[i] for i in indices])
        plt.xlabel('Importance Score', fontsize=12, fontweight='bold')
        plt.title(f'{title} - Top {top_n} Features', fontsize=16, fontweight='bold', pad=20)
        plt.grid(True, alpha=0.3, axis='x')
        plt.gca().set_axisbelow(True)
        
        # Add value labels
        for i, (idx, importance) in enumerate(zip(indices, importances[indices])):
            plt.text(importance + max(importances) * 0.01, i, f'{importance:.3f}', 
                    va='center', fontsize=9)
        
        plt.tight_layout()
        self.save_and_close(save_path)
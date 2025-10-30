"""
Report Generator
================

Generates comprehensive ML experiment reports and visualizations.

"""

import logging
from pathlib import Path
from typing import Dict, List, Any, Optional
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import roc_curve, auc, precision_recall_curve  # Still need these for curve generation
import sys


# Add parent directory to path
sys.path.append(str(Path(__file__).parent.parent))

from ..metrics import MetricsWrapper
logger = logging.getLogger(__name__)
from .plotter import Plotter


class ReportGenerator:
    """Generate comprehensive experiment reports and visualizations."""
    
    def __init__(self, config: Dict[str, Any]):
        """Initialize report generator with configuration."""
        self.config = config
        self.viz_config = config.get('visualization', {})
        self.plotter = Plotter()
        
        # Set style
        plt.style.use('seaborn-v0_8-darkgrid')
        sns.set_palette("husl")
    
    def generate_all_reports(self, 
                           results: Dict[str, Dict],
                           trained_models: Dict[str, Any],
                           data_splits: Dict[str, np.ndarray],
                           preprocessing_state: Any,
                           output_dir: Path) -> None:
        """Generate all configured reports and visualizations."""
        if not self.viz_config.get('create_report', False):
            return
        
        logger.info("\n" + "="*60)
        logger.info("GENERATING VISUALIZATIONS")
        logger.info("="*60)
        
        plots_dir = output_dir / 'plots'
        plots_dir.mkdir(parents=True, exist_ok=True)
        
        # Configure plot settings
        plot_format = self.viz_config.get('plot_format', 'png')
        dpi = self.viz_config.get('dpi', 300)
        
        # Get requested plots
        plots_to_generate = self.viz_config.get('plots', [
            'model_comparison', 'confusion_matrix', 'roc_curve', 
            'precision_recall_curve', 'threshold_analysis'
        ])
        
        plot_count = 0
        
        # Generate each type of plot
        if 'model_comparison' in plots_to_generate and len(trained_models) > 1:
            self.plot_model_comparison(results, plots_dir, plot_format, dpi)
            plot_count += 1
        
        if 'confusion_matrix' in plots_to_generate:
            self.plot_confusion_matrices(results, plots_dir, plot_format, dpi)
            plot_count += 1
        
        if 'roc_curve' in plots_to_generate:
            self.plot_roc_curves(trained_models, data_splits, plots_dir, plot_format, dpi)
            plot_count += 1
        
        if 'precision_recall_curve' in plots_to_generate:
            self.plot_precision_recall_curves(trained_models, data_splits, plots_dir, plot_format, dpi)
            plot_count += 1
        
        if 'threshold_analysis' in plots_to_generate:
            self.plot_threshold_analysis(trained_models, results, data_splits, plots_dir, plot_format, dpi)
            plot_count += 1
        
        # Always generate these summary plots
        self.plot_performance_by_category(results, trained_models, plots_dir, plot_format, dpi)
        plot_count += 1
        
        self.plot_preprocessing_summary(preprocessing_state, plots_dir, plot_format, dpi)
        plot_count += 1
        
        logger.info(f"  ✓ Generated {plot_count} plots")
    
    def plot_model_comparison(self, results: Dict[str, Dict], plots_dir: Path, 
                            format: str, dpi: int) -> None:
        """Create model comparison plots."""
        # Filter successful models
        successful_results = {k: v for k, v in results.items() if 'error' not in v}
        
        if not successful_results:
            logger.warning("No successful models to plot")
            return
        
        results_df = pd.DataFrame(successful_results).T
        
        # Ensure numeric types
        numeric_columns = ['train_f1', 'val_f1', 'test_f1', 'test_accuracy', 
                          'test_precision', 'test_recall', 'test_auc', 
                          'training_time', 'total_parameters']
        
        for col in numeric_columns:
            if col in results_df.columns:
                results_df[col] = pd.to_numeric(results_df[col], errors='coerce')
        
        results_df = results_df.dropna(subset=['val_f1', 'test_f1'])
        
        if results_df.empty:
            logger.warning("No valid results to plot after filtering")
            return
        
        # Create subplots
        fig, axes = plt.subplots(2, 2, figsize=(15, 12))
        fig.suptitle('Model Performance Comparison', fontsize=16, fontweight='bold')
        
        # 1. F1 Score Comparison
        self._plot_f1_comparison(axes[0, 0], results_df)
        
        # 2. Multiple Metrics
        self._plot_metrics_comparison(axes[0, 1], results_df)
        
        # 3. Training Time vs Performance
        self._plot_time_vs_performance(axes[1, 0], results_df)
        
        # 4. Parameters vs Performance
        self._plot_parameters_vs_performance(axes[1, 1], results_df)
        
        plt.tight_layout()
        plt.savefig(plots_dir / f'model_comparison.{format}', dpi=dpi, bbox_inches='tight')
        plt.close()
        
        logger.info("  ✓ Generated model comparison plots")
    
    def _plot_f1_comparison(self, ax, results_df: pd.DataFrame) -> None:
        """Plot F1 score comparison."""
        metrics = ['train_f1', 'val_f1', 'test_f1']
        available = [m for m in metrics if m in results_df.columns]
        
        if available:
            data = results_df[available].sort_values('val_f1', ascending=False)
            data.plot(kind='bar', ax=ax)
            ax.set_title('F1 Scores by Dataset Split')
            ax.set_ylabel('F1 Score')
            ax.legend(['Train', 'Validation', 'Test'][:len(available)])
            ax.set_xticklabels(ax.get_xticklabels(), rotation=45, ha='right')
    
    def _plot_metrics_comparison(self, ax, results_df: pd.DataFrame) -> None:
        """Plot multiple metrics comparison."""
        metrics = ['test_accuracy', 'test_precision', 'test_recall', 'test_f1']
        available = [m for m in metrics if m in results_df.columns]
        
        if available:
            data = results_df[available].sort_values('test_f1', ascending=False)
            data.plot(kind='bar', ax=ax)
            ax.set_title('Test Set Performance Metrics')
            ax.set_ylabel('Score')
            ax.legend(['Accuracy', 'Precision', 'Recall', 'F1'][:len(available)])
            ax.set_xticklabels(ax.get_xticklabels(), rotation=45, ha='right')
    
    def _plot_time_vs_performance(self, ax, results_df: pd.DataFrame) -> None:
        """Plot training time vs performance."""
        if 'training_time' in results_df.columns and 'test_f1' in results_df.columns:
            ax.scatter(results_df['training_time'], results_df['test_f1'])
            for idx, model in enumerate(results_df.index):
                ax.annotate(model, 
                           (results_df.loc[model, 'training_time'], 
                            results_df.loc[model, 'test_f1']),
                           fontsize=8, rotation=15)
            ax.set_xlabel('Training Time (seconds)')
            ax.set_ylabel('Test F1 Score')
            ax.set_title('Training Time vs Performance')
            ax.grid(True, alpha=0.3)
    
    def _plot_parameters_vs_performance(self, ax, results_df: pd.DataFrame) -> None:
        """Plot model parameters vs performance."""
        if 'total_parameters' in results_df.columns:
            valid_params = results_df[
                (results_df['total_parameters'] > 0) & 
                (results_df['total_parameters'].notna())
            ]
            
            if not valid_params.empty:
                ax.scatter(valid_params['total_parameters'], valid_params['test_f1'])
                for idx, model in enumerate(valid_params.index):
                    ax.annotate(model, 
                               (valid_params.loc[model, 'total_parameters'], 
                                valid_params.loc[model, 'test_f1']),
                               fontsize=8, rotation=15)
                ax.set_xlabel('Total Parameters')
                ax.set_ylabel('Test F1 Score')
                ax.set_title('Model Complexity vs Performance')
                ax.set_xscale('log')
                ax.grid(True, alpha=0.3)
            else:
                self._plot_top_models(ax, results_df)
        else:
            self._plot_top_models(ax, results_df)
    
    def _plot_top_models(self, ax, results_df: pd.DataFrame) -> None:
        """Plot top models by validation F1."""
        top_n = min(10, len(results_df))
        if 'val_f1' in results_df.columns and 'test_f1' in results_df.columns:
            top_models = results_df.nlargest(top_n, 'val_f1')[['val_f1', 'test_f1']]
            top_models.plot(kind='barh', ax=ax)
            ax.set_title(f'Top {top_n} Models by Validation F1')
            ax.set_xlabel('F1 Score')
            ax.legend(['Validation', 'Test'])
    
    def _get_model_probabilities(self, model, model_name: str, X: np.ndarray, 
                            X_nmf: Optional[np.ndarray] = None) -> Optional[np.ndarray]:
        """
        Get probability predictions from a model, handling NMF models appropriately.
        
        Args:
            model: The model to get predictions from
            model_name: Name of the model (for logging)
            X: Input features
            X_nmf: NMF features (optional)
        
        Returns:
            Probability array or None if prediction fails
        """
        try:
            if hasattr(model, 'uses_nmf') and model.uses_nmf and X_nmf is not None:
                return model.predict_proba(X, X_nmf=X_nmf)
            else:
                return model.predict_proba(X)
        except Exception as e:
            logger.warning(f"Could not get probabilities for {model_name}: {str(e)}")
            return None

    def plot_confusion_matrices(self, results: Dict[str, Dict], plots_dir: Path,
                               format: str, dpi: int) -> None:
        """Plot confusion matrices for all models."""
        models_with_cm = [(name, res['confusion_matrix']) 
                         for name, res in results.items() 
                         if 'confusion_matrix' in res]
        
        if not models_with_cm:
            return
        
        n_models = len(models_with_cm)
        cols = min(3, n_models)
        rows = (n_models + cols - 1) // cols
        
        fig, axes = plt.subplots(rows, cols, figsize=(6*cols, 5*rows))
        if n_models == 1:
            axes = [axes]
        else:
            axes = axes.flatten()
        
        for idx, (model_name, cm) in enumerate(models_with_cm):
            cm_array = np.array(cm)
            sns.heatmap(cm_array, annot=True, fmt='d', cmap='Blues', 
                       ax=axes[idx], cbar=idx==0)
            axes[idx].set_title(model_name)
            axes[idx].set_xlabel('Predicted')
            axes[idx].set_ylabel('Actual')
        
        # Hide unused subplots
        for i in range(len(models_with_cm), len(axes)):
            axes[i].set_visible(False)
        
        plt.suptitle('Confusion Matrices (Test Set)', fontsize=16, fontweight='bold')
        plt.tight_layout()
        plt.savefig(plots_dir / f'confusion_matrices.{format}', dpi=dpi, bbox_inches='tight')
        plt.close()
        
        logger.info("  ✓ Generated confusion matrices")
    
    def plot_roc_curves(self, trained_models: Dict[str, Any], 
                       data_splits: Dict[str, np.ndarray],
                       plots_dir: Path, format: str, dpi: int) -> None:
        """Plot ROC curves for models with probability predictions."""
        X_test = data_splits['X_test']
        y_test = data_splits['y_test']
        X_test_nmf = data_splits.get('X_test_nmf')

        plt.figure(figsize=(10, 8))
        
        has_curves = False

        for model_name, model in trained_models.items():
            if hasattr(model, 'predict_proba'):
                y_proba = self._get_model_probabilities(model, model_name, X_test, X_test_nmf)
                if y_proba is not None:
                    # Generate ROC curve points
                    fpr, tpr, _ = roc_curve(y_test, y_proba[:, 1])
                    
                    # Use MetricsWrapper to compute AUC
                    roc_auc = MetricsWrapper.get_eval_metrics(
                        metrics_names='auc',
                        y_true=y_test,
                        y_pred=y_proba
                    )
                    
                    if roc_auc is not None and not np.isnan(roc_auc):
                        plt.plot(fpr, tpr, label=f'{model_name} (AUC = {roc_auc:.3f})')
                        has_curves = True
        
        if has_curves:
            plt.plot([0, 1], [0, 1], 'k--', label='Random Classifier')
            plt.xlim([0.0, 1.0])
            plt.ylim([0.0, 1.05])
            plt.xlabel('False Positive Rate')
            plt.ylabel('True Positive Rate')
            plt.title('ROC Curves', fontsize=16, fontweight='bold')
            plt.legend(loc="lower right")
            plt.grid(True, alpha=0.3)
            
            plt.tight_layout()
            plt.savefig(plots_dir / f'roc_curves.{format}', dpi=dpi, bbox_inches='tight')
            plt.close()
            
            logger.info("  ✓ Generated ROC curves")
        else:
            plt.close()
    
    def plot_precision_recall_curves(self, trained_models: Dict[str, Any],
                                   data_splits: Dict[str, np.ndarray],
                                   plots_dir: Path, format: str, dpi: int) -> None:
        """Plot precision-recall curves."""
        X_test = data_splits['X_test']
        y_test = data_splits['y_test']
        X_test_nmf = data_splits.get('X_test_nmf')

        plt.figure(figsize=(10, 8))
        
        has_curves = False    
        for model_name, model in trained_models.items():
            if hasattr(model, 'predict_proba'):
                y_proba = self._get_model_probabilities(model, model_name, X_test, X_test_nmf)
                if y_proba is not None:
                    # Generate PR curve points
                    precision, recall, _ = precision_recall_curve(y_test, y_proba[:, 1])
                    
                    # Use MetricsWrapper to compute AUPRC
                    ap = MetricsWrapper.get_eval_metrics(
                        metrics_names='auprc',
                        y_true=y_test,
                        y_pred=y_proba
                    )
                    
                    if ap is not None and not np.isnan(ap):
                        plt.plot(recall, precision, label=f'{model_name} (AP = {ap:.3f})')
                        has_curves = True
        if has_curves:
            plt.xlim([0.0, 1.0])
            plt.ylim([0.0, 1.05])
            plt.xlabel('Recall')
            plt.ylabel('Precision')
            plt.title('Precision-Recall Curves', fontsize=16, fontweight='bold')
            plt.legend(loc="lower left")
            plt.grid(True, alpha=0.3)
            
            plt.tight_layout()
            plt.savefig(plots_dir / f'precision_recall_curves.{format}', dpi=dpi, bbox_inches='tight')
            plt.close()
            
            logger.info("  ✓ Generated precision-recall curves")
        else:
            plt.close()
    
    def plot_threshold_analysis(self, trained_models: Dict[str, Any],
                               results: Dict[str, Dict],
                               data_splits: Dict[str, np.ndarray],
                               plots_dir: Path, format: str, dpi: int) -> None:
        """Plot threshold analysis for models with probability output."""
        from sklearn.metrics import f1_score, precision_score, recall_score
        
        X_val = data_splits['X_val']
        y_val = data_splits['y_val']
        X_val_nmf = data_splits.get('X_val_nmf')  # Get NMF features if available

        if X_val is None or y_val is None:
            logger.warning("No validation data available for threshold analysis")
            return

        models_with_proba = [(name, model) for name, model in trained_models.items() 
                            if hasattr(model, 'predict_proba')]
        
        if not models_with_proba:
            return
        
        thresholds = np.linspace(0.1, 0.9, 17)
        
        fig, axes = plt.subplots(2, 2, figsize=(15, 12))
        
        # Plot metrics vs threshold for top 5 models
        top_models = models_with_proba[:5]
        
        # Helper function to plot metric vs threshold
        def plot_metric_threshold(ax, metric_name: str, ylabel: str, title: str):
            for model_name, model in top_models:
                y_proba = self._get_model_probabilities(model, model_name, X_val, X_val_nmf)
                if y_proba is not None:
                    # Use MetricsWrapper to compute scores at different thresholds
                    scores = []
                    for t in thresholds:
                        score = MetricsWrapper.get_eval_metrics(
                            metrics_names=metric_name,
                            y_true=y_val,
                            y_pred=y_proba,
                            prob_thr=t
                        )
                        scores.append(score if score is not None else 0)
                    
                    ax.plot(thresholds, scores, label=model_name)
            
            ax.set_xlabel('Threshold')
            ax.set_ylabel(ylabel)
            ax.set_title(title)
            ax.legend()
            ax.grid(True, alpha=0.3)

        # F1 vs threshold
        plot_metric_threshold(axes[0, 0], 'f1', 'F1 Score', 'F1 Score vs Decision Threshold')
        
        # Precision vs threshold
        plot_metric_threshold(axes[0, 1], 'precision', 'Precision', 'Precision vs Decision Threshold')
        
        # Recall vs threshold
        plot_metric_threshold(axes[1, 0], 'recall', 'Recall', 'Recall vs Decision Threshold')
            
        # Optimal thresholds
        ax = axes[1, 1]
        optimal_data = [(name, res['optimal_threshold']) 
                       for name, res in results.items() 
                       if 'optimal_threshold' in res and res['optimal_threshold'] != 0.5]
        
        if optimal_data:
            names, opt_thresholds = zip(*optimal_data)
            ax.barh(range(len(names)), opt_thresholds)
            ax.set_yticks(range(len(names)))
            ax.set_yticklabels(names)
            ax.set_xlabel('Optimal Threshold')
            ax.set_title('Optimal Thresholds by Model')
            ax.axvline(x=0.5, color='r', linestyle='--', label='Default (0.5)')
            ax.legend()
        
        plt.suptitle('Decision Threshold Analysis', fontsize=16, fontweight='bold')
        plt.tight_layout()
        plt.savefig(plots_dir / f'threshold_analysis.{format}', dpi=dpi, bbox_inches='tight')
        plt.close()
        
        logger.info("  ✓ Generated threshold analysis plots")
    
    def plot_performance_by_category(self, results: Dict[str, Dict],
                                   trained_models: Dict[str, Any],
                                   plots_dir: Path, format: str, dpi: int) -> None:
        """Plot performance grouped by model category."""
        # Define model categories
        category_mapping = {
            'logistic_regression': 'Classical',
            'svm': 'Classical',
            'naive_bayes': 'Classical',
            'knn': 'Classical',
            'neural_network': 'Neural Networks',
            'mlp': 'Neural Networks',
            'transformer': 'Deep Learning',
            'gnn': 'Deep Learning'
        }
        
        # Create performance data
        perf_data = []
        for model_name, metrics in results.items():
            if 'error' not in metrics:
                # Determine category
                category = 'Other'
                for key, cat in category_mapping.items():
                    if key in model_name.lower():
                        category = cat
                        break
                
                perf_data.append({
                    'Model': model_name,
                    'Category': category,
                    'Val F1': metrics.get('val_f1', 0),
                    'Test F1': metrics.get('test_f1', 0),
                    'Test Accuracy': metrics.get('test_accuracy', 0)
                })
        
        if not perf_data:
            return
        
        df = pd.DataFrame(perf_data)
        
        # Create plots
        fig, axes = plt.subplots(1, 2, figsize=(16, 6))
        
        # Box plot
        ax = axes[0]
        df_melted = df.melt(id_vars=['Model', 'Category'], 
                           value_vars=['Val F1', 'Test F1'],
                           var_name='Metric', value_name='Score')
        sns.boxplot(data=df_melted, x='Category', y='Score', hue='Metric', ax=ax)
        ax.set_title('Performance Distribution by Model Category')
        ax.set_xticklabels(ax.get_xticklabels(), rotation=45, ha='right')
        
        # Average performance
        ax = axes[1]
        category_avg = df.groupby('Category')[['Val F1', 'Test F1', 'Test Accuracy']].mean()
        category_avg.plot(kind='bar', ax=ax)
        ax.set_title('Average Performance by Model Category')
        ax.set_ylabel('Score')
        ax.legend(['Val F1', 'Test F1', 'Test Accuracy'])
        ax.set_xticklabels(ax.get_xticklabels(), rotation=45, ha='right')
        
        plt.suptitle('Model Performance by Category', fontsize=16, fontweight='bold')
        plt.tight_layout()
        plt.savefig(plots_dir / f'performance_by_category.{format}', dpi=dpi, bbox_inches='tight')
        plt.close()
        
        logger.info("  ✓ Generated category performance analysis")
    
    def plot_preprocessing_summary(self, preprocessing_state: Any,
                                 plots_dir: Path, format: str, dpi: int) -> None:
        """Create preprocessing pipeline summary visualization."""
        import matplotlib.patches as mpatches
        
        fig, ax = plt.subplots(figsize=(12, 8))
        
        stages = preprocessing_state.stages
        n_stages = len(stages)
        
        if n_stages == 0:
            ax.text(0.5, 0.5, 'No preprocessing stages applied', 
                   ha='center', va='center', fontsize=16)
            ax.axis('off')
            plt.savefig(plots_dir / f'preprocessing_summary.{format}', dpi=dpi)
            plt.close()
            return
        
        # Layout parameters
        y_positions = np.linspace(0.8, 0.2, n_stages)
        box_width = 0.25
        box_height = 0.08
        
        # Color scheme
        stage_colors = {
            'feature_range': '#FF6B6B',
            'multicollinearity': '#4ECDC4',
            'nmf': '#45B7D1',
            'scaling': '#96CEB4'
        }
        
        # Draw pipeline
        for i, stage in enumerate(stages):
            y = y_positions[i]
            
            # Draw box
            color = stage_colors.get(stage['name'], '#95A5A6')
            box = mpatches.FancyBboxPatch(
                (0.35, y - box_height/2), box_width, box_height,
                boxstyle="round,pad=0.01",
                facecolor=color, edgecolor='black', linewidth=2
            )
            ax.add_patch(box)
            
            # Add text
            ax.text(0.475, y, stage['name'].replace('_', ' ').title(),
                   ha='center', va='center', fontweight='bold', fontsize=10)
            
            ax.text(0.25, y, f"{stage['n_features_in']} features",
                   ha='right', va='center', fontsize=9)
            ax.text(0.65, y, f"{stage['n_features_out']} features",
                   ha='left', va='center', fontsize=9)
            
            # Draw arrow
            if i < n_stages - 1:
                ax.arrow(0.475, y - box_height/2 - 0.01, 0, 
                        -(y_positions[i] - y_positions[i+1] - box_height - 0.02),
                        head_width=0.02, head_length=0.01, fc='black', ec='black')
            
            # Add reduction percentage
            if stage['n_features_in'] > 0:
                reduction = (1 - stage['n_features_out'] / stage['n_features_in']) * 100
                if reduction > 0:
                    ax.text(0.72, y, f"(-{reduction:.1f}%)",
                           ha='left', va='center', fontsize=9, color='red')
        
        # Add title and summary
        ax.text(0.5, 0.95, 'Preprocessing Pipeline Summary', 
               ha='center', va='top', fontsize=16, fontweight='bold')
        
        ax.text(0.475, 0.9, f"Original Features: {stages[0]['n_features_in']}",
               ha='center', va='center', fontsize=11)
        
        final_features = stages[-1]['n_features_out']
        total_reduction = ((1 - final_features / stages[0]['n_features_in']) * 100 
                          if stages[0]['n_features_in'] > 0 else 0)
        
        ax.text(0.475, 0.1, f"Final Features: {final_features} (Total reduction: {total_reduction:.1f}%)",
               ha='center', va='center', fontsize=11, fontweight='bold')
        
        # Clean up axes
        ax.set_xlim(0, 1)
        ax.set_ylim(0, 1)
        ax.axis('off')
        
        plt.tight_layout()
        plt.savefig(plots_dir / f'preprocessing_summary.{format}', dpi=dpi, bbox_inches='tight')
        plt.close()
        
        logger.info("  ✓ Generated preprocessing summary plot")
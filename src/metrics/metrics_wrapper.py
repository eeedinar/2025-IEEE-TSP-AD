import numpy as np
from typing import Union, List, Dict
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score, 
    roc_auc_score, matthews_corrcoef, balanced_accuracy_score, 
    average_precision_score, cohen_kappa_score, log_loss
)

class MetricsWrapper:
    """
    A metrics wrapper for classification evaluation.
    
    Provides unified interface for computing various classification metrics,
    automatically handling both predictions and probabilities.
    """
    
    # Define once, use everywhere
    METRICS = {
        # Standard prediction-based metrics
        'accuracy': accuracy_score,
        'f1': lambda y_t, y_p: f1_score(y_t, y_p, average='weighted', zero_division=0),
        'f1_macro': lambda y_t, y_p: f1_score(y_t, y_p, average='macro', zero_division=0),
        'f1_micro': lambda y_t, y_p: f1_score(y_t, y_p, average='micro', zero_division=0),
        'precision': lambda y_t, y_p: precision_score(y_t, y_p, average='weighted', zero_division=0),
        'precision_macro': lambda y_t, y_p: precision_score(y_t, y_p, average='macro', zero_division=0),
        'precision_micro': lambda y_t, y_p: precision_score(y_t, y_p, average='micro', zero_division=0),
        'recall': lambda y_t, y_p: recall_score(y_t, y_p, average='weighted', zero_division=0),
        'recall_macro': lambda y_t, y_p: recall_score(y_t, y_p, average='macro', zero_division=0),
        'recall_micro': lambda y_t, y_p: recall_score(y_t, y_p, average='micro', zero_division=0),
        'balanced_accuracy': balanced_accuracy_score,
        'matthews_corrcoef': matthews_corrcoef,
        'cohen_kappa': cohen_kappa_score,
        
        # ROC AUC variants
        'auc': roc_auc_score,  # Standard ROC AUC
        'auc_weighted': lambda y_t, y_p: roc_auc_score(y_t, y_p, average='weighted', multi_class='ovr'),
        'auc_macro': lambda y_t, y_p: roc_auc_score(y_t, y_p, average='macro', multi_class='ovr'),
        'auc_micro': lambda y_t, y_p: roc_auc_score(y_t, y_p, average='micro', multi_class='ovr'),
        'auc_ovr': lambda y_t, y_p: roc_auc_score(y_t, y_p, multi_class='ovr'),
        'auc_ovo': lambda y_t, y_p: roc_auc_score(y_t, y_p, multi_class='ovo'),
        
        # Precision-Recall AUC variants (AUPRC)
        'auprc': average_precision_score,  # Standard AUPRC
        'auprc_weighted': lambda y_t, y_p: average_precision_score(y_t, y_p, average='weighted'),
        'auprc_macro': lambda y_t, y_p: average_precision_score(y_t, y_p, average='macro'),
        'auprc_micro': lambda y_t, y_p: average_precision_score(y_t, y_p, average='micro'),
        
        # Loss metrics
        'log_loss': log_loss,
    }
    
    PROB_METRICS = {
        'auc', 'auc_weighted', 'auc_macro', 'auc_micro', 'auc_ovr', 'auc_ovo',
        'auprc', 'auprc_weighted', 'auprc_macro', 'auprc_micro',
        'log_loss'
    }
    
    @staticmethod
    def get_eval_metrics(metrics_names: Union[str, List[str]] = None, 
                        y_true=None, y_pred=None, prob_thr: float = 0.5) -> Union[Dict, float, None]:
        """
        Get metric functions dict or compute scores.
        
        Args:
            metrics_names: Metric name(s) to compute. If None, returns all metrics.
            y_true: True labels. If None, returns metric functions dict.
            y_pred: Predictions or probabilities. Required if y_true provided.
            
        Returns:
            Dict of metric functions or computed scores.
            
        Available metrics:
            Standard: accuracy, f1, precision, recall, balanced_accuracy, matthews_corrcoef, cohen_kappa
            ROC AUC: auc, roc_auc, auc_weighted, auc_macro, auc_micro, auc_ovr, auc_ovo
            PR AUC: auprc, average_precision, pr_auc, auprc_weighted, auprc_macro, auprc_micro
            Loss: log_loss
            
        Examples:
            >>> # Get all metric functions
            >>> metrics = MetricsWrapper.get_eval_metrics()
            
            >>> # Compute all scores
            >>> scores = MetricsWrapper.get_eval_metrics(y_true=y_true, y_pred=y_pred)
            
            >>> # Compute specific AUC metrics
            >>> auc_scores = MetricsWrapper.get_eval_metrics(['auc', 'auprc', 'auc_macro'], y_true, y_pred)
        """

        if y_true is None:
            return None
        
        # Select metrics
        is_single_metric = isinstance(metrics_names, str)
        
        if metrics_names is None:
            selected = MetricsWrapper.METRICS
        elif is_single_metric:
            if metrics_names not in MetricsWrapper.METRICS:
                raise ValueError(f"Metric '{metrics_names}' not found. Available metrics: {list(MetricsWrapper.METRICS.keys())}")
            selected = {metrics_names: MetricsWrapper.METRICS[metrics_names]}
        else:
            selected = {}
            for name in metrics_names:
                if name not in MetricsWrapper.METRICS:
                    raise ValueError(f"Metric '{name}' not found. Available metrics: {list(MetricsWrapper.METRICS.keys())}")
                selected[name] = MetricsWrapper.METRICS[name]
        
        # Detect probabilities
        is_proba = (
            len(y_pred.shape) > 1 and y_pred.shape[1] > 1 or  # 2D array with multiple columns
            (len(y_pred.shape) == 1 and np.any((y_pred > 0) & (y_pred < 1)) and not np.all(np.isin(y_pred, [0, 1])))  # 1D probabilities (not just 0/1)
        )
        
        results = {}
        for name, func in selected.items():
            try:
                if name in MetricsWrapper.PROB_METRICS:
                    # Probability-based metrics
                    if is_proba:
                        if name == 'log_loss':
                            # log_loss needs full probability matrix
                            results[name] = func(y_true, y_pred)
                        elif len(y_pred.shape) > 1 and y_pred.shape[1] > 1:
                            # Check if metric function accepts multi-class parameters
                            try:
                                # Try with full probability matrix for multi-class metrics
                                if any(param in str(func) for param in ['multi_class', 'average']):
                                    results[name] = func(y_true, y_pred)
                                else:
                                    # Binary probability metrics - extract positive class
                                    y_p = y_pred[:, 1]
                                    results[name] = func(y_true, y_p)
                            except:
                                # Fallback to binary probability
                                y_p = y_pred[:, 1]
                                results[name] = func(y_true, y_p)
                        else:
                            # 1D probabilities
                            results[name] = func(y_true, y_pred)
                    else:
                        # No probabilities available for probability metrics
                        results[name] = np.nan
                else:
                    # Standard prediction-based metrics
                    if is_proba:
                        # Convert probabilities to predictions
                        if len(y_pred.shape) > 1 and y_pred.shape[1] > 1:
                            # 2D probability matrix - use positive class probabilities
                            y_proba_pos = y_pred[:, 1]
                            y_p = (y_proba_pos > prob_thr).astype(int)
                        else:
                            # 1D probabilities
                            y_p = (y_pred > prob_thr).astype(int)
                    else:
                        # Already predictions
                        y_p = y_pred.astype(int) if y_pred.dtype != int else y_pred
                    
                    results[name] = func(y_true, y_p)
                    
            except Exception as e:
                # For debugging uncomment: print(f"Error computing {name}: {e}")
                results[name] = np.nan
        
        # Return single value if single metric requested
        if is_single_metric:
            return list(results.values())[0]
        
        return results
"""
Model Analysis Utilities
========================

Provides parameter counting and model analysis functionality.

"""

import numpy as np
from typing import Any, Dict


class ModelAnalyzer:
    """Analyze model complexity and parameters."""
    
    @staticmethod
    def count_parameters(model: Any) -> int:
        """
        Count total parameters in any model type.
        
        Args:
            model: The model to analyze
            
        Returns:
            Total number of parameters
        """
        # Try PyTorch models
        try:
            return ModelAnalyzer._count_pytorch_parameters(model)
        except:
            pass
        
        # Try TensorFlow/Keras models
        try:
            return ModelAnalyzer._count_tensorflow_parameters(model)
        except:
            pass
        
        # Try scikit-learn models
        try:
            return ModelAnalyzer._count_sklearn_parameters(model)
        except:
            pass
        
        # Try custom models with count_parameters method
        if hasattr(model, 'count_parameters'):
            return model.count_parameters()
        
        # If model has a wrapped model attribute
        if hasattr(model, 'model'):
            return ModelAnalyzer.count_parameters(model.model)
        
        return 0
    
    @staticmethod
    def _count_pytorch_parameters(model: Any) -> int:
        """Count parameters in PyTorch models."""
        # Check if it's a PyTorch model
        if hasattr(model, 'parameters'):
            return sum(p.numel() for p in model.parameters())
        
        # Check for wrapped PyTorch models
        if hasattr(model, 'model') and hasattr(model.model, 'parameters'):
            return sum(p.numel() for p in model.model.parameters())
        
        # Check for network attribute
        if hasattr(model, 'network') and hasattr(model.network, 'parameters'):
            return sum(p.numel() for p in model.network.parameters())
        
        raise AttributeError("Not a PyTorch model")
    
    @staticmethod
    def _count_tensorflow_parameters(model: Any) -> int:
        """Count parameters in TensorFlow/Keras models."""
        if hasattr(model, 'count_params'):
            return model.count_params()
        
        if hasattr(model, 'model') and hasattr(model.model, 'count_params'):
            return model.model.count_params()
        
        raise AttributeError("Not a TensorFlow/Keras model")
    
    @staticmethod
    def _count_sklearn_parameters(model: Any) -> int:
        """Count parameters in scikit-learn models."""
        param_count = 0
        
        # Get the actual sklearn model if wrapped
        sklearn_model = model
        if hasattr(model, 'model'):
            sklearn_model = model.model
        elif hasattr(model, 'estimator'):
            sklearn_model = model.estimator
        
        # Linear models (LogisticRegression, LinearRegression, SVM, etc.)
        if hasattr(sklearn_model, 'coef_'):
            coef = sklearn_model.coef_
            if hasattr(coef, 'size'):
                param_count += coef.size
            elif hasattr(coef, 'shape'):
                param_count += np.prod(coef.shape)
            
            if hasattr(sklearn_model, 'intercept_'):
                intercept = sklearn_model.intercept_
                if hasattr(intercept, 'size'):
                    param_count += intercept.size
                elif isinstance(intercept, (int, float)):
                    param_count += 1
        
        # Tree-based models
        elif hasattr(sklearn_model, 'tree_'):
            # Single decision tree
            param_count += sklearn_model.tree_.node_count
        
        # Ensemble models
        elif hasattr(sklearn_model, 'estimators_'):
            # Random Forest, AdaBoost, etc.
            for estimator in sklearn_model.estimators_:
                if hasattr(estimator, 'tree_'):
                    param_count += estimator.tree_.node_count
                else:
                    param_count += ModelAnalyzer._count_sklearn_parameters(estimator)
        
        # KNN models (no trainable parameters, but store number of samples)
        elif hasattr(sklearn_model, '_fit_X'):
            param_count = sklearn_model._fit_X.shape[0] * sklearn_model._fit_X.shape[1]
        
        # SVM with support vectors
        elif hasattr(sklearn_model, 'support_vectors_'):
            param_count = sklearn_model.support_vectors_.size
        
        # Naive Bayes
        elif hasattr(sklearn_model, 'theta_'):
            param_count += sklearn_model.theta_.size
            if hasattr(sklearn_model, 'sigma_'):
                param_count += sklearn_model.sigma_.size
        
        return param_count
    
    @staticmethod
    def get_model_info(model: Any) -> Dict[str, Any]:
        """
        Get comprehensive model information.
        
        Args:
            model: The model to analyze
            
        Returns:
            Dictionary with model information
        """
        info = {
            'parameters': ModelAnalyzer.count_parameters(model),
            'type': type(model).__name__,
            'framework': ModelAnalyzer._detect_framework(model)
        }
        
        # Add additional info based on model type
        if hasattr(model, 'get_params'):
            info['hyperparameters'] = model.get_params()
        
        return info
    
    @staticmethod
    def _detect_framework(model: Any) -> str:
        """Detect the framework of the model."""
        model_str = str(type(model))
        
        if 'torch' in model_str or hasattr(model, 'parameters'):
            return 'pytorch'
        elif 'tensorflow' in model_str or 'keras' in model_str:
            return 'tensorflow'
        elif 'sklearn' in model_str or hasattr(model, 'get_params'):
            return 'sklearn'
        else:
            return 'custom'
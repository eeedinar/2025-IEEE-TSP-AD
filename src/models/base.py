"""Base model interface and factory classes.

This module provides the foundation for all machine learning models in the framework.
It includes abstract base classes, factory patterns for component creation, and
utility functions for safe type conversion.

Key Components:
    - BaseModel: Abstract base class for all models
    - ModelFactory: Factory for model creation and registration
    - LossFactory: Factory for creating loss functions
    - OptimizerFactory: Factory for creating optimizers
    - SchedulerFactory: Factory for creating learning rate schedulers
    - EarlyStopping: Early stopping mechanism for training
"""

from abc import ABC, abstractmethod
import numpy as np
from typing import Any, Dict, Optional, Union, List
from loguru import logger
from pathlib import Path
import joblib
import torch
import torch.nn as nn
import torch.optim as optim
import copy

def safe_int(value: Any, default: int) -> int:
    """
    Safely convert value to integer.
    
    Handles various input types including strings with scientific notation,
    floats, and None values. Returns default if conversion fails.
    
    Args:
        value: Value to convert
        default: Default value if conversion fails
        
    Returns:
        Converted integer value or default
    """
    if value is None:
        return default
    try:
        return int(float(str(value)))
    except:
        return default

def safe_float(value: Any, default: float) -> float:
    """
    Safely convert value to float.
    
    Handles various input types including strings with scientific notation,
    integers, and None values. Returns default if conversion fails.
    
    Args:
        value: Value to convert
        default: Default value if conversion fails
        
    Returns:
        Converted float value or default
    """
    if value is None:
        return default
    try:
        return float(str(value))
    except:
        return default

class BaseModel(ABC):
    """
    Abstract base class for all machine learning models.
    
    Provides a consistent interface for model training, prediction, and persistence.
    All models in the framework must inherit from this class and implement the
    abstract methods.
    
    Attributes:
        config: Configuration dictionary for the model
        model: The underlying model implementation
        fitted: Whether the model has been trained
        model_name: Name of the model class
    """
    
    def __init__(self, config: Optional[Dict[str, Any]] = None):
        """
        Initialize base model.
        
        Args:
            config: Model configuration dictionary containing hyperparameters
                   and settings specific to each model type
        """
        self.config = config or {}
        self.model = None
        self.fitted = False
        self.model_name = self.__class__.__name__
        
    @abstractmethod
    def train(self, X: np.ndarray, y: np.ndarray) -> None:
        """
        Train the model on the provided data.
        
        This method must be implemented by all subclasses to define
        model-specific training procedures.
        
        Args:
            X: Training features of shape (n_samples, n_features)
            y: Training labels of shape (n_samples,)
        """
        pass
    
    @abstractmethod
    def predict(self, X: np.ndarray) -> np.ndarray:
        """
        Generate predictions for the provided features.
        
        This method must be implemented by all subclasses to define
        model-specific prediction procedures.
        
        Args:
            X: Features of shape (n_samples, n_features)
            
        Returns:
            Predicted labels of shape (n_samples,)
        """
        pass
    
    def predict_proba(self, X: np.ndarray) -> Optional[np.ndarray]:
        """
        Predict class probabilities if supported by the model.
        
        Not all models support probability prediction. Returns None
        if the model doesn't have a predict_proba method.
        
        Args:
            X: Features of shape (n_samples, n_features)
            
        Returns:
            Class probabilities of shape (n_samples, n_classes) or None
        """
        if hasattr(self.model, 'predict_proba'):
            return self.model.predict_proba(X)
        return None

    def save(self, filepath: str) -> None:
        """
        Save model to disk.
        
        Handles both standard scikit-learn models and PyTorch models.
        PyTorch models are saved with separate files for metadata (.joblib)
        and weights (.pth).
        
        Args:
            filepath: Path where the model should be saved
            
        Raises:
            ValueError: If attempting to save an unfitted model
        """
        if not self.fitted:
            raise ValueError("Cannot save unfitted model")
        
        filepath = Path(filepath).with_suffix('.joblib')
        metadata = {'model_name': self.model_name, 'config': self.config, 'fitted' : self.fitted}
        
        # Check for PyTorch
        if hasattr(self.model, 'state_dict'):
            joblib.dump(metadata, filepath)
            
            torch_state = {}
            for name, obj in vars(self).items():
                if hasattr(obj, 'state_dict'):
                    torch_state[name] = obj.state_dict()
            torch.save(torch_state, filepath.with_suffix('.pth'))
            logger.info(f"Saved {self.model_name}: {filepath} + .pth")
            return

        # otherwise save sklearn model
        metadata['model'] = self.model
        joblib.dump(metadata, filepath)
        logger.info(f"Saved {self.model_name}: {filepath}")
    
    def load(self, filepath: str) -> None:
        """
        Load model from disk.
        
        Handles both standard scikit-learn models and PyTorch models.
        Automatically detects the model type and loads appropriately.
        
        Args:
            filepath: Path to the saved model file
        """
        filepath = Path(filepath).with_suffix('.joblib')
        metadata = joblib.load(filepath)
        
        self.config = metadata.get('config', {})
        self.model_name = metadata.get('model_name', self.__class__.__name__)

        if not metadata.get('fitted', False):
            raise ValueError("Cannot load unfitted model")
        
        # Sklearn model
        if 'model' in metadata:
            self.model = metadata['model']
        else:
            # PyTorch: weights in separate file
            
            if not filepath.with_suffix('.pth').exists():
                logger.info("PyTorch model - reconstruction required")
                raise FileNotFoundError(f"Missing weights: {filepath.with_suffix('.pth')}")

            torch_state = torch.load(filepath.with_suffix('.pth'), map_location='cpu')
            # Load each component
            for name, state_dict in torch_state.items():
                if hasattr(self, name):
                    obj = getattr(self, name)
                    if hasattr(obj, 'load_state_dict'):
                        obj.load_state_dict(state_dict)

        logger.info(f"Loaded {self.model_name}: {filepath}")

    # def optimize_threshold(
    #     self, 
    #     X: np.ndarray, 
    #     y: np.ndarray,
    #     metric: str = 'f1'
    # ) -> float:
    #     """
    #     Find optimal classification threshold.
        
    #     Args:
    #         X: Validation features
    #         y: Validation labels
    #         metric: Metric to optimize ('f1', 'accuracy', 'precision', 'recall')
            
    #     Returns:
    #         Optimal threshold
    #     """
    #     from sklearn.metrics import f1_score, accuracy_score, precision_score, recall_score
        
    #     probas = self.predict_proba(X)
    #     if probas is None:
    #         logger.warning(f"{self.model_name} does not support probability predictions")
    #         return 0.5
        
    #     # Get probabilities for positive class
    #     y_proba = probas[:, 1]
        
    #     # Define metric function
    #     metric_funcs = {
    #         'f1': f1_score,
    #         'accuracy': accuracy_score,
    #         'precision': lambda y_true, y_pred: precision_score(y_true, y_pred, zero_division=0),
    #         'recall': lambda y_true, y_pred: recall_score(y_true, y_pred, zero_division=0)
    #     }
        
    #     if metric not in metric_funcs:
    #         raise ValueError(f"Unknown metric: {metric}")
        
    #     metric_func = metric_funcs[metric]
        
    #     # Try different thresholds
    #     best_threshold = 0.5
    #     best_score = 0
        
    #     for threshold in np.arange(0.1, 0.95, 0.05):
    #         y_pred = (y_proba >= threshold).astype(int)
    #         score = metric_func(y, y_pred)
            
    #         if score > best_score:
    #             best_score = score
    #             best_threshold = threshold
        
    #     logger.info(
    #         f"Optimal threshold for {self.model_name}: {best_threshold:.2f} "
    #         f"(best {metric}: {best_score:.4f})"
    #     )
        
    #     return best_threshold

# ============================================================================
# FACTORY CLASSES
# ============================================================================

class ModelFactory:
    """
    Factory class for creating and managing model instances.
    
    Implements the factory pattern to provide a centralized way to create
    models by name. Models must be registered before they can be created.
    
    This design allows for easy extension of the framework with new models
    without modifying existing code.
    """
    
    _models = {}

    @classmethod
    def register_model(cls, name: str, model_class: type) -> None:
        """
        Register a model class with the factory.
        
        Once registered, the model can be created by name using create_model.
        
        Args:
            name: Name to register the model under
            model_class: Model class that inherits from BaseModel
        """
        cls._models[name] = model_class

    @classmethod
    def create_model(cls, name: str, config: Optional[Dict[str, Any]] = None, **kwargs) -> BaseModel:
        """
        Create a model instance by name.

        Args:
            name: Registered name of the model
            config: Configuration dictionary for the model
            **kwargs: Additional keyword arguments merged into config
            
        Returns:
            Instantiated model object
            
        Raises:
            ValueError: If the model name is not registered
        """
        if name not in cls._models:
            raise ValueError(f"Unknown model: {name}")
        
        full_config = {**(config or {}), **kwargs}
        return cls._models[name](config=full_config)

    @classmethod
    def list_models(cls) -> list:
        """
        Get list of all registered model names.

        Returns:
            List of registered model names
        """
        return list(cls._models.keys())

    @classmethod
    def resolve_model_name(cls, model_name: str) -> str:
        """
        Resolve model name to registered name.
        
        Attempts to find exact or partial matches for the given model name.
        This allows for some flexibility in model naming.

        Args:
            model_name: Name to resolve
            
        Returns:
            Registered model name or None if no match found
        """
        if model_name in cls._models:
            return model_name

        # Find partial match
        for registered in cls._models:
            if model_name.startswith(registered) or registered in model_name:
                return registered

        return None

class LossFactory:
    """Factory for creating PyTorch loss functions."""
    
    LOSSES = {
        'CrossEntropyLoss': nn.CrossEntropyLoss,
        'MSELoss': nn.MSELoss,
        'L1Loss': nn.L1Loss,
        'SmoothL1Loss': nn.SmoothL1Loss,
        'BCEWithLogitsLoss': nn.BCEWithLogitsLoss,
        'NLLLoss': nn.NLLLoss,
    }
    
    # Custom losses with import paths
    CUSTOM_LOSSES = {
        'WeightedCrossEntropyLoss': ('custom_losses', 'WeightedCrossEntropyLoss'),
        'BalancedCrossEntropyLoss': ('custom_losses', 'BalancedCrossEntropyLoss'),
        'DiceLoss': ('custom_losses', 'DiceLoss'),
        'UnifiedFocalLoss': ('custom_losses', 'UnifiedFocalLoss'),
        'CombinedLoss': ('custom_losses', 'CombinedLoss'),
        'CenterLoss': ('center_loss', 'CenterLoss'),
        'ContrastiveCenterLoss': ('center_loss', 'ContrastiveCenterLoss'),
        'RingLoss': ('center_loss', 'RingLoss'),
    }
    
    # Losses that use features instead of outputs
    FEATURE_LOSSES = ['CenterLoss', 'ContrastiveCenterLoss', 'RingLoss']

    # Parameter type conversions
    CONVERTERS = {
        'float': lambda v: safe_float(v, 0.0),
        'str': str,
        'int': lambda v: safe_int(v, 0),
        'bool': bool,
        'tensor': lambda v: torch.tensor(v, dtype=torch.float32) if isinstance(v, (list, tuple)) else v,
        'special': lambda v: v,  # Keep as-is for alpha
    }

    # Parameter type mappings
    PARAM_TYPES = {
        **{k: 'float' for k in ['gamma', 'label_smoothing', 'margin', 'radius', 
                               'eps', 'smooth', 'beta', 'temperature', 'scale', 'lambda_c', 'ring_weight']},
        **{k: 'tensor' for k in ['weight', 'pos_weight']},
        **{k: 'int' for k in ['ignore_index', 'blank', 'num_classes', 'feat_dim']},
        **{k: 'bool' for k in ['binary']},
        'alpha': 'special',
        'reduction': 'str'
    }

    @classmethod
    def create_loss(cls, loss_config: Union[str, Dict[str, Any]],
                           num_classes: int = None,
                           feat_dim: int = None) -> nn.Module:
        """Create a single loss function."""
        
        # Parse config
        if not loss_config:
            return
        name = loss_config if isinstance(loss_config, str) else loss_config.get('type', 'CrossEntropyLoss')
        params = {} if isinstance(loss_config, str) else {k: v for k, v in loss_config.items() if k != 'type'}
        
        # Handle combined loss specially
        if name == 'CombinedLoss':
            losses = []
            for loss_def in params.get('losses', []):
                losses.append(cls.create_loss(loss_def, num_classes, feat_dim))
            
            weights = params.get('weights', None)
            try:
                from src.losses.custom_losses import CombinedLoss
                return cls._create_feature_aware_combined_loss(losses, weights)
            except ImportError:
                logger.warning("CombinedLoss not found, using CrossEntropyLoss")
                return nn.CrossEntropyLoss()
        
        # Convert parameters
        converted = {}
        for key, value in params.items():
            if value is not None:
                param_type = cls.PARAM_TYPES.get(key, 'float')  # ['auto', 'inverse'] not in PARAM_TYPES so use str
                converted[key] = cls.CONVERTERS[param_type](value)
        
        # Add special parameters for center losses
        if name in ['CenterLoss', 'ContrastiveCenterLoss']:
            converted['num_classes'] = safe_int(num_classes, 2)
            converted['feat_dim'] = safe_int(feat_dim, None)
        
        # Create loss
        loss_class = cls._get_loss_class(name)
        if loss_class:
            try:
                return loss_class(**converted)
            except Exception as e:
                logger.warning(f"Failed to create {name}: {e}")
        else:
            logger.warning(f"NO {name} found in the config or loss factory")
            return None
    
    @classmethod
    def _get_loss_class(cls, name: str):
        """Get loss class by name."""
        if name in cls.LOSSES:
            return cls.LOSSES[name]
        elif name in cls.CUSTOM_LOSSES:
            module_name, class_name = cls.CUSTOM_LOSSES[name]
            try:
                module = __import__(f'src.losses.{module_name}', fromlist=[class_name])
                return getattr(module, class_name)
            except (ImportError, AttributeError) as e:
                logger.warning(f"Could not import {class_name}: {e}")
        return None

    @classmethod
    def _create_feature_aware_combined_loss(cls, losses: List[nn.Module], weights: Optional[List[float]]):
        """Create a CombinedLoss wrapper that handles feature routing."""
        from src.losses.custom_losses import CombinedLoss
        
        # Create wrapper that knows about feature routing
        class FeatureAwareCombinedLoss(CombinedLoss):
            def forward(self, outputs: torch.Tensor, targets: torch.Tensor, 
                       features: Optional[torch.Tensor] = None) -> torch.Tensor:
                """Calculate combined loss with automatic feature routing."""
                total_loss = 0
                
                for loss_fn, weight in zip(self.losses, self.weights):
                    loss_name = loss_fn.__class__.__name__
                    
                    # Route to appropriate inputs based on loss type
                    if loss_name in cls.FEATURE_LOSSES and features is not None:
                        if loss_name == 'RingLoss':
                            loss_val = loss_fn(features)
                        else:  # CenterLoss, ContrastiveCenterLoss
                            loss_val = loss_fn(features, targets)
                    else:
                        loss_val = loss_fn(outputs, targets)
                    
                    total_loss += weight * loss_val
                
                return total_loss
        
        return FeatureAwareCombinedLoss(losses, weights)

# ============================================================================
# Optimizer Factory
# ============================================================================
class OptimizerFactory:
    """Factory for creating PyTorch optimizers."""
    
    OPTIMIZERS = {
        'Adam': optim.Adam,
        'AdamW': optim.AdamW,
        'SGD': optim.SGD,
        'RMSprop': optim.RMSprop,
        'Adagrad': optim.Adagrad,
        'Adadelta': optim.Adadelta,
        'Adamax': optim.Adamax,
    }
    
    # Parameter type conversions
    CONVERTERS = {
        'float': lambda v: safe_float(v, 0.001),
        'bool': bool,
        'tuple_float': lambda v: tuple(safe_float(x, 0.9) for x in (v if isinstance(v, (list, tuple)) else [v]))
    }
    
    # Parameter type mappings
    PARAM_TYPES = {
        **{k: 'float' for k in ['lr', 'weight_decay', 'momentum', 'eps', 'alpha', 'rho', 
                                'lr_decay', 'initial_accumulator_value', 'dampening']},
        **{k: 'bool' for k in ['amsgrad', 'centered', 'nesterov', 'maximize', 'foreach', 
                               'capturable', 'differentiable', 'fused']},
        'betas': 'tuple_float'
    }
    
    @classmethod
    def create_optimizer(cls, optimizer_config: Union[str, Dict[str, Any]], 
                        parameters, learning_rate: float = None) -> optim.Optimizer:
        """Create optimizer from config."""
        # Parse config
        name = optimizer_config if isinstance(optimizer_config, str) else optimizer_config.get('type', 'Adam')
        params = {} if isinstance(optimizer_config, str) else {k: v for k, v in optimizer_config.items() if k != 'type'}
        
        # Set learning rate
        if learning_rate is not None:
            params['lr'] = learning_rate
        elif 'lr' not in params:
            params['lr'] = 0.001
            logger.warning(f"Lerning rate is not found in config, setting to {params['lr']}")
        
        # Get optimizer class
        optimizer_class = cls.OPTIMIZERS.get(name, optim.Adam)
        
        # Convert parameters
        converted = {}
        for key, value in params.items():
            if value is not None:
                param_type = cls.PARAM_TYPES.get(key, 'float')
                converted[key] = cls.CONVERTERS[param_type](value)
        
        # Create optimizer
        try:
            return optimizer_class(parameters, **converted)
        except Exception as e:
            logger.warning(f"Failed to create optimizer {name}: {e}")
            return optim.Adam(parameters, lr=0.001)  # Fallback

# ============================================================================
# Scheduler Factory
# ============================================================================

class SchedulerFactory:
    """Factory for creating PyTorch learning rate schedulers."""

    SCHEDULERS = {
        'StepLR': optim.lr_scheduler.StepLR,
        'MultiStepLR': optim.lr_scheduler.MultiStepLR,
        'ExponentialLR': optim.lr_scheduler.ExponentialLR,
        'CosineAnnealingLR': optim.lr_scheduler.CosineAnnealingLR,
        'ReduceLROnPlateau': optim.lr_scheduler.ReduceLROnPlateau,
        'CyclicLR': optim.lr_scheduler.CyclicLR,
        'OneCycleLR': optim.lr_scheduler.OneCycleLR,
        'CosineAnnealingWarmRestarts': optim.lr_scheduler.CosineAnnealingWarmRestarts,
        'LambdaLR': optim.lr_scheduler.LambdaLR,
        'MultiplicativeLR': optim.lr_scheduler.MultiplicativeLR,
        'PolynomialLR': optim.lr_scheduler.PolynomialLR,
        'LinearLR': optim.lr_scheduler.LinearLR,
        'ConstantLR': optim.lr_scheduler.ConstantLR,
        'SequentialLR': optim.lr_scheduler.SequentialLR,
        'ChainedScheduler': optim.lr_scheduler.ChainedScheduler,
    }

    # Parameter type conversions
    CONVERTERS = {
        'int': lambda v: safe_int(v, 10),
        'float': lambda v: safe_float(v, 0.1),
        'str': str,
        'bool': bool,
        'list_int': lambda v: [safe_int(x, 10) for x in (v if isinstance(v, list) else [v])]
    }

    # Parameter type mappings
    PARAM_TYPES = {
        **{k: 'int' for k in ['step_size', 'T_max', 'patience', 'cooldown', 'T_0', 'T_mult', 'epochs', 'total_steps']},
        **{k: 'float' for k in ['gamma', 'eta_min', 'factor', 'threshold', 'min_lr', 'eps', 'base_lr', 'max_lr']},
        **{k: 'str' for k in ['mode', 'anneal_strategy', 'scale_mode', 'threshold_mode']},
        **{k: 'bool' for k in ['verbose', 'cycle_momentum', 'scale_fn', 'three_phase']},
        'milestones': 'list_int'
    }

    @classmethod
    def create_scheduler(cls, scheduler_config: Union[str, Dict[str, Any]], 
                        optimizer: optim.Optimizer) -> Optional[object]:
        
        logger.info(f"SchedulerFactory received: {scheduler_config}")


        """Create scheduler from config."""
        if not scheduler_config or scheduler_config == 'null':
            return None
        
        # Parse config
        name = scheduler_config if isinstance(scheduler_config, str) else scheduler_config.get('type', '')    
        params = {} if isinstance(scheduler_config, str) else {k: v for k, v in scheduler_config.items() if k != 'type'}
        logger.info(f"Scheduler name: {name} and params: {params}")

        # Get scheduler class
        scheduler_class = cls.SCHEDULERS.get(name)
        logger.info(f"Scheduler class: {scheduler_class}")

        if not scheduler_class:
            return None
        
        # Convert parameters
        converted = {}
        for key, value in params.items():
            if value is not None:
                param_type = cls.PARAM_TYPES.get(key, 'str')
                converted[key] = cls.CONVERTERS[param_type](value)
        
        # Create scheduler
        try:
            return scheduler_class(optimizer, **converted)
        except Exception as e:
            logger.warning(f"Failed to create scheduler {name}: {e}")
            return None

# ============================================================================
# EARLY STOPPING
# ============================================================================

class EarlyStopping:
    """
    Early stopping mechanism to prevent overfitting.
    
    Monitors a validation metric and stops training when the metric
    stops improving for a specified number of epochs (patience).
    Optionally restores the best model weights upon stopping.
    
    Attributes:
        patience: Number of epochs to wait for improvement
        min_delta: Minimum change to consider as improvement
        restore_best: Whether to restore best model weights
        mode: 'min' for loss-like metrics, 'max' for accuracy-like metrics
    """
    
    def __init__(self, patience: int = 2000, min_delta: float = 0.0, 
                 restore_best: bool = True, mode: str = 'min'):
        """
        Initialize early stopping monitor.
        
        Args:
            patience: Epochs without improvement before stopping
            min_delta: Minimum change threshold for improvement
            restore_best: Restore model to best checkpoint on stop
            mode: Optimization direction ('min' or 'max')
        """
        self.patience = patience
        self.min_delta = min_delta
        self.restore_best = restore_best
        self.mode = mode
        
        # Internal state
        self.counter = 0
        self.best_model = None
        self.best_value = None
        self.best_epoch = None
        
    def __call__(self, model: nn.Module, value: float, epoch: int) -> bool:
        """
        Check if training should stop based on metric value.
        
        Args:
            model: Current model state
            value: Current metric value
            epoch: Current epoch number
            
        Returns:
            True if training should stop, False otherwise
        """
        # First epoch - establish baseline
        if self.best_value is None:
            self._save_checkpoint(model, value, epoch)
            return False
        
        # Calculate improvement based on optimization mode
        improved = (self.best_value - value >= self.min_delta if self.mode == 'min' 
                   else (value - self.best_value) >= self.min_delta)
        
        if improved:
            # Reset patience counter on improvement
            self.counter = 0
            self._save_checkpoint(model, value, epoch)

        else:
            # Increment patience counter
            self.counter += 1
            if self.counter >= self.patience:
                # Patience exhausted - stop training
                if self.restore_best:
                    model.load_state_dict(self.best_model)
                    logger.info(f"Restored best model from epoch {self.best_epoch} "
                              f"(best {self.mode}: {self.best_value:.4f})")
                return True
        
        return False
    
    def _save_checkpoint(self, model: nn.Module, value: float, epoch: int):
        """
        Save model checkpoint and update best metrics.
        
        Args:
            model: Model to checkpoint
            value: Metric value
            epoch: Current epoch
        """
        self.best_value = value
        self.best_model = copy.deepcopy(model.state_dict())
        self.best_epoch = epoch
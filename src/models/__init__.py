"""Models module for ML framework.

This module provides various machine learning models including:
- Classical ML models (scikit-learn based)
- Neural networks (PyTorch based)
- Ensemble methods
- Advanced architectures (Transformers, GNNs, Physics-informed)
"""

# Base classes and factories
from .base import (
    BaseModel,
    ModelFactory,
    LossFactory,
    OptimizerFactory,
    SchedulerFactory,
    EarlyStopping,
    safe_int,
    safe_float
)

# PyTorch base class
from .pytorch_base import TorchModel

# Classical models
from .classical import (
    LogisticRegressionModel,
    SVMModel,
    KNNModel,
    KMeansClassifierModel,
    DecisionTreeModel,
    RandomForestModel,
    GradientBoostingModel,
    NaiveBayesModel
)

# Neural network implementations
from .neural import (
    NeuralNetworkModel,
    TransformerModel,
    GraphNeuralNetworkModel,
)

# Ensemble models
from .ensemble import (
    VotingEnsemble,
    StackingEnsemble
)

# Advanced models
from .advanced import (
    AdvancedContrastiveModel,
    PhysicsInformedClassifier
)

# Version
__version__ = '1.0.0'

# Public API
__all__ = [
    # Base classes
    'BaseModel',
    'ModelFactory',
    'TorchModel',
    
    # Factories
    'LossFactory',
    'OptimizerFactory', 
    'SchedulerFactory',
    
    # Utilities
    'EarlyStopping',
    'safe_int',
    'safe_float',
    
    # Classical models
    'LogisticRegressionModel',
    'SVMModel',
    'KNNModel',
    'KMeansClassifierModel',
    'DecisionTreeModel',
    'RandomForestModel',
    'GradientBoostingModel',
    'NaiveBayesModel',
    
    # Neural network models
    'NeuralNetworkModel',
    'TransformerModel',
    'GraphNeuralNetworkModel',
    
    # Ensemble models
    'VotingEnsemble',
    'StackingEnsemble',
    
    # Advanced models
    'AdvancedContrastiveModel',
    'PhysicsInformedClassifier',
]
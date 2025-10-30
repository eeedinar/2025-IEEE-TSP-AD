"""
Feature Scaler Module
====================

Simple, professional feature scaling for ML pipelines.

"""

import numpy as np
import joblib
from pathlib import Path
from sklearn.preprocessing import StandardScaler, MinMaxScaler, RobustScaler
from typing import Dict, Tuple, Optional, List, Any, Union
import logging

logger = logging.getLogger(__name__)

class FeatureScaler:
    """Simple feature scaler with save/load capabilities."""
    
    SCALERS = {
        'standard': StandardScaler,
        'minmax': MinMaxScaler,
        'robust': RobustScaler
    }
    
    def __init__(self, scaler_type: str = 'standard'):
        """
        Initialize scaler.
        
        Args:
            scaler_type: Type of scaler ('standard', 'minmax', 'robust')
        """
        if scaler_type not in self.SCALERS:
            logger.warning(f"Unknown scaler '{scaler_type}', using 'standard'")
            scaler_type = 'standard'

        self.scaler_type = scaler_type
        self.scaler = self.SCALERS[scaler_type]()
    
    def fit_transform(self, X_train: np.ndarray, *arrays: np.ndarray) -> Union[np.ndarray, tuple]:
        """
        Fit on training data and transform all provided arrays.
        
        Args:
            X_train: Training data to fit on
            *arrays: Additional arrays to transform (e.g., X_val, X_test)
            
        Returns:
            Single array if only X_train provided, tuple of arrays otherwise
        """
        # Fit and transform training data
        X_train_scaled = self.scaler.fit_transform(X_train)
        
        logger.info(f"Fitted {self.scaler_type} scaler on {X_train.shape[1]} features")
        
        if not arrays:
            return X_train_scaled
        
        # Transform additional arrays
        transformed = [X_train_scaled]
        for arr in arrays:
            transformed.append(self.transform(arr))
        return tuple(transformed)
    
    def transform(self, X: np.ndarray) -> np.ndarray:
        """Transform new data using fitted scaler."""
        # sklearn uses check_is_fitted internally which checks for fitted attributes
        # ending with underscore (e.g., mean_, scale_, min_, etc.)
        return self.scaler.transform(X)
    
    def save(self, filepath: Union[str, Path]):
        """Save fitted scaler to file."""
        # sklearn's check_is_fitted will be called when we try to access fitted attributes
        # Let it handle the validation
        filepath = Path(filepath)
        filepath.parent.mkdir(parents=True, exist_ok=True)
        
        joblib.dump({
            'scaler': self.scaler,
            'scaler_type': self.scaler_type
        }, filepath)
        
        logger.info(f"Saved {self.scaler_type} scaler to {filepath}")
    
    @classmethod
    def load(cls, filepath: Union[str, Path]) -> 'FeatureScaler':
        """Load fitted scaler from file."""
        filepath = Path(filepath)
        if not filepath.exists():
            raise FileNotFoundError(f"Scaler file not found: {filepath}")
        
        data = joblib.load(filepath)
        
        # Create instance and restore state
        instance = cls(data['scaler_type'])
        instance.scaler = data['scaler']
        
        logger.info(f"Loaded {instance.scaler_type} scaler from {filepath}")
        return instance
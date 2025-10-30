"""Ensemble models including XGBoost and custom ensembles."""

import numpy as np
from typing import Dict, Any, Optional, List
from loguru import logger


from .base import BaseModel, ModelFactory


class VotingEnsemble(BaseModel):
    """Voting ensemble of multiple models."""
    
    def __init__(self, config: Optional[Dict[str, Any]] = None):
        super().__init__(config)
        
        self.models = []
        self.model_names = config.get('models', [
            'random_forest',
            'gradient_boosting',
            'svm'
        ])
        self.voting = config.get('voting', 'soft')  # 'hard' or 'soft'
        self.weights = config.get('weights', None)
        
    def _initialize_models(self, input_dim: int):
        """Initialize component models."""
        for model_name in self.model_names:
            model_config = self.config.get(f'{model_name}_config', {})
            model_config['input_dim'] = input_dim
            
            try:
                model = ModelFactory.create_model(model_name, model_config)
                self.models.append(model)
                logger.info(f"Added {model_name} to ensemble")
            except Exception as e:
                logger.warning(f"Failed to add {model_name}: {e}")
    
    def train(self, X: np.ndarray, y: np.ndarray) -> None:
        if not self.models:
            self._initialize_models(X.shape[1])
        
        # Train each model
        for i, model in enumerate(self.models):
            logger.info(f"Training model {i+1}/{len(self.models)}: {model.model_name}")
            model.train(X, y)
        
        self.fitted = True
        logger.info(f"Ensemble trained with {len(self.models)} models")
    
    def predict(self, X: np.ndarray) -> np.ndarray:
        if not self.fitted:
            raise ValueError("Model must be trained before prediction")
        
        if self.voting == 'hard':
            # Collect predictions from each model
            predictions = np.array([model.predict(X) for model in self.models])
            
            # Weighted voting
            if self.weights is not None:
                weighted_predictions = np.zeros((len(X), 2))
                for i, pred in enumerate(predictions):
                    weighted_predictions[pred == 0, 0] += self.weights[i]
                    weighted_predictions[pred == 1, 1] += self.weights[i]
                return np.argmax(weighted_predictions, axis=1)
            else:
                # Simple majority voting
                return np.round(np.mean(predictions, axis=0)).astype(int)
        
        else:  # soft voting
            # Use probabilities
            probas = self.predict_proba(X)
            return np.argmax(probas, axis=1)
    
    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        if not self.fitted:
            raise ValueError("Model must be trained before prediction")
        
        # Collect probability predictions
        all_probas = []
        
        for model in self.models:
            probas = model.predict_proba(X)
            if probas is not None:
                all_probas.append(probas)
            else:
                # For models without probability support, use one-hot encoding
                preds = model.predict(X)
                probas = np.zeros((len(X), 2))
                probas[np.arange(len(X)), preds] = 1
                all_probas.append(probas)
        
        all_probas = np.array(all_probas)
        
        # Average probabilities (with optional weights)
        if self.weights is not None:
            weights = np.array(self.weights)[:, np.newaxis, np.newaxis]
            weighted_probas = all_probas * weights
            return np.sum(weighted_probas, axis=0) / np.sum(self.weights)
        else:
            return np.mean(all_probas, axis=0)


class StackingEnsemble(BaseModel):
    """Stacking ensemble with meta-learner."""
    
    def __init__(self, config: Optional[Dict[str, Any]] = None):
        super().__init__(config)
        
        self.base_models = []
        self.base_model_names = config.get('base_models', [
            'random_forest',
            'gradient_boosting',
            'svm',
            'neural_network'
        ])
        self.meta_model_name = config.get('meta_model', 'logistic_regression')
        self.meta_model = None
        self.use_probas = config.get('use_probas', True)
        
    def _initialize_models(self, input_dim: int):
        """Initialize base and meta models."""
        # Initialize base models
        for model_name in self.base_model_names:
            model_config = self.config.get(f'{model_name}_config', {})
            model_config['input_dim'] = input_dim
            
            try:
                model = ModelFactory.create_model(model_name, model_config)
                self.base_models.append(model)
                logger.info(f"Added {model_name} as base model")
            except Exception as e:
                logger.warning(f"Failed to add {model_name}: {e}")
        
        # Initialize meta model
        meta_config = self.config.get('meta_config', {})
        if self.use_probas:
            meta_config['input_dim'] = len(self.base_models) * 2
        else:
            meta_config['input_dim'] = len(self.base_models)
        
        self.meta_model = ModelFactory.create_model(
            self.meta_model_name, 
            meta_config
        )
    
    def _get_meta_features(self, X: np.ndarray) -> np.ndarray:
        """Generate meta features from base models."""
        meta_features = []
        
        for model in self.base_models:
            if self.use_probas:
                probas = model.predict_proba(X)
                if probas is not None:
                    meta_features.append(probas)
                else:
                    # Fallback to predictions
                    preds = model.predict(X).reshape(-1, 1)
                    meta_features.append(preds)
                    meta_features.append(1 - preds)  # Add complement
            else:
                preds = model.predict(X).reshape(-1, 1)
                meta_features.append(preds)
        
        return np.hstack(meta_features)
    
    def train(self, X: np.ndarray, y: np.ndarray) -> None:
        if not self.base_models:
            self._initialize_models(X.shape[1])
        
        # Split data for training base models and meta model
        from sklearn.model_selection import KFold
        kf = KFold(n_splits=5, shuffle=True, random_state=42)
        
        # Collect out-of-fold predictions
        meta_features = np.zeros((len(X), len(self.base_models) * (2 if self.use_probas else 1)))
        
        for i, model in enumerate(self.base_models):
            logger.info(f"Training base model {i+1}/{len(self.base_models)}: {model.model_name}")
            
            # Out-of-fold predictions
            if self.use_probas:
                oof_probas = np.zeros((len(X), 2))
            else:
                oof_preds = np.zeros(len(X))
            
            for train_idx, val_idx in kf.split(X):
                X_train, X_val = X[train_idx], X[val_idx]
                y_train, y_val = y[train_idx], y[val_idx]
                
                model.train(X_train, y_train)
                
                if self.use_probas:
                    probas = model.predict_proba(X_val)
                    if probas is not None:
                        oof_probas[val_idx] = probas
                    else:
                        preds = model.predict(X_val)
                        oof_probas[val_idx, 0] = 1 - preds
                        oof_probas[val_idx, 1] = preds
                else:
                    oof_preds[val_idx] = model.predict(X_val)
            
            # Store meta features
            if self.use_probas:
                meta_features[:, i*2:(i+1)*2] = oof_probas
            else:
                meta_features[:, i] = oof_preds
            
            # Retrain on full data
            model.train(X, y)
        
        # Train meta model
        logger.info(f"Training meta model: {self.meta_model_name}")
        self.meta_model.train(meta_features, y)
        
        self.fitted = True
        logger.info(f"Stacking ensemble trained successfully")
    
    def predict(self, X: np.ndarray) -> np.ndarray:
        if not self.fitted:
            raise ValueError("Model must be trained before prediction")
        
        meta_features = self._get_meta_features(X)
        return self.meta_model.predict(meta_features)
    
    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        if not self.fitted:
            raise ValueError("Model must be trained before prediction")
        
        meta_features = self._get_meta_features(X)
        return self.meta_model.predict_proba(meta_features)


# Register models with factory
ModelFactory.register_model('voting_ensemble', VotingEnsemble)
ModelFactory.register_model('stacking_ensemble', StackingEnsemble)
"""Classical machine learning models."""

import numpy as np
from typing import Dict, Any, Optional
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.tree import DecisionTreeClassifier
from sklearn.cluster import KMeans
from loguru import logger

from .base import BaseModel, ModelFactory, safe_float


class SklearnModel(BaseModel):
    """Base wrapper for sklearn models."""
    
    def __init__(self, model_class, config: Optional[Dict[str, Any]] = None, **defaults):
        super().__init__(config)
        
        # Start with defaults
        params = defaults.copy()
        
        # Only update with config params that model accepts
        if config:
            sig = model_class.__init__.__code__.co_varnames
            for k, v in config.items():
                if k in sig:
                    params[k] = v
        
        self.model = model_class(**params)
        
    def train(self, X: np.ndarray, y: np.ndarray) -> None:
        self.model.fit(X, y)
        self.fitted = True
        logger.info(f"{self.model_name} trained")
        
    def predict(self, X: np.ndarray) -> np.ndarray:
        if not self.fitted:
            raise ValueError("Model not trained")
        return self.model.predict(X)


# Individual model classes for cleaner imports

class LogisticRegressionModel(SklearnModel):
    """Logistic Regression classifier."""
    def __init__(self, config: Optional[Dict[str, Any]] = None):
        super().__init__(LogisticRegression, config, max_iter=1000, random_state=42)


class SVMModel(SklearnModel):
    """Support Vector Machine classifier."""
    def __init__(self, config: Optional[Dict[str, Any]] = None):
        super().__init__(SVC, config, probability=True, gamma='scale', random_state=42)


class KNNModel(SklearnModel):
    """K-Nearest Neighbors classifier."""
    def __init__(self, config: Optional[Dict[str, Any]] = None):
        super().__init__(KNeighborsClassifier, config, n_neighbors=5)


class DecisionTreeModel(SklearnModel):
    """Decision Tree classifier."""
    def __init__(self, config: Optional[Dict[str, Any]] = None):
        super().__init__(DecisionTreeClassifier, config, random_state=42)


class RandomForestModel(SklearnModel):
    """Random Forest classifier."""
    def __init__(self, config: Optional[Dict[str, Any]] = None):
        super().__init__(RandomForestClassifier, config, n_estimators=100, n_jobs=-1, random_state=42)


class GradientBoostingModel(SklearnModel):
    """Gradient Boosting classifier."""
    def __init__(self, config: Optional[Dict[str, Any]] = None):
        super().__init__(GradientBoostingClassifier, config, n_estimators=100, random_state=42)


class NaiveBayesModel(SklearnModel):
    """Gaussian Naive Bayes classifier."""
    def __init__(self, config: Optional[Dict[str, Any]] = None):
        # Convert var_smoothing to float if it comes as string
        if config and 'var_smoothing' in config:
            config = config.copy()
            config['var_smoothing'] = safe_float(config['var_smoothing'], 1e-9)
        super().__init__(GaussianNB, config, var_smoothing=1e-9)


class KMeansClassifierModel(BaseModel):
    """KMeans adapted for classification."""
    
    def __init__(self, config: Optional[Dict[str, Any]] = None):
        super().__init__(config)
        
        # Get parameters
        n_clusters = config.get('n_clusters', 2) if config else 2
        if config and 'n_classes' in config:
            n_clusters = config['n_classes']
            
        # Only pass valid KMeans parameters
        params = {'n_clusters': n_clusters, 'n_init': 10, 'random_state': 42}
        if config:
            sig = KMeans.__init__.__code__.co_varnames
            for k, v in config.items():
                if k in sig and k not in params:
                    params[k] = v
        
        self.model = KMeans(**params)
        self.cluster_to_class = {}
        
    def train(self, X: np.ndarray, y: np.ndarray) -> None:
        clusters = self.model.fit_predict(X)
        
        # Map each cluster to most common class
        for i in range(self.model.n_clusters):
            mask = clusters == i
            if mask.any():
                self.cluster_to_class[i] = np.bincount(y[mask]).argmax()
            else:
                self.cluster_to_class[i] = 0
                
        self.fitted = True
        logger.info(f"{self.model_name} trained with {self.model.n_clusters} clusters")
        
    def predict(self, X: np.ndarray) -> np.ndarray:
        if not self.fitted:
            raise ValueError("Model not trained")
        clusters = self.model.predict(X)
        return np.array([self.cluster_to_class[c] for c in clusters])
    
    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        """Predict class probabilities based on distance to cluster centers."""
        if not self.fitted:
            raise ValueError("Model not trained")
        
        # Calculate distances to all cluster centers
        distances = self.model.transform(X)
        
        # Convert distances to similarities
        epsilon = 1e-10
        similarities = 1 / (distances + epsilon)
        
        # Create probability matrix
        n_samples = X.shape[0]
        n_classes = len(set(self.cluster_to_class.values()))
        probabilities = np.zeros((n_samples, n_classes))
        
        # Aggregate similarities by class
        for cluster_id, class_label in self.cluster_to_class.items():
            probabilities[:, class_label] += similarities[:, cluster_id]
        
        # Normalize to get probabilities
        row_sums = probabilities.sum(axis=1, keepdims=True)
        row_sums[row_sums == 0] = 1
        probabilities = probabilities / row_sums
        
        return probabilities


# Register all models
ModelFactory.register_model('logistic_regression', LogisticRegressionModel)
ModelFactory.register_model('svm', SVMModel)
ModelFactory.register_model('svm_linear', 
                           lambda config=None: SVMModel({**config, 'kernel': 'linear'} if config else {'kernel': 'linear'}))
ModelFactory.register_model('svm_poly',
                           lambda config=None: SVMModel({**config, 'kernel': 'poly'} if config else {'kernel': 'poly'}))
ModelFactory.register_model('knn', KNNModel)
ModelFactory.register_model('knn_weighted',
                           lambda config=None: KNNModel({**config, 'weights': 'distance'} if config else {'weights': 'distance'}))
ModelFactory.register_model('kmeans', KMeansClassifierModel)
ModelFactory.register_model('decision_tree', DecisionTreeModel)
ModelFactory.register_model('random_forest', RandomForestModel)
ModelFactory.register_model('gradient_boosting', GradientBoostingModel)
ModelFactory.register_model('naive_bayes', NaiveBayesModel)
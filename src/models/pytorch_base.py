"""PyTorch base class for neural network models.

This module provides the foundation for all PyTorch-based neural network models
in the framework. It implements a complete training pipeline with extensive
configuration options and advanced features.

Key Features:
    - Flexible loss function configuration (single or multi-loss)
    - Automatic optimizer and scheduler management
    - Validation set monitoring with early stopping
    - Gradient clipping support
    - Comprehensive metric tracking and history
    - Device-agnostic training (CPU/GPU)

"""
import copy
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import TensorDataset, DataLoader
from typing import Dict, Any, Optional, Tuple, List, Union
from loguru import logger
from ..metrics.metrics_wrapper import MetricsWrapper

from .base import (
    BaseModel, LossFactory, OptimizerFactory, 
    SchedulerFactory, EarlyStopping, safe_int, safe_float
)

class TorchModel(BaseModel):
    """
    All behavior is controlled through configuration dictionaries,
    ensuring no hardcoded values and maximum flexibility.
    
    Attributes:
        device: PyTorch device for computation (CPU/GPU)
        criterion: Primary loss function
        auxiliary_losses: List of auxiliary loss functions for multi-loss training
        loss_weights: Weights for combining multiple losses
        optimizer: Gradient optimizer
        scheduler: Learning rate scheduler
        history: Training history dictionary
    """
    
    def __init__(self, config: Optional[Dict[str, Any]] = None):
        """
        Initialize PyTorch model with configuration.
        
        Args:
            config: Configuration dictionary containing:
                - device: Computation device ('cuda' or 'cpu')
                - epochs: Number of training epochs
                - batch_size: Mini-batch size
                - learning_rate: Initial learning rate
                - loss: Loss function name or configuration
                - multi_loss: Multi-loss configuration with primary and auxiliary losses
                - optimizer: Optimizer name or configuration
                - scheduler: LR scheduler name or configuration
                - early_stopping: Early stopping configuration
                - gradient_clipping: Gradient clipping configuration
                - model_selection: Metric monitoring configuration
                - log_interval: Frequency of progress logging
        """
        super().__init__(config)
        
        # Device configuration
        self.device = torch.device(config.get('device', 'cuda' if torch.cuda.is_available() else 'cpu'))

        # Components to be initialized in train()
        self.criterion = None
        self.auxiliary_losses = []
        self.optimizer = None
        self.scheduler = None
        self.scheduler_best_epoch = None
        self.scheduler_best_value = None
        self.history = {}

        # NMF flag - will be set by train.py
        self.uses_nmf = False

    def train(self, X: np.ndarray, y: np.ndarray, 
              X_val: Optional[np.ndarray] = None, 
              y_val: Optional[np.ndarray] = None,
              X_train_nmf: Optional[np.ndarray] = None,           
              X_val_nmf: Optional[np.ndarray] = None) -> None:
        """
        Train the neural network model.
        
        Implements complete training pipeline with:
        - Mini-batch gradient descent
        - Optional validation monitoring
        - Early stopping
        - Learning rate scheduling
        - Gradient clipping
        - Progress logging
        - Best model restoration when model selection is enabled
        
        Args:
            X: Training features of shape (n_samples, n_features)
            y: Training labels of shape (n_samples,)
            X_val: Validation features (optional)
            y_val: Validation labels (optional)

        Raises:
            ValueError: If model is not initialized
        """
        if self.model is None:
            raise ValueError("Model not initialized")
        
        # Get training configuration
        epochs = safe_int(self.config.get('epochs', 100), 100)
        batch_size = safe_int(self.config.get('batch_size', 32), 32)
        learning_rate = safe_float(self.config.get('learning_rate', 0.001), 0.001)
        
        # Initialize components
        self._initialize_training_components(len(np.unique(y)), learning_rate)
        
        # Create data loader with reproducible shuffling
        generator = torch.Generator()
        generator.manual_seed(self.config.get('random_state', 42))

        # Create data loader with NMF support

        if self.uses_nmf and X_train_nmf is not None:
            # For NMF models: include both raw and NMF features
            train_dataset = TensorDataset(
                torch.FloatTensor(X), 
                torch.FloatTensor(X_train_nmf),
                torch.LongTensor(y)
            )
        else:
            # For regular models: only raw features
            train_dataset = TensorDataset(
                torch.FloatTensor(X), 
                torch.LongTensor(y)
            )

        # For regular models: only raw features
        train_loader = DataLoader(
            train_dataset,
            batch_size=batch_size,
            shuffle=True,
            generator=generator
            )
    
        # Initialize history
        self.history = {
            'train_loss': [], 'train_f1': [], 
            'val_loss': [], 'val_f1': [], 
            'epochs': []
        }
        
        # check if validation avaiable
        has_validation = X_val is not None and y_val is not None

        # Model selection configuration
        model_selection_config = self.config.get('model_selection', {})
        model_selection_enabled = model_selection_config.get('enabled', False)
        monitor_metric = model_selection_config.get('monitor', 'val_loss' if has_validation else 'train_loss')
        mode = 'min' if 'loss' in monitor_metric else 'max'

        # Early stopping setup
        if has_validation:
            # Validation setup
            X_val_tensor = torch.FloatTensor(X_val).to(self.device)
            y_val_tensor = torch.LongTensor(y_val).to(self.device)

            # Setup early stopping mechanism
            early_stop_config = self.config.get('early_stopping', {})
            if early_stop_config.get('enabled', False):
                early_stopping = EarlyStopping(
                    patience=early_stop_config.get('patience', 20),
                    min_delta=early_stop_config.get('min_delta', 0.0),
                    restore_best=early_stop_config.get('restore_best_weights', True),
                    mode=mode
                )
            else:
                early_stopping = None

        # Best model tracking
        best_metric = None
        best_epoch = None
        best_model_state = None
        
        # Pre-determine which metric to monitor
        use_validation_metric = 'val_' in monitor_metric and has_validation
        metric_key = monitor_metric.replace('val_', '').replace('train_', '')  # 'loss' or 'f1'

        # Training loop
        self.model.train()
        early_stopped = False

        for epoch in range(1, epochs + 1):
            # Train one epoch
            train_metrics = self._train_epoch(train_loader, has_nmf=self.uses_nmf)
            
            # Update history
            self.history['train_loss'].append(train_metrics['loss'])
            self.history['train_f1'].append(train_metrics['f1'])
            self.history['epochs'].append(epoch)
            
            # Validation
            val_metrics = None
            if has_validation:
                if self.uses_nmf and X_val_nmf is not None:
                    X_val_nmf_tensor = torch.FloatTensor(X_val_nmf).to(self.device)
                    val_metrics = self._validate(X_val_tensor, y_val_tensor, X_val_nmf_tensor)
                else:
                    val_metrics = self._validate(X_val_tensor, y_val_tensor)
                self.history['val_loss'].append(val_metrics['loss'])
                self.history['val_f1'].append(val_metrics['f1'])

            # Log progress
            if epoch % self.config.get('log_interval', 10) == 0:
                self._log_progress(epoch, epochs, train_metrics, val_metrics)

            # Model selection - track best model
            current_metric = train_metrics.get(metric_key, train_metrics['loss'])  # default metric - will be changed if use_validation_metric
            if model_selection_enabled:
                # Get current metric value based on pre-determined source
                if use_validation_metric:
                    current_metric = val_metrics.get(metric_key, val_metrics['loss'])

                # Check if this is the best model so far
                is_best = (best_metric is None or 
                          (mode == 'min' and current_metric < best_metric) or
                          (mode == 'max' and current_metric > best_metric))
                
                if is_best:
                    best_metric = current_metric
                    best_epoch = epoch
                    best_model_state = copy.deepcopy(self.model.state_dict())
                    logger.debug(f"New best model at epoch {best_epoch} with {monitor_metric}={best_metric:.4f}")

            # Early stopping check
            if early_stopping and val_metrics:
                # Use model selection monitor metric for early stopping if specified
                monitor_value = val_metrics.get(metric_key, val_metrics['loss']) if use_validation_metric else val_metrics['loss']
                
                if early_stopping(self.model, monitor_value, epoch):
                    logger.info(f"Early stopping at epoch {epoch}")
                    early_stopped = True
                    break

            # Update learning rate
            self._update_scheduler(current_metric, mode, epoch)
    
        # Restore best model if model selection is enabled and training completed naturally
        if model_selection_enabled and best_model_state is not None and not early_stopped:
            self.model.load_state_dict(best_model_state)
            logger.info(f"Restored best model from epoch {best_epoch} with {monitor_metric}={best_metric:.4f}")
        
        # Log final center info if using center loss
        if hasattr(self.criterion, 'centers'):
            logger.info(f"Final centers shape: {self.criterion.centers.shape}")
            logger.info(f"Centers norm: {torch.norm(self.criterion.centers, dim=1).mean():.4f}")
        elif hasattr(self.criterion, 'losses'):
            for i, loss in enumerate(self.criterion.losses):
                if hasattr(loss, 'centers'):
                    logger.info(f"{loss.__class__.__name__} final centers norm: {torch.norm(loss.centers, dim=1).mean():.4f}")
        
        self.model.eval()
        self.fitted = True
        logger.info("Training completed")
    
    def _initialize_training_components(self, num_classes: int, learning_rate: float):
        """
        Initialize loss functions, optimizer, and scheduler.
        
        Sets up all training components based on configuration. Handles both
        single loss and multi-loss setups, creates the optimizer with proper
        parameters, and initializes the learning rate scheduler.
        
        Args:
            num_classes: Number of output classes for loss functions
            learning_rate: Initial learning rate for optimizer
        """
        # Create loss function(s)
        loss_config = self.config.get('loss', None)
        feat_dim = getattr(self.model, 'feat_dim', self.config.get('feat_dim', None))
        
        self.criterion = LossFactory.create_loss(loss_config, num_classes, feat_dim)
        if not self.criterion:
            # Use a default Cross Entropy Loss
            logger.warning('No loss defined in config, using CrossEntropyLoss as default')
            self.criterion = nn.CrossEntropyLoss()

        
        # LOGS after creating the losses:
        logger.info(f"Loss: {self.criterion.__class__.__name__}")
        if hasattr(self.criterion, 'losses'):  # CombinedLoss
            for i, loss in enumerate(self.criterion.losses):
                logger.info(f"  Sub-loss {i}: {loss.__class__.__name__} - {getattr(loss, '__dict__', {})}")
            logger.info(f"  Weights: {getattr(self.criterion, 'weights', [])}")
        else:
            logger.info(f"  Config: {getattr(self.criterion, '__dict__', {})}")

        # Create optimizer
        self.optimizer = OptimizerFactory.create_optimizer(
            self.config.get('optimizer', 'adam'),
            self.model.parameters(),
            learning_rate
        )
        logger.info(f"Optimizer: {self.optimizer.__class__.__name__} - {self.optimizer.state_dict()}")

        # Create scheduler
        self.scheduler = SchedulerFactory.create_scheduler(
            self.config.get('scheduler'),
            self.optimizer
        )
        logger.info(f"Scheduler: {self.scheduler.__class__.__name__ if self.scheduler else 'None'} - {getattr(self.scheduler, '__dict__', {}) if self.scheduler else 'N/A'}")

        # Move to device
        self.model.to(self.device)
        self.criterion.to(self.device)

    def _update_centers(self, features, labels):
        """Update all centers."""
        if hasattr(self.criterion, 'update_centers'):
            self.criterion.update_centers(features, labels)
        elif hasattr(self.criterion, 'losses'):
            for loss_fn in self.criterion.losses:
                if hasattr(loss_fn, 'update_centers'):
                    loss_fn.update_centers(features, labels)

    def _train_epoch(self, train_loader: DataLoader, has_nmf: bool = False) -> Dict[str, float]:

        """
        Train for one epoch.
        
        Performs forward pass, loss computation, backpropagation, and
        parameter updates for all batches in the training data.
        
        Args:
            train_loader: DataLoader for training batches
            has_nmf: Whether the data includes NMF features
        Returns:
            Dictionary containing epoch metrics (loss and f1)
        """
        epoch_loss = 0
        all_preds = []
        all_targets = []
        
        for batch_data in train_loader:
            if has_nmf:
                # Unpack: (raw_features, nmf_features, labels)
                batch_X, batch_X_train_nmf, batch_y = batch_data
                batch_X = batch_X.to(self.device)
                batch_X_train_nmf = batch_X_train_nmf.to(self.device)
                batch_y = batch_y.to(self.device)
            else:
                # Unpack: (features, labels)
                batch_X, batch_y = batch_data
                batch_X = batch_X.to(self.device)
                batch_y = batch_y.to(self.device)
                batch_X_train_nmf = None

            # Forward pass
            self.optimizer.zero_grad()
            
            if has_nmf:
                outputs, features = self._forward_pass(batch_X, batch_X_train_nmf)
            else:
                outputs, features = self._forward_pass(batch_X)

            loss = self._compute_loss(outputs, batch_y, features)
            
            # Backward pass
            loss.backward()
            
            # Gradient clipping if configured
            grad_clip_config = self.config.get('gradient_clipping', {})
            if grad_clip_config.get('enabled', False):
                max_norm = grad_clip_config.get('max_norm', 1.0)
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm)
            
            self.optimizer.step()
            
            if features is not None:
                self._update_centers(features, batch_y)

            # Track metrics
            epoch_loss += loss.item()
            _, preds = torch.max(outputs, 1)
            all_preds.extend(preds.cpu().numpy())
            all_targets.extend(batch_y.cpu().numpy())
        
        # Compute epoch metrics
        avg_loss = epoch_loss / len(train_loader)
        f1 = MetricsWrapper.get_eval_metrics('f1', y_true=np.array(all_targets), y_pred=np.array(all_preds))

        return {'loss': avg_loss, 'f1': f1}
    
    def _validate(self, X_val: torch.Tensor, y_val: torch.Tensor, 
              X_val_nmf: Optional[torch.Tensor] = None) -> Dict[str, float]:
        """
        Validate model on validation set.
        
        Evaluates the model without gradient computation to save memory
        and computation time.
        
        Args:
            X_val: Validation features tensor
            y_val: Validation labels tensor
            X_val_nmf: Validation NMF features tensor (optional)
        Returns:
            Dictionary containing validation metrics (loss and f1)
        """
        self.model.eval()
        with torch.no_grad():
            if X_val_nmf is not None:
                outputs, features = self._forward_pass(X_val, X_val_nmf)
            else:
                outputs, features = self._forward_pass(X_val)
            
            loss = self._compute_loss(outputs, y_val, features)
            _, preds = torch.max(outputs, 1)
            f1 = MetricsWrapper.get_eval_metrics('f1', y_true=y_val.cpu().numpy(), y_pred=preds.cpu().numpy())

        self.model.train()
        
        return {'loss': loss.item(), 'f1': f1}
    
    def _forward_pass(self, X: torch.Tensor, X_train_nmf: Optional[torch.Tensor] = None) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        """
        Unified forward pass for different model architectures.
        
        Handles various model interfaces:
        - Standard models: Return predictions only
        - Feature models: Return predictions and intermediate features
        
        This abstraction allows the same training code to work with
        different model architectures.
        
        Args:
            X: Input tensor
            X_train_nmf: NMF features tensor (optional)
        Returns:
            Tuple of (predictions, features) where features may be None
        """
        # Check if model supports NMF features
        if X_train_nmf is not None and hasattr(self.model, 'forward_with_features'):
            # Try NMF-aware forward pass
            try:
                return self.model.forward_with_features(X, X_train_nmf)
            except TypeError:
                # Fallback: model doesn't accept NMF features
                return self.model.forward_with_features(X)
        elif hasattr(self.model, 'forward_with_features'):
            return self.model.forward_with_features(X)
        else:
            # Standard forward
            outputs = self.model(X)
            features = None

        return outputs, features
    
    def _compute_loss(self, outputs: torch.Tensor, targets: torch.Tensor,
                     features: Optional[torch.Tensor] = None) -> torch.Tensor:
        """
        Compute loss using the criterion.

        Args:
            outputs: Model predictions
            targets: Ground truth labels
            features: Intermediate features for auxiliary losses
            
        Returns:
            Loss value
        """
        # Check if loss accepts features (like CombinedLoss with feature-based losses)
        loss_name = self.criterion.__class__.__name__
        if features is not None: 
            if loss_name in ['CenterLoss', 'ContrastiveCenterLoss']:
                return self.criterion(features, targets)
            elif loss_name == 'RingLoss':
                return self.criterion(features)
            elif loss_name == 'FeatureAwareCombinedLoss':
                return self.criterion(outputs, targets, features)
        
        return self.criterion(outputs, targets) # Standard loss computation


    def _update_scheduler(self, metric_value: float, mode: str, current_epoch: int):
        """
        Update learning rate scheduler.

        Args:
            metric_value: The metric value to use for scheduling
            mode: 'min' or 'max' for optimization direction
            current_epoch: Current epoch number (1-indexed)
        """
        if self.scheduler is None:
            return

        scheduler_type = type(self.scheduler).__name__
        current_lr = self.optimizer.param_groups[0]['lr']

        # Only relevant for ReduceLROnPlateau
        if 'ReduceLROnPlateau' in scheduler_type:
            if self.scheduler_best_epoch is None:
                self.scheduler_best_epoch = current_epoch
                self.scheduler_best_value = metric_value
            else:
                improved = (
                    (mode == 'min' and metric_value < self.scheduler_best_value) or
                    (mode == 'max' and metric_value > self.scheduler_best_value)
                )
                if improved:
                    self.scheduler_best_epoch = current_epoch
                    self.scheduler_best_value = metric_value
                    logger.info(f"New best metric: {metric_value:.4f} at epoch {current_epoch}")

            self.scheduler.step(metric_value)

            new_lr = self.optimizer.param_groups[0]['lr']
            if new_lr != current_lr:
                epochs_ago = current_epoch - self.scheduler_best_epoch
                logger.info(f"[Epoch {current_epoch}] LR: {current_lr:.6f} → {new_lr:.6f} | "
                            f"Metric: {metric_value:.4f}, Best: {self.scheduler_best_value:.4f} @ epoch {self.scheduler_best_epoch} ({epochs_ago} ago)")
        else:
            self.scheduler.step()
            new_lr = self.optimizer.param_groups[0]['lr']
            if new_lr != current_lr:
                if "CosineAnnealingLR" not in scheduler_type: # CosineAnnealingLR updates frequently, clutters log
                    logger.info(f"[Epoch {current_epoch}] LR: {current_lr:.6f} → {new_lr:.6f}")

    def _log_progress(self, epoch: int, total_epochs: int, 
                     train_metrics: Dict[str, float], 
                     val_metrics: Optional[Dict[str, float]]):
        """
        Log training progress.
        
        Provides a concise summary of training progress including loss,
        metrics, and current learning rate.
        
        Args:
            epoch: Current epoch (0-indexed)
            total_epochs: Total number of epochs
            train_metrics: Training metrics for the epoch
            val_metrics: Validation metrics for the epoch (if available)
        """
        lr = self.optimizer.param_groups[0]['lr']
        
        msg = f"Epoch {epoch}/{total_epochs} | "
        msg += f"Train Loss: {train_metrics['loss']:.4f} | "
        msg += f"Train F1: {train_metrics['f1']:.4f} | "
        
        if val_metrics:
            msg += f"Val Loss: {val_metrics['loss']:.4f} | "
            msg += f"Val F1: {val_metrics['f1']:.4f} | "
        
        msg += f"LR: {lr:.6f}"
        
        logger.debug(msg)
    
    def predict(self, X: np.ndarray, X_nmf: Optional[np.ndarray] = None) -> np.ndarray:
        """
        Generate class predictions for input data.
        
        Args:
            X: Input features of shape (n_samples, n_features)
            X_nmf: NMF features of shape (n_samples, n_nmf_features) (optional)
            
        Returns:
            Predicted class labels of shape (n_samples,)
            
        Raises:
            ValueError: If model is not trained
        """
        if not self.fitted:
            raise ValueError("Model not trained")
        

        self.model.eval()
        with torch.no_grad():
            X_tensor = torch.FloatTensor(X).to(self.device)
            
            # Handle NMF models
            if self.uses_nmf and X_nmf is not None:
                X_nmf_tensor = torch.FloatTensor(X_nmf).to(self.device)
                # Try to call model with both inputs
                try:
                    outputs = self.model(X_tensor, X_nmf_tensor)
                except TypeError:
                    # Fallback to regular forward if model doesn't accept NMF
                    outputs = self.model(X_tensor)
            else:
                outputs = self.model(X_tensor)
                
            _, predicted = torch.max(outputs, 1)
        
        return predicted.cpu().numpy()
    
    def predict_proba(self, X: np.ndarray, X_nmf: Optional[np.ndarray] = None) -> np.ndarray:
        """
        Generate class probability predictions.
        
        Uses softmax to convert model outputs to probabilities.
        
        Args:
            X: Input features of shape (n_samples, n_features)
            X_nmf: NMF features of shape (n_samples, n_nmf_features) (optional)
            
        Returns:
            Class probabilities of shape (n_samples, n_classes)
            
        Raises:
            ValueError: If model is not trained
        """
        if not self.fitted:
            raise ValueError("Model not trained")
        
        self.model.eval()
        with torch.no_grad():
            X_tensor = torch.FloatTensor(X).to(self.device)
            
            # Handle NMF models
            if self.uses_nmf and X_nmf is not None:
                X_nmf_tensor = torch.FloatTensor(X_nmf).to(self.device)
                # Try to call model with both inputs
                try:
                    outputs = self.model(X_tensor, X_nmf_tensor)
                except TypeError:
                    # Fallback to regular forward if model doesn't accept NMF
                    outputs = self.model(X_tensor)
            else:
                outputs = self.model(X_tensor)
                
            probas = torch.softmax(outputs, dim=1)
        
        return probas.cpu().numpy()
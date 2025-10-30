"""Weighted loss functions for handling class imbalance."""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from typing import Optional, Union, List
from loguru import logger

class WeightedCrossEntropyLoss(nn.Module):
    """
    Weighted Cross Entropy Loss for handling class imbalance.
    
    Automatically calculates class weights based on class distribution,
    or users provided weights.
    """
    
    def __init__(
        self,
        weight: Optional[Union[torch.Tensor, List[float], str]] = 'auto',
        reduction: str = 'mean',
        label_smoothing: float = 0.0
    ):
        """
        Initialize Weighted Cross Entropy Loss.
        
        Args:
            weight: Class weights. Can be:
                - 'auto': Automatically calculate balanced weights
                - 'inverse': Use inverse class frequencies
                - torch.Tensor or list: Manual weights for each class
                - None: No weighting (standard cross entropy)
            reduction: Specifies the reduction to apply to the output
            label_smoothing: Label smoothing factor (0.0 = no smoothing)
        """
        super().__init__()
        self.weight_type = weight
        self.reduction = reduction
        self.label_smoothing = label_smoothing
        self.weight = None
        
    def calculate_weights(self, targets: torch.Tensor) -> torch.Tensor:
        """Calculate class weights based on targets."""
        unique_classes, counts = torch.unique(targets, return_counts=True)
        n_classes = targets.max().item() + 1
        n_samples = len(targets)
        
        # Initialize weights tensor
        weights = torch.ones(n_classes, device=targets.device)
        
        if self.weight_type == 'auto':
            # Balanced weights: n_samples / (n_classes * n_samples_per_class)
            for cls, count in zip(unique_classes, counts):
                weights[cls] = n_samples / (n_classes * count.float())
                
        elif self.weight_type == 'inverse':
            # Inverse frequency
            for cls, count in zip(unique_classes, counts):
                weights[cls] = 1.0 / count.float()
            # Normalize weights
            weights = weights / weights.sum() * n_classes

        return weights

    def forward(self, inputs: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        """
        Calculate weighted cross entropy loss.
        
        Args:
            inputs: Predictions from model (logits)
            targets: Ground truth labels

        Returns:
            Computed weighted cross entropy loss
        """
        # Calculate or update weights if needed
        if self.weight_type in ['auto', 'inverse']:
            self.weight = self.calculate_weights(targets)
        elif isinstance(self.weight_type, (list, torch.Tensor)):
            self.weight = torch.tensor(self.weight_type, device=inputs.device, dtype=torch.float32)
        
        # Apply label smoothing if specified
        if self.label_smoothing > 0:
            n_classes = inputs.size(-1)
            smooth_targets = torch.zeros_like(inputs).scatter_(
                1, targets.unsqueeze(1), 1.0
            )   # one-hot encoded labels from class indices - It places the value 1.0 at the specified class index for each row.
            smooth_targets = smooth_targets * (1 - self.label_smoothing) + \
                           self.label_smoothing / n_classes
            
            # Manual cross entropy with label smoothing
            log_probs = F.log_softmax(inputs, dim=-1)
            loss = -(smooth_targets * log_probs).sum(dim=-1)

            # Apply class weights
            if self.weight is not None:
                weight_expanded = self.weight[targets]
                loss = loss * weight_expanded
        else:
            # Standard weighted cross entropy
            loss = F.cross_entropy(
                inputs, targets, 
                weight=self.weight, 
                reduction='none'
            )
        
        # Apply reduction
        if self.reduction == 'mean':
            return loss.mean()
        elif self.reduction == 'sum':
            return loss.sum()
        else:
            return loss


class BalancedCrossEntropyLoss(nn.Module):
    """
    Balanced Cross Entropy Loss that automatically adjusts to class distribution.
    Particularly effective for datasets with significant class imbalance.
    """
    
    def __init__(
        self,
        beta: float = 0.999,
        reduction: str = 'mean'
    ):
        """
        Initialize Balanced Cross Entropy Loss.
        
        Args:
            beta: The balancing parameter (0 < beta < 1)
                  Higher beta = more weight to rare classes
            reduction: Reduction method
        """
        super().__init__()
        self.beta = beta
        self.reduction = reduction
        self.class_freq = None
        
    def update_frequency(self, targets: torch.Tensor):
        """Update class frequency statistics."""
        unique, counts = torch.unique(targets, return_counts=True)
        n_classes = targets.max().item() + 1
        
        if self.class_freq is None:
            self.class_freq = torch.zeros(n_classes, device=targets.device)
            
        # Update with exponential moving average
        for cls, count in zip(unique, counts):
            self.class_freq[cls] = self.beta * self.class_freq[cls] + (1 - self.beta) * count
    
    def forward(self, inputs: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        """Calculate balanced cross entropy loss."""
        # Update class frequencies
        self.update_frequency(targets)
        
        # Calculate effective number of samples
        effective_num = 1.0 - torch.pow(self.beta, self.class_freq)
        weights = (1.0 - self.beta) / effective_num  # Gives smooth inverse-weighting based on rarity
        weights = weights / weights.sum() * len(weights)  # Ensures that average weight across classes is 1. Prevents scale mismatch across batches.

        
        # Apply weighted cross entropy
        loss = F.cross_entropy(
            inputs, targets,
            weight=weights,
            reduction=self.reduction
        )
        
        return loss


class DiceLoss(nn.Module):
    """
    Dice Loss for binary and multiclass classification.
    Particularly effective for imbalanced datasets.
    """
    
    def __init__(self, smooth: float = 1.0, reduction: str = 'mean'):
        """
        Initialize Dice Loss.
        
        Args:
            smooth: Smoothing factor to avoid division by zero
            reduction: Reduction method
        """
        super().__init__()
        self.smooth = smooth
        self.reduction = reduction
        
    def forward(self, inputs: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        # Handle multiclass case
        if inputs.dim() > 1 and inputs.size(1) > 1:
            # Convert targets to float for multiclass
            targets = targets.float()
            # Convert to probabilities
            probs = F.softmax(inputs, dim=1)
            # Use class 1 probabilities for binary case
            probs_flat = probs[:, 1].view(-1)
            targets_flat = targets.view(-1)
        else:
            # Binary case
            probs = torch.sigmoid(inputs)
            probs_flat = probs.view(-1)
            targets_flat = targets.view(-1).float()
        
        # Calculate Dice coefficient
        intersection = (probs_flat * targets_flat).sum()
        dice = (2.0 * intersection + self.smooth) / (
            probs_flat.sum() + targets_flat.sum() + self.smooth
        )
        
        return 1.0 - dice


class UnifiedFocalLoss(nn.Module):
    """
    General-purpose Focal Loss for binary or multiclass classification,
    with optional class balancing (α) and focusing (γ).
    """
    def __init__(
        self,
        alpha: Union[float, str, List[float], torch.Tensor, None] = 'auto',
        gamma: float = 2.0,
        reduction: str = 'mean',
        binary: bool = False
    ):
        """
        Args:
            alpha: 
                - 'auto' to compute per-batch class weights
                - float (for binary): positive class weight (α1), α0 = 1 - α1
                - list or tensor of per-class weights
                - None: no weighting
            gamma: Focal loss focusing parameter
            reduction: 'mean' | 'sum' | 'none'
            binary: True for binary classification (uses sigmoid)
        """
        super().__init__()
        self.alpha_type = alpha
        self.gamma = gamma
        self.reduction = reduction
        self.binary = binary
        self.alpha = None

    def calculate_alpha(self, targets: torch.Tensor) -> Optional[torch.Tensor]:
        """Calculate alpha weights based on class distribution or manual config."""
        if self.alpha_type == 'auto':
            unique, counts = torch.unique(targets, return_counts=True)
            n_classes = targets.max().item() + 1
            n_samples = len(targets)

            weights = torch.zeros(n_classes, device=targets.device)
            for cls, count in zip(unique, counts):
                weights[cls] = n_samples / (n_classes * count.float())

            weights = weights / weights.mean()
            return weights

        elif isinstance(self.alpha_type, (list, torch.Tensor)):
            return torch.tensor(self.alpha_type, device=targets.device, dtype=torch.float32)

        elif isinstance(self.alpha_type, (int, float)):
            # For binary case
            return torch.tensor([1 - self.alpha_type, self.alpha_type], device=targets.device)

        return None

    def forward(self, inputs: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        """
        Compute the focal loss.
        
        Args:
            inputs: Logits, shape [B, C] or [B] for binary
            targets: Class indices (0 or 1 for binary)
        """
        if self.binary:
            # Binary case
            probs = torch.sigmoid(inputs)
            targets = targets.float()
            p_t = probs * targets + (1 - probs) * (1 - targets)

            bce = F.binary_cross_entropy_with_logits(inputs, targets, reduction='none')
            focal_weight = (1 - p_t) ** self.gamma

            if isinstance(self.alpha_type, (float, int)):
                alpha_t = self.alpha_type * targets + (1 - self.alpha_type) * (1 - targets)
                loss = alpha_t * focal_weight * bce
            else:
                loss = focal_weight * bce

        else:
            # Multiclass case
            ce_loss = F.cross_entropy(inputs, targets, reduction='none')
            p_t = torch.exp(-ce_loss)  # exp(-log(p_t)) = p_t
            focal_weight = (1 - p_t) ** self.gamma

            self.alpha = self.calculate_alpha(targets)

            if self.alpha is not None:
                alpha_t = self.alpha[targets]
                loss = alpha_t * focal_weight * ce_loss
            else:
                loss = focal_weight * ce_loss

        # Reduce
        if self.reduction == 'mean':
            return loss.mean()
        elif self.reduction == 'sum':
            return loss.sum()
        else:
            return loss

class CombinedLoss(nn.Module):
    """
    Combined loss that can use multiple loss functions with weights.
    Useful for multi-objective optimization.
    """
    
    def __init__(
        self,
        losses: List[nn.Module],
        weights: Optional[List[float]] = None
    ):
        """
        Initialize Combined Loss.
        
        Args:
            losses: List of loss functions to combine
            weights: Weights for each loss (default: equal weights)
        """
        super().__init__()
        self.losses = nn.ModuleList(losses)
        
        if weights is None:
            weights = [1.0] * len(losses)
        self.weights = weights
        
    def forward(self, inputs: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        """Calculate combined loss."""
        total_loss = 0
        
        for loss_fn, weight in zip(self.losses, self.weights):
            total_loss += weight * loss_fn(inputs, targets)
            
        return total_loss

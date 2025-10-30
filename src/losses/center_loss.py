"""Center Loss implementation for feature learning."""

import torch
import torch.nn as nn
from typing import Optional
from loguru import logger

class CenterBasedLoss(nn.Module):
    """Center loss implementation with proper center updates."""
    
    def __init__(self, num_classes: int, feat_dim: int, alpha: float = 0.5):
        super().__init__()
        self.num_classes = num_classes
        self.feat_dim = feat_dim
        self.alpha = alpha
        
        # Initialize centers as buffers
        self.register_buffer('centers', torch.randn(num_classes, feat_dim))
        nn.init.xavier_uniform_(self.centers)

    def update_centers(self, features: torch.Tensor, labels: torch.Tensor):
        """Update centers using exponential moving average."""
        with torch.no_grad():
            for class_idx in range(self.num_classes):
                class_mask = (labels == class_idx)
                
                if class_mask.sum() > 0:
                    class_features = features[class_mask]
                    class_center = class_features.mean(dim=0)                
                    self.centers[class_idx] = (1 - self.alpha) * self.centers[class_idx] + self.alpha * class_center

class CenterLoss(CenterBasedLoss):
    """Center loss - pulls features to class centers."""
    
    def forward(self, features: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
        """Compute center loss."""
        batch_centers = self.centers[labels]
        loss = torch.mean(torch.sum((features - batch_centers) ** 2, dim=1))  # squared Euclidean distance without sqrt
        return loss

class ContrastiveCenterLoss(CenterBasedLoss):
    """Contrastive center loss - pulls to own center, pushes from others."""
    
    def __init__(self, num_classes: int, feat_dim: int, 
                 alpha: float = 0.5, margin: float = 1.0, lambda_c: float = 0.5):
        super().__init__(num_classes, feat_dim, alpha)
        self.margin = margin
        self.lambda_c = lambda_c
    
    def forward(self, features: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
        """Compute contrastive center loss."""

        assert features.dim() == 2, "Features must be [B, D]"
        assert labels.dim() == 1, "Labels must be 1D"
        assert features.size(0) == labels.size(0), "Batch size mismatch between features and labels"
        assert labels.max().item() < self.num_classes, "Label index exceeds number of classes"
        assert labels.min().item() >= 0, "Label index must be non-negative"

        batch_size = features.size(0) # Get the number of samples in the batch (B). Assumes features shape is [B, D]
        
        # Center loss component
        batch_centers = self.centers[labels] # Gather the center vector corresponding to each sample's label; shape: [B, D]
        center_distances = torch.norm(features - batch_centers, p=2, dim=1) # Compute Euclidean distance (L2 norm)-with sqrt between each feature and its class center; shape: [B]
        center_loss = center_distances.mean() # Average over all center distances to get the center loss (pulls features toward their own center)
        
        # Contrastive component
        features_expanded = features.unsqueeze(1)     # shape: [B, 1, D]
        centers_expanded = self.centers.unsqueeze(0)  # shape: [1, C, D]
        # distances from each sample to each center shape: [B, C]
        all_distances = torch.norm(features_expanded - centers_expanded, p=2, dim=2) # the L2 norm (Euclidean distance) along the last dimension 
        
        mask = torch.ones_like(all_distances, dtype=torch.bool)
        mask[torch.arange(batch_size), labels] = False # keep all distances except the correct class center
        
        other_distances = all_distances.masked_fill(~mask, float('inf')) # replace the correct class distance with inf
        min_other_dist, _ = other_distances.min(dim=1) # For each sample, get the minimum distance to a wrong class center.
        
        contrastive_loss = torch.relu(center_distances - min_other_dist + self.margin).mean()
        
        return self.lambda_c * center_loss + (1 - self.lambda_c) * contrastive_loss

class RingLoss(nn.Module):
    """Ring Loss for feature normalization."""
    
    def __init__(self, radius: float = 1.0, ring_weight: float = 1):
        super(RingLoss, self).__init__()
        self.radius = radius
        self.ring_weight = ring_weight
        
    def forward(self, features: torch.Tensor) -> torch.Tensor:
        """Calculate ring loss.
        
        Args:
            features: Feature embeddings tensor
        Returns:
            Ring loss value
        """
        feature_norms = torch.norm(features, p=2, dim=1)
        loss = torch.mean((feature_norms - self.radius) ** 2)
        return self.ring_weight * loss
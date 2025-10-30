"""CustomNN neural network models for scattering data classification."""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, Any, Optional, List, Tuple
from loguru import logger
import math
import numpy as np
from .base import ModelFactory, safe_int, safe_float
from .pytorch_base import TorchModel


from ..losses import UnifiedFocalLoss, DiceLoss, ContrastiveCenterLoss

# ============================================================================
# PYTORCH MODELS
# ============================================================================

class CustomNN(nn.Module):
    """Ultra-lightweight neural network architecture."""
    
    def __init__(self, input_dim: int, dropout: float = 0.3, n_classes: int = 2):
        super().__init__()

        # Ensure input_dim is an integer for comparison
        input_dim = safe_int(input_dim, 10)
        dropout = safe_float(dropout, 0.3)
        n_classes = safe_int(n_classes, 2)
        
        self.feat_dim = 4

        # model structure
        in_channels = 1
        conv_out_channels = 4

        # Adjusted parameters for efficiency
        kernel_size = 3  # Smaller kernel
        padding = 0      # Add padding to maintain size
        dilation = 1     # dilation 1 for simplicity
        stride = 1       # Stride 2 to reduce dimensions
        
        # Single conv layer
        self.conv1 = nn.Conv1d(in_channels, conv_out_channels, kernel_size=kernel_size, 
                              padding=padding, dilation=dilation, 
                              stride=stride, bias=True)

        # Calculate output dimension
        conv_length = (input_dim + 2*padding - dilation*(kernel_size-1) - 1)//stride + 1
        conv_output_dim = conv_out_channels * conv_length

        self.act_conv = nn.LeakyReLU(0.01)

        # Minimal fully connected layers
        self.layer1 = nn.Linear(conv_output_dim, 4)
        # self.bn1 = nn.BatchNorm1d(8)
        self.act1 = nn.LeakyReLU(0.01)

        self.dropout    = nn.Dropout(dropout)
        self.classifier = nn.Linear(4, n_classes)

    def forward(self, x):
        """Standard forward pass."""
        # outputs, _ = self.forward_with_features(x)

        # Forward through network
        if len(x.shape) == 2:
            x = x.unsqueeze(1)
        
        x = self.conv1(x)
        x = self.act_conv(x)

        x = x.view(x.size(0), -1)
        x = self.layer1(x)
        # x = self.bn1(x)
        x = self.act1(x)
        x = self.dropout(x)

        outputs = self.classifier(x) # For classification loss

        return outputs

class NeuralNetworkModel(TorchModel):
    """Standard neural network with configurable architecture."""
    
    def __init__(self, config: Optional[Dict[str, Any]] = None):
        super().__init__(config)
        
        # Ensure numeric types using safe conversion
        input_dim = safe_int(config.get('input_dim', 1), 1)
        dropout = safe_float(config.get('dropout', 0.3), 0.3)
        n_classes = safe_int(config.get('n_classes', 2), 2)
        

        self.model = CustomNN(input_dim, dropout=dropout, n_classes=n_classes)

class ContrastiveNeuralNetworkModel(TorchModel):
    """Standard neural network with configurable architecture."""
    
    def __init__(self, config: Optional[Dict[str, Any]] = None):
        super().__init__(config)
        
        # Ensure numeric types using safe conversion
        input_dim = safe_int(config.get('input_dim', 1), 1)
        dropout = safe_float(config.get('dropout', 0.3), 0.3)
        n_classes = safe_int(config.get('n_classes', 2), 2)
        
        self.model = ContrastiveNN(input_dim, dropout=dropout, n_classes=n_classes)

class ContrastiveNN(nn.Module):
    """Ultra-lightweight neural network architecture."""
    
    def __init__(self, input_dim: int, dropout: float = 0.3, n_classes: int = 2):
        super().__init__()

        # Ensure input_dim is an integer for comparison
        input_dim = safe_int(input_dim, 10)
        dropout = safe_float(dropout, 0.3)
        n_classes = safe_int(n_classes, 2)
        
        self.feat_dim = 4

        # model structure
        in_channels = 1
        conv_out_channels = 4

        # Adjusted parameters for efficiency
        kernel_size = 3  # Smaller kernel
        padding = 0      # Add padding to maintain size
        dilation = 1     # dilation 1 for simplicity
        stride = 1       # Stride 2 to reduce dimensions
        
        # Single conv layer
        self.conv1 = nn.Conv1d(in_channels, conv_out_channels, kernel_size=kernel_size, 
                              padding=padding, dilation=dilation, 
                              stride=stride, bias=True)

        # Calculate output dimension
        conv_length = (input_dim + 2*padding - dilation*(kernel_size-1) - 1)//stride + 1
        conv_output_dim = conv_out_channels * conv_length

        self.act_conv = nn.LeakyReLU(0.01)

        # Minimal fully connected layers
        self.layer1 = nn.Linear(conv_output_dim, 4)
        # self.bn1 = nn.BatchNorm1d(8)
        self.act1 = nn.LeakyReLU(0.01)

        # Two branches
        self.feature_proj = nn.Linear(4, self.feat_dim)
        self.dropout    = nn.Dropout(dropout)
        self.classifier = nn.Linear(4, n_classes)

    def forward_with_features(self, x):
        """
        Forward pass returning outputs and features.
        Features are projected through a learned projection head.
        """
        if len(x.shape) == 2:
            x = x.unsqueeze(1)
        
        x = self.conv1(x)
        x = self.act_conv(x)

        x = x.view(x.size(0), -1)
        x = self.layer1(x)
        # x = self.bn1(x)
        x = self.act1(x)
        x = self.dropout(x)
        
        shared_repr = x  # Shared representation

        features = self.feature_proj(shared_repr)            # For center loss
        outputs = self.classifier(shared_repr) # For classification loss
        
        return outputs, features

    def forward(self, x):
        """Standard forward pass."""
        # outputs, _ = self.forward_with_features(x)

        outputs, _ = self.forward_with_features(x)
        return outputs

# ============================================================================
# TRANSFORMERS
# ============================================================================

class TransformerClassifier(nn.Module):
    """Transformer-based classifier."""
    
    def __init__(
        self, 
        input_dim: int, 
        d_model: int = 64, 
        nhead: int = 4, 
        num_layers: int = 2
    ):
        super(TransformerClassifier, self).__init__()
        
        # Ensure all parameters are the correct type
        input_dim = int(input_dim)
        d_model = int(d_model)
        nhead = int(nhead)
        num_layers = int(num_layers)
        
        self.input_projection = nn.Linear(input_dim, d_model)
        self.positional_encoding = nn.Parameter(torch.randn(1, 1, d_model))
        
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model, 
            nhead=nhead, 
            dim_feedforward=4,
            batch_first=True
        )
        self.transformer_encoder = nn.TransformerEncoder(encoder_layer, num_layers)
        
        self.classifier = nn.Sequential(
            nn.Linear(d_model, 2),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(2, 2)
        )
        
    def forward(self, x):
        batch_size = x.shape[0]
        x = x.unsqueeze(1)
        x = self.input_projection(x)
        x = x + self.positional_encoding
        
        x = self.transformer_encoder(x)
        x = x.squeeze(1)
        
        return self.classifier(x)


class TransformerModel(TorchModel):
    """Transformer-based classification model."""
    
    def __init__(self, config: Optional[Dict[str, Any]] = None):
        super().__init__(config)
        
        # Ensure numeric types using safe conversion
        input_dim = safe_int(config.get('input_dim', 1), 1)
        d_model = safe_int(config.get('d_model', 64), 64)
        nhead = safe_int(config.get('nhead', 4), 4)
        num_layers = safe_int(config.get('num_layers', 2), 2)
        
        self.model = TransformerClassifier(
            input_dim=input_dim,
            d_model=d_model,
            nhead=nhead,
            num_layers=num_layers
        )

# ============================================================================
# GRAPH NEURAL NETWORKS
# ============================================================================

class SimpleFeatureGraphConstructor(nn.Module):
    """Simple adjacency matrix constructor with stable statistics."""
    
    def __init__(self, input_dim: int):
        super().__init__()
        self.input_dim = input_dim
        
        # Register buffers for running statistics
        self.register_buffer('running_mean', torch.zeros(input_dim))
        self.register_buffer('running_std', torch.ones(input_dim))
        self.register_buffer('num_batches_tracked', torch.tensor(0, dtype=torch.long))
        self.momentum = 0.1
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Compute feature correlation-based adjacency matrix.
        x: [batch_size, input_dim]
        Returns: adj [input_dim, input_dim]
        """
        # Compute current batch statistics
        batch_mean = x.mean(dim=0)
        batch_std = x.std(dim=0) + 1e-8
        
        if self.training:
            # Update running statistics during training
            with torch.no_grad():
                if self.num_batches_tracked == 0:
                    self.running_mean.copy_(batch_mean)
                    self.running_std.copy_(batch_std)
                else:
                    self.running_mean.mul_(1 - self.momentum).add_(batch_mean, alpha=self.momentum)
                    self.running_std.mul_(1 - self.momentum).add_(batch_std, alpha=self.momentum)
                self.num_batches_tracked += 1
            
            # Use batch statistics during training
            mean = batch_mean
            std = batch_std
        else:
            # Use running statistics during evaluation
            mean = self.running_mean
            std = self.running_std
        
        # Normalize features
        x_norm = (x - mean) / std
        
        # Compute correlation matrix
        adj = torch.abs(torch.corrcoef(x_norm.T))
        
        # Add self-loops and normalize
        adj = adj + torch.eye(self.input_dim, device=adj.device)
        adj = adj / adj.sum(dim=1, keepdim=True)
        
        return adj


class SimpleGNNClassifier(nn.Module):
    """Simplified GNN Classifier for feature-level graph operations."""
    
    def __init__(
        self, 
        input_dim: int, 
        hidden_dim: int = 64, 
        num_classes: int = 2,
        dropout: float = 0.3
    ):
        super().__init__()
        
        self.input_dim = int(input_dim)
        self.hidden_dim = int(hidden_dim)
        self.num_classes = int(num_classes)
        
        # Graph constructor
        self.graph_constructor = SimpleFeatureGraphConstructor(input_dim)
        
        # Simple 3-layer architecture
        # Layer 1: Graph convolution
        self.gc_weight = nn.Parameter(torch.FloatTensor(input_dim, hidden_dim))
        self.gc_bias = nn.Parameter(torch.FloatTensor(hidden_dim))
        
        # Layer 2: Hidden layer
        self.hidden = nn.Linear(hidden_dim, hidden_dim)
        self.bn = nn.BatchNorm1d(hidden_dim)
        
        # Layer 3: Output layer
        self.output = nn.Linear(hidden_dim, num_classes)
        
        # Dropout
        self.dropout = nn.Dropout(dropout)
        
        # Initialize weights
        self.reset_parameters()
        
    def reset_parameters(self):
        """Initialize parameters."""
        stdv = 1. / math.sqrt(self.gc_weight.size(1))
        self.gc_weight.data.uniform_(-stdv, stdv)
        self.gc_bias.data.uniform_(-stdv, stdv)
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass.
        x: [batch_size, input_dim]
        """
        # Construct adjacency matrix
        adj = self.graph_constructor(x)
        
        # Graph convolution: aggregate features based on adjacency
        # First aggregate features: X' = X @ A^T
        x_agg = torch.matmul(x, adj.T)
        
        # Then apply linear transformation
        h = torch.matmul(x_agg, self.gc_weight) + self.gc_bias
        h = F.relu(h)
        h = self.dropout(h)
        
        # Hidden layer
        h = self.hidden(h)
        h = self.bn(h)
        h = F.relu(h)
        h = self.dropout(h)
        
        # Output layer
        output = self.output(h)
        
        return output
    
    def get_graph_metrics(self) -> dict:
        """Get metrics about the graph constructor statistics."""
        with torch.no_grad():
            return {
                'running_mean_avg': self.graph_constructor.running_mean.mean().item(),
                'running_std_avg': self.graph_constructor.running_std.mean().item(),
                'num_batches_tracked': self.graph_constructor.num_batches_tracked.item()
            }



class UltraSimpleGNNClassifier(nn.Module):
    """Ultra-simple GNN that just uses correlation as attention weights."""
    
    def __init__(
        self, 
        input_dim: int, 
        hidden_dim: int = 64, 
        num_classes: int = 2,
        dropout: float = 0.3
    ):
        super().__init__()
        
        # Learnable correlation weight
        self.correlation_weight = nn.Parameter(torch.ones(1))
        
        # Simple MLP
        self.mlp = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim),
            nn.BatchNorm1d(hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, num_classes)
        )
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass with correlation-weighted features.
        x: [batch_size, input_dim]
        """
        # Compute feature correlations
        x_norm = F.normalize(x, p=2, dim=0)
        corr = torch.abs(torch.matmul(x_norm.T, x_norm))
        
        # Use correlation to weight features
        corr_weighted = corr * self.correlation_weight
        x_weighted = torch.matmul(x, corr_weighted)
        
        # Pass through MLP
        return self.mlp(x_weighted)

class GraphNeuralNetworkModel(TorchModel):
    """Graph Neural Network model."""
    
    def __init__(self, config: Optional[Dict[str, Any]] = None):
        super().__init__(config)
        
        # Ensure numeric types using safe conversion
        input_dim = safe_int(config.get('input_dim', 1), 1)
        hidden_dim = safe_int(config.get('hidden_dim', 64), 64)
        
        self.model = SimpleGNNClassifier(
            input_dim=input_dim,
            hidden_dim=hidden_dim
        )

### loss function

class CombinedFocalContrastiveWithCrossModal(nn.Module):
    """
    Combined loss with focal, dice, contrastive center, and cross-modal contrastive.
    Integrates with existing loss functions from the framework.
    """
    
    def __init__(self, config: Dict[str, Any], num_classes: int, feat_dim: int):
        super().__init__()
        
    
        # Initialize focal loss
        self.focal_loss = UnifiedFocalLoss(
            alpha=config.get('alpha', 'auto'),
            gamma=config.get('gamma', 3.0),
            reduction='mean',
            binary=False
        )
        
        # Initialize dice loss
        self.dice_loss = DiceLoss(
            smooth=config.get('smooth', 2.0),
            reduction='mean'
        )
        
        # Initialize contrastive center loss
        self.contrastive_center_loss = ContrastiveCenterLoss(
            num_classes=num_classes,
            feat_dim=feat_dim,
            alpha=config.get('alpha_center', 0.5),
            margin=config.get('margin', 0.1),
            lambda_c=config.get('lambda_c', 0.3)
        )
        
        # Weights for combining losses
        self.weights = config.get('weights', [0.7, 0.1, 0.2])
        
        # Additional weight for cross-modal contrastive
        self.cross_modal_weight = config.get('cross_modal_weight', 0.1)
        self.temperature = config.get('temperature', 0.5)
        
    def forward(self, outputs, targets, features=None, contrastive_features=None):
        """
        Compute combined loss with cross-modal contrastive.
        
        Args:
            outputs: Model predictions (logits)
            targets: Ground truth labels
            features: Features for center loss
            contrastive_features: Dict of features for cross-modal contrastive
        """
        # Focal loss
        focal = self.focal_loss(outputs, targets)
        
        # Dice loss
        dice = self.dice_loss(outputs, targets)
        
        # Combine focal and dice
        total_loss = self.weights[0] * focal + self.weights[1] * dice
        
        # Contrastive center loss if features provided
        if features is not None:
            self.contrastive_center_loss.update_centers(features, targets)
            center_loss = self.contrastive_center_loss(features, targets)
            total_loss += self.weights[2] * center_loss
        
        # Cross-modal contrastive loss if provided
        if contrastive_features is not None:
            cross_modal_loss = self._compute_cross_modal_loss(contrastive_features)
            total_loss += self.cross_modal_weight * cross_modal_loss
        
        return total_loss
    
    def _compute_cross_modal_loss(self, features_dict):
        """Compute cross-modal contrastive loss between different modalities."""
        z_raw = features_dict['raw']
        z_nmf = features_dict['nmf']
        z_cross = features_dict['cross']
        
        # Normalize features
        z_raw = F.normalize(z_raw, p=2, dim=1)
        z_nmf = F.normalize(z_nmf, p=2, dim=1)
        z_cross = F.normalize(z_cross, p=2, dim=1)
        
        batch_size = z_raw.size(0)
        
        # Loss 1: Raw <-> NMF contrastive
        sim_raw_nmf = torch.matmul(z_raw, z_nmf.T) / self.temperature
        labels = torch.arange(batch_size, device=z_raw.device)
        loss_raw_nmf = F.cross_entropy(sim_raw_nmf, labels)
        
        # Loss 2: Raw <-> Cross contrastive
        sim_raw_cross = torch.matmul(z_raw, z_cross.T) / self.temperature
        loss_raw_cross = F.cross_entropy(sim_raw_cross, labels)
        
        # Loss 3: NMF <-> Cross contrastive
        sim_nmf_cross = torch.matmul(z_nmf, z_cross.T) / self.temperature
        loss_nmf_cross = F.cross_entropy(sim_nmf_cross, labels)
        
        # Combined loss
        total_loss = (loss_raw_nmf + loss_raw_cross + loss_nmf_cross) / 3.0
        
        return total_loss

####
class LightweightNMFSimCLRNetwork(nn.Module):
    """Lightweight SimCLR with cross-modal contrastive learning."""
    
    def __init__(self, raw_dim: int = 30, nmf_dim: int = 16, 
                 hidden_dims: list = [32, 16, 8],  # Reduced from [96, 48, 24, 12]
                 dropout: float = 0.3, n_classes: int = 2,
                 temperature: float = 0.5):
        super().__init__()
        
        self.raw_dim = raw_dim
        self.nmf_dim = nmf_dim
        self.temperature = temperature
        self.feat_dim = hidden_dims[-1]   # 8 instead of 12
        
        # Raw feature encoder (Conv1D + FC layers)
        # Conv for raw features
        self.raw_conv = nn.Conv1d(1, 4, kernel_size=3, padding=0, stride=1, bias=True)
        
        raw_conv_length = raw_dim - 2  # kernel=3, no padding
        raw_conv_output_dim = 4 * raw_conv_length
        
        # Simplified raw encoder - only 2 layers instead of 4
        self.raw_encoder = nn.Sequential(
            nn.Linear(raw_conv_output_dim, hidden_dims[-1]),  # 112 -> 32
            # nn.BatchNorm1d(hidden_dims[-1]),
            nn.LeakyReLU(0.01),
            nn.Dropout(dropout),

            # nn.Linear(hidden_dims[0], hidden_dims[-1]),  # 112 -> 32
            # # nn.BatchNorm1d(hidden_dims[-1]),
            # nn.LeakyReLU(0.01),
            # nn.Dropout(dropout),
        )
        
        # Simplified NMF encoder - only 2 layers
        self.nmf_encoder = nn.Sequential(
            nn.Linear(nmf_dim, hidden_dims[-1]),  # 16 -> 32
            # nn.BatchNorm1d(hidden_dims[-1]),
            nn.LeakyReLU(0.01),
            nn.Dropout(dropout),
            
            # nn.Linear(hidden_dims[0], hidden_dims[-1]),  # 32 -> 8
            # # nn.BatchNorm1d(hidden_dims[-1]),
            # nn.LeakyReLU(0.01)
        )
        
        # Simplified cross-modal attention
        self.cross_attention = nn.MultiheadAttention(
            embed_dim=hidden_dims[-1],  # 8
            num_heads=1,  # Single head for simplicity
            batch_first=True,
            dropout=dropout
        )
        
        # Simplified feature projections
        proj_dim = hidden_dims[-1]  # 4 instead of 6
        self.feature_proj_raw = nn.Linear(hidden_dims[-1], proj_dim)  # 8 -> 4
        self.feature_proj_nmf = nn.Linear(hidden_dims[-1], proj_dim)  # 8 -> 4
        self.feature_proj_cross = nn.Linear(hidden_dims[-1], self.feat_dim)  # 8 -> 8 (identity)
        
        # Classifier uses concatenated features
        # self.classifier = nn.Linear(hidden_dims[-1] * 2, n_classes)  # 16 -> 2
        
        self.classifier = nn.Sequential(
            nn.Linear(hidden_dims[-1] * 2, hidden_dims[-1]),  # 16 -> 32
            nn.LeakyReLU(0.01),
            nn.Dropout(dropout),
            
            nn.Linear(hidden_dims[-1], n_classes),  # 32 -> 8
        )

        self.dropout = nn.Dropout(dropout)
    
    def forward_with_features(self, x_raw, x_nmf):
        """Forward pass with cross-modal contrastive features."""
        # Encode raw features through Conv1D
        if len(x_raw.shape) == 2:
            x_raw = x_raw.unsqueeze(1)
        
        h_raw = self.raw_conv(x_raw)
        h_raw = F.leaky_relu(h_raw, 0.01)
        h_raw = h_raw.view(h_raw.size(0), -1)
        
        # Continue with raw encoder
        h_raw = self.raw_encoder(h_raw)
        
        # Encode NMF features
        h_nmf = self.nmf_encoder(x_nmf)
        
        # Cross-modal attention
        h_raw_unsq = h_raw.unsqueeze(1)  # (batch, 1, 8)
        h_nmf_unsq = h_nmf.unsqueeze(1)  # (batch, 1, 8)
        
        h_cross, _ = self.cross_attention(
            query=h_nmf_unsq,
            key=h_raw_unsq,
            value=h_raw_unsq
        )
        h_cross = h_cross.squeeze(1)  # (batch, 8)

        # Apply dropout
        h_raw   = self.dropout(h_raw)
        h_nmf   = self.dropout(h_nmf)
        h_cross = self.dropout(h_cross)
        
        # Project features for contrastive losses
        z_raw   =  self.feature_proj_raw(h_raw)      # 8 -> 4  
        z_nmf   =  self.feature_proj_nmf(h_nmf)      # 8 -> 4
        z_cross =  self.feature_proj_cross(h_cross) # 8 -> 8
        
        # Classification using concatenated representations
        h_combined = torch.cat([h_raw, h_cross], dim=1)  # 16 dims
        outputs = self.classifier(h_combined)
        
        # Return features
        contrastive_features = {
            'raw': z_raw,
            'nmf': z_nmf,
            'cross': z_cross
        }
        
        return outputs, z_cross, contrastive_features
    
    def forward(self, x_raw, x_nmf):
        """Standard forward pass for inference."""
        outputs, _, _ = self.forward_with_features(x_raw, x_nmf)
        return outputs


class LightweightNMFSimCLRModel(TorchModel):
    """Lightweight NMF-SimCLR model wrapper."""
    
    def __init__(self, config: Optional[Dict[str, Any]] = None):
        super().__init__(config)
        
        self.uses_nmf = True
        
        # Model dimensions
        self.raw_dim = safe_int(config.get('raw_dim', 30), 30)
        self.nmf_dim = safe_int(config.get('nmf_dim', 16), 16)
        self.hidden_dims = config.get('hidden_dims', [16, 8])  # Lightweight dims
        dropout = safe_float(config.get('dropout', 0.3), 0.3)
        n_classes = safe_int(config.get('n_classes', 2), 2)
        temperature = safe_float(config.get('temperature', 0.5), 0.5)
        
        # Initialize lightweight network
        self.model = LightweightNMFSimCLRNetwork(
            raw_dim=self.raw_dim,
            nmf_dim=self.nmf_dim,
            hidden_dims=self.hidden_dims,
            dropout=dropout,
            n_classes=n_classes,
            temperature=temperature
        )
        
        self.feat_dim = self.model.feat_dim
        

    def _initialize_training_components(self, num_classes: int, learning_rate: float):
        """Initialize loss functions, optimizer, and scheduler."""
        # First, call parent to set up optimizer and scheduler
        super()._initialize_training_components(num_classes, learning_rate)
        
        # Then override the criterion with our custom loss
        loss_params = {
            'alpha': self.config.get('focal_alpha', 'auto'),
            'gamma': self.config.get('focal_gamma', 3.0),
            'smooth': self.config.get('dice_smooth', 2.0),
            'alpha_center': self.config.get('center_alpha', 0.5),
            'margin': self.config.get('contrastive_margin', 0.1),
            'lambda_c': self.config.get('contrastive_lambda', 0.3),
            'weights': self.config.get('loss_weights', [0.7, 0.1, 0.2]),
            'cross_modal_weight': self.config.get('cross_modal_weight', 0.1),
            'temperature': self.config.get('temperature', 0.5)
        }

        # Create the custom loss
        self.criterion = CombinedFocalContrastiveWithCrossModal(
            loss_params, num_classes, self.feat_dim
        )
        
        # Move criterion to device (parent might have done this, but it's safe to repeat)
        self.criterion.to(self.device)
        
        # Log the loss change
        logger.info(f"Overrode criterion with {self.criterion.__class__.__name__}")
    
    def _forward_pass(self, X: torch.Tensor, X_nmf: Optional[torch.Tensor] = None) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        """Forward pass for lightweight model."""
        if X_nmf is None:
            raise ValueError("Lightweight NMF-SimCLR requires NMF features")
        
        outputs, features, self.contrastive_features = self.model.forward_with_features(X, X_nmf)
        return outputs, features
    
    def _compute_loss(self, outputs: torch.Tensor, targets: torch.Tensor,
                     features: Optional[torch.Tensor] = None) -> torch.Tensor:
        """Compute loss with cross-modal contrastive features."""
        if isinstance(self.criterion, CombinedFocalContrastiveWithCrossModal):
            return self.criterion(outputs, targets, features, 
                                getattr(self, 'contrastive_features', None))
        else:
            return super()._compute_loss(outputs, targets, features)
    
    def predict(self, X: np.ndarray, X_nmf: Optional[np.ndarray] = None) -> np.ndarray:
        """Generate predictions."""
        if X_nmf is None:
            raise ValueError("Lightweight NMF-SimCLR requires NMF features")
        
        return super().predict(X, X_nmf)
    
    def predict_proba(self, X: np.ndarray, X_nmf: Optional[np.ndarray] = None) -> np.ndarray:
        """Generate probability predictions."""
        if X_nmf is None:
            raise ValueError("Lightweight NMF-SimCLR requires NMF features")
        
        return super().predict_proba(X, X_nmf)





### graph
class CustomGCNConv(nn.Module):
    """
    Custom Graph Convolutional Layer without torch_geometric dependency.
    Implements the GCN layer: H' = σ(D^(-1/2) A D^(-1/2) H W)
    """
    
    def __init__(self, in_features, out_features, bias=True):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        
        # Weight matrix
        self.weight = nn.Parameter(torch.FloatTensor(in_features, out_features))
        if bias:
            self.bias = nn.Parameter(torch.FloatTensor(out_features))
        else:
            self.register_parameter('bias', None)
        
        # Initialize parameters
        self.reset_parameters()
    
    def reset_parameters(self):
        # Xavier initialization
        stdv = 1. / np.sqrt(self.weight.size(1))
        self.weight.data.uniform_(-stdv, stdv)
        if self.bias is not None:
            self.bias.data.uniform_(-stdv, stdv)
    
    def forward(self, x, adj):
        """
        x: Node features [num_nodes, in_features]
        adj: Adjacency matrix [num_nodes, num_nodes]
        """
        # Add self-loops to adjacency matrix
        num_nodes = adj.size(0)
        adj = adj + torch.eye(num_nodes, device=adj.device)
        
        # Compute degree matrix
        degree = adj.sum(1)
        d_inv_sqrt = degree.pow(-0.5)
        d_inv_sqrt[torch.isinf(d_inv_sqrt)] = 0.
        
        # Normalize adjacency matrix: D^(-1/2) A D^(-1/2)
        d_mat_inv_sqrt = torch.diag(d_inv_sqrt)
        norm_adj = torch.mm(torch.mm(d_mat_inv_sqrt, adj), d_mat_inv_sqrt)
        
        # Graph convolution: (normalized_adj * x) * weight
        support = torch.mm(x, self.weight)
        output = torch.mm(norm_adj, support)
        
        if self.bias is not None:
            output = output + self.bias
        
        return output


class GraphNMFSimCLRNetworkCustom(nn.Module):
    """Graph-based SimCLR with custom GCN implementation."""
    
    def __init__(self, raw_dim: int = 30, nmf_dim: int = 16, 
                 hidden_dims: list = [32, 16, 8],
                 dropout: float = 0.3, n_classes: int = 2,
                 temperature: float = 0.5):
        super().__init__()
        
        self.raw_dim = raw_dim
        self.nmf_dim = nmf_dim
        self.temperature = temperature
        self.feat_dim = hidden_dims[-1]
        
        # Graph construction parameters
        self.k_neighbors = 5
        
        # Raw feature encoder
        self.raw_conv = nn.Conv1d(1, 4, kernel_size=3, padding=0, stride=1, bias=True)
        raw_conv_length = raw_dim - 2
        raw_conv_output_dim = 4 * raw_conv_length
        
        # Custom GCN layers for raw features
        self.raw_gcn1 = CustomGCNConv(raw_conv_output_dim, hidden_dims[0])
        self.raw_gcn2 = CustomGCNConv(hidden_dims[0], hidden_dims[-1])
        
        # Custom GCN layers for NMF features
        self.nmf_gcn1 = CustomGCNConv(nmf_dim, hidden_dims[0])
        self.nmf_gcn2 = CustomGCNConv(hidden_dims[0], hidden_dims[-1])
        
        # Cross-modal attention
        self.cross_attention = nn.MultiheadAttention(
            embed_dim=hidden_dims[-1],
            num_heads=1,
            batch_first=True,
            dropout=dropout
        )
        
        # Feature projections
        proj_dim = hidden_dims[-1]
        self.feature_proj_raw = nn.Linear(hidden_dims[-1], proj_dim)
        self.feature_proj_nmf = nn.Linear(hidden_dims[-1], proj_dim)
        self.feature_proj_cross = nn.Linear(hidden_dims[-1], self.feat_dim)
        
        # Classifier
        self.classifier = nn.Linear(hidden_dims[-1] * 2, n_classes)
        
        self.dropout = nn.Dropout(dropout)
    
    def construct_adjacency_matrix(self, features, k=5):
        """
        Construct adjacency matrix based on k-NN or similarity.
        Returns a normalized adjacency matrix.
        """
        batch_size = features.size(0)
        
        # Compute pairwise distances
        features_norm = F.normalize(features, p=2, dim=1)
        similarity = torch.mm(features_norm, features_norm.t())
        
        # Method 1: k-NN based adjacency
        if k > 0 and k < batch_size:
            # Keep only top-k similarities for each node
            _, indices = torch.topk(similarity, k + 1, dim=1)
            
            # Create adjacency matrix
            adj = torch.zeros_like(similarity)
            for i in range(batch_size):
                adj[i, indices[i]] = similarity[i, indices[i]]
            
            # Make symmetric
            adj = (adj + adj.t()) / 2
        else:
            # Method 2: Threshold-based adjacency
            threshold = 0.5
            adj = (similarity > threshold).float() * similarity
        
        return adj
    
    def apply_graph_conv(self, x, gcn1, gcn2):
        """Apply graph convolution layers with adjacency matrix."""
        # Construct adjacency matrix
        adj = self.construct_adjacency_matrix(x, k=self.k_neighbors)
        
        # Apply GCN layers
        h = gcn1(x, adj)
        h = F.leaky_relu(h, 0.01)
        h = self.dropout(h)
        
        h = gcn2(h, adj)
        h = F.leaky_relu(h, 0.01)
        
        return h
    
    def forward_with_features(self, x_raw, x_nmf):
        """Forward pass with cross-modal contrastive features."""
        # Process raw features through Conv1D
        if len(x_raw.shape) == 2:
            x_raw = x_raw.unsqueeze(1)
        
        h_raw = self.raw_conv(x_raw)
        h_raw = F.leaky_relu(h_raw, 0.01)
        h_raw = h_raw.view(h_raw.size(0), -1)
        
        # Apply graph convolution
        h_raw = self.apply_graph_conv(h_raw, self.raw_gcn1, self.raw_gcn2)
        h_nmf = self.apply_graph_conv(x_nmf, self.nmf_gcn1, self.nmf_gcn2)
        
        # Cross-modal attention
        h_raw_unsq = h_raw.unsqueeze(1)
        h_nmf_unsq = h_nmf.unsqueeze(1)
        
        h_cross, _ = self.cross_attention(
            query=h_nmf_unsq,
            key=h_raw_unsq,
            value=h_raw_unsq
        )
        h_cross = h_cross.squeeze(1)
        
        # Apply dropout
        h_raw = self.dropout(h_raw)
        h_nmf = self.dropout(h_nmf)
        h_cross = self.dropout(h_cross)
        
        # Project features
        z_raw = self.feature_proj_raw(h_raw)
        z_nmf = self.feature_proj_nmf(h_nmf)
        z_cross = self.feature_proj_cross(h_cross)
        
        # Classification
        h_combined = torch.cat([h_raw, h_cross], dim=1)
        outputs = self.classifier(h_combined)
        
        contrastive_features = {
            'raw': z_raw,
            'nmf': z_nmf,
            'cross': z_cross
        }
        
        return outputs, z_cross, contrastive_features
    
    def forward(self, x_raw, x_nmf):
        """Standard forward pass for inference."""
        outputs, _, _ = self.forward_with_features(x_raw, x_nmf)
        return outputs


# Updated Model Wrapper
class GraphNMFSimCLRModel(TorchModel):
    """Graph-based NMF-SimCLR model wrapper with custom GCN."""
    
    def __init__(self, config: Optional[Dict[str, Any]] = None):
        super().__init__(config)
        
        self.uses_nmf = True
        
        # Model dimensions
        self.raw_dim = safe_int(config.get('raw_dim', 30), 30)
        self.nmf_dim = safe_int(config.get('nmf_dim', 16), 16)
        self.hidden_dims = config.get('hidden_dims', [32, 16, 8])
        dropout = safe_float(config.get('dropout', 0.3), 0.3)
        n_classes = safe_int(config.get('n_classes', 2), 2)
        temperature = safe_float(config.get('temperature', 0.5), 0.5)
        
        # Use custom GCN implementation
        self.model = GraphNMFSimCLRNetworkCustom(
            raw_dim=self.raw_dim,
            nmf_dim=self.nmf_dim,
            hidden_dims=self.hidden_dims,
            dropout=dropout,
            n_classes=n_classes,
            temperature=temperature
        )
        
        self.feat_dim = self.model.feat_dim
        
    def _initialize_training_components(self, num_classes: int, learning_rate: float):
        """Initialize loss functions, optimizer, and scheduler."""
        super()._initialize_training_components(num_classes, learning_rate)
        
        loss_params = {
            'alpha': self.config.get('focal_alpha', 'auto'),
            'gamma': self.config.get('focal_gamma', 3.0),
            'smooth': self.config.get('dice_smooth', 2.0),
            'alpha_center': self.config.get('center_alpha', 0.5),
            'margin': self.config.get('contrastive_margin', 0.1),
            'lambda_c': self.config.get('contrastive_lambda', 0.3),
            'weights': self.config.get('loss_weights', [0.7, 0.1, 0.2]),
            'cross_modal_weight': self.config.get('cross_modal_weight', 0.1),
            'temperature': self.config.get('temperature', 0.5)
        }
        
        self.criterion = CombinedFocalContrastiveWithCrossModal(
            loss_params, num_classes, self.feat_dim
        )
        self.criterion.to(self.device)
        
        logger.info(f"Initialized Graph NMF-SimCLR with custom GCN and {self.criterion.__class__.__name__}")
    
    def _forward_pass(self, X: torch.Tensor, X_nmf: Optional[torch.Tensor] = None) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        """Forward pass for graph-based model."""
        if X_nmf is None:
            raise ValueError("Graph NMF-SimCLR requires NMF features")
        
        outputs, features, self.contrastive_features = self.model.forward_with_features(X, X_nmf)
        return outputs, features
    
    def _compute_loss(self, outputs: torch.Tensor, targets: torch.Tensor,
                     features: Optional[torch.Tensor] = None) -> torch.Tensor:
        """Compute loss with cross-modal contrastive features."""
        if isinstance(self.criterion, CombinedFocalContrastiveWithCrossModal):
            return self.criterion(outputs, targets, features, 
                                getattr(self, 'contrastive_features', None))
        else:
            return super()._compute_loss(outputs, targets, features)
    
    def predict(self, X: np.ndarray, X_nmf: Optional[np.ndarray] = None) -> np.ndarray:
        """Generate predictions."""
        if X_nmf is None:
            raise ValueError("Graph NMF-SimCLR requires NMF features")
        
        return super().predict(X, X_nmf)
    
    def predict_proba(self, X: np.ndarray, X_nmf: Optional[np.ndarray] = None) -> np.ndarray:
        """Generate probability predictions."""
        if X_nmf is None:
            raise ValueError("Graph NMF-SimCLR requires NMF features")
        
        return super().predict_proba(X, X_nmf)


# Register the model
# ModelFactory.register_model('nmf_simclr', SimpleNMFModel)  # SimpleNMFModel  NMFSimCLRModel


# Register models with factory
ModelFactory.register_model('contrastive_nn', ContrastiveNeuralNetworkModel)
ModelFactory.register_model('nmf', LightweightNMFSimCLRModel )  # LightweightNMFSimCLRModel
ModelFactory.register_model('neural_network', NeuralNetworkModel)
ModelFactory.register_model('transformer', TransformerModel)
ModelFactory.register_model('gnn', GraphNeuralNetworkModel)
ModelFactory.register_model('gnn_simclr', GraphNMFSimCLRModel)
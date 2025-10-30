"""Advanced neural network models for scattering data classification."""

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset
from typing import Dict, Any, Optional, Tuple
from dataclasses import dataclass
from loguru import logger

from .base import BaseModel, ModelFactory
from .pytorch_base import TorchModel, OptimizerFactory, SchedulerFactory


# ============================================================================
# HELPER FUNCTIONS
# ============================================================================

def safe_int(value, default=1):
    """Safely convert value to int."""
    try:
        return int(value)
    except (TypeError, ValueError):
        return default

def safe_float(value, default=0.0):
    """Safely convert value to float."""
    try:
        return float(value)
    except (TypeError, ValueError):
        return default


# ============================================================================
# CONFIGURATION FOR SMALL INPUTS
# ============================================================================

@dataclass
class ModelConfig:
    """Configuration for advanced models"""
    # Common parameters - ULTRA REDUCED
    n_points: int = 210    # Changed from 210
    embed_dim: int = 8   # Minimal embedding
    num_heads: int = 2   # Minimum for attention
    num_layers: int = 2  # Single layer
    dropout: float = 0.1
    
    # VAE specific
    latent_dim: int = 4  # Match input dim
    encoder_channels: list = None
    decoder_channels: list = None
    kernel_size: int = 3  # Small kernel
    
    def __post_init__(self):
        if self.encoder_channels is None:
            self.encoder_channels = [16, 32, 64]  # Reduced channels
        if self.decoder_channels is None:
            self.decoder_channels = [64, 32, 16]



class SimpleAttention(nn.Module):
    """Simplified attention for small inputs"""
    def __init__(self, input_dim):
        super().__init__()
        self.attention = nn.Sequential(
            nn.Linear(input_dim, input_dim),
            nn.Tanh(),
            nn.Linear(input_dim, 1)
        )
        
    def forward(self, x):
        # x shape: [batch, features]
        weights = torch.softmax(self.attention(x), dim=-1)
        return x * weights, weights


class AdvancedContrastiveClassifier(nn.Module):
    """Fixed classifier for 16-30 dimensional inputs"""
    def __init__(self, input_dim=30, hidden_dim=32, embed_dim=16,
                 num_classes=2, dropout_rate=0.2, use_attention=True):
        super().__init__()
        
        # Ensure all numeric parameters are the correct type
        input_dim = int(input_dim)
        hidden_dim = int(hidden_dim)
        embed_dim = int(embed_dim)
        num_classes = int(num_classes)
        dropout_rate = float(dropout_rate)
        
        self.input_dim = input_dim
        self.embed_dim = embed_dim  # Store for contrastive loss
        self.use_attention = use_attention
        
        # Simple attention (no positional encoding needed for features)
        if self.use_attention:
            self.attention = SimpleAttention(input_dim)
        
        # Encoder
        self.encoder = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.BatchNorm1d(hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout_rate),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.BatchNorm1d(hidden_dim // 2),
            nn.ReLU()
        )
        
        # Embedding layer (for contrastive loss)
        self.embed = nn.Linear(hidden_dim // 2, embed_dim)
        self.embed_norm = nn.LayerNorm(embed_dim)
        
        # Classification head
        self.classifier = nn.Linear(embed_dim, num_classes)
        
        # For attention visualization
        self.last_attention_weights = None
        
    def forward(self, x, return_embed=False):
        # Apply attention if enabled
        if self.use_attention:
            x, attention_weights = self.attention(x)
            self.last_attention_weights = attention_weights.detach()
        
        # Encode
        encoded = self.encoder(x)
        
        # Generate embeddings
        embeddings = self.embed(encoded)
        embeddings = self.embed_norm(embeddings)
        embeddings_normalized = F.normalize(embeddings, p=2, dim=1)
        
        # Classification
        outputs = self.classifier(embeddings)
        
        if return_embed:
            return outputs, embeddings_normalized
        return outputs
    
    def get_embeddings(self, x):
        """Get embeddings without classification outputs"""
        if self.use_attention:
            x, attention_weights = self.attention(x)
            self.last_attention_weights = attention_weights.detach()
        
        encoded = self.encoder(x)
        embeddings = self.embed(encoded)
        embeddings = self.embed_norm(embeddings)
        return F.normalize(embeddings, p=2, dim=1)


class AdvancedContrastiveModel(TorchModel):
    """Fixed Contrastive Model that properly integrates with your existing framework"""
    
    def __init__(self, config: Optional[Dict[str, Any]] = None):
        # Initialize parent class
        super().__init__(config)
        
        # Ensure numeric types using safe conversion
        input_dim = self.safe_int(config.get('input_dim', 30), 30)
        dropout = self.safe_float(config.get('dropout', 0.3), 0.3)
        n_classes = self.safe_int(config.get('n_classes', 2), 2)
        
        # Set appropriate dimensions for 16-30 features
        if input_dim <= 20:
            hidden_dim = 32
            embed_dim = 8
        else:
            hidden_dim = 64
            embed_dim = 16
        
        # Override with config if provided
        hidden_dim = self.safe_int(config.get('hidden_dim', hidden_dim), hidden_dim)
        embed_dim = self.safe_int(config.get('embed_dim', embed_dim), embed_dim)
        dropout_rate = self.safe_float(config.get('dropout_rate', dropout), dropout)
        use_attention = config.get('use_attention', True)
        
        # Create the neural network
        # Use whatever attribute your base class expects (self.model, self.net, etc.)
        self.model = AdvancedContrastiveClassifier(
            input_dim=input_dim,
            hidden_dim=hidden_dim,
            embed_dim=embed_dim,
            num_classes=n_classes,
            dropout_rate=dropout_rate,
            use_attention=use_attention
        )
        
        # Store feature dimension for loss functions
        self.feat_dim = embed_dim
        
        # For contrastive loss
        self.use_contrastive = config.get('use_contrastive', False)
        self.contrastive_weight = self.safe_float(config.get('contrastive_weight', 0.1), 0.1)
    
    def forward(self, x):
        """Forward pass - required by parent class"""
        return self.model(x)
    
    def forward_with_features(self, x):
        """Forward pass with features for combined loss"""
        return self.model(x, return_embed=True)
    
    def get_embeddings(self, x):
        """Get embeddings for center loss"""
        return self.model.get_embeddings(x)
    
    def safe_int(self, value, default):
        """Safely convert value to int with fallback"""
        try:
            return int(value) if value is not None else default
        except (ValueError, TypeError):
            return default
    
    def safe_float(self, value, default):
        """Safely convert value to float with fallback"""
        try:
            return float(value) if value is not None else default
        except (ValueError, TypeError):
            return default





# ============================================================================
# PHYSICS-INFORMED COMPONENTS
# ============================================================================

class PhysicsRefinementLayer(nn.Module):
    """Ultra-simple physics layer for 4D input"""
    
    def __init__(self, config: ModelConfig):
        super().__init__()
        self.config = config
        
        # For small inputs, use simpler refinement
        if config.n_points <= 10:
            # Simple feature enhancement
            self.refine = nn.Sequential(
                nn.Linear(config.n_points, 8),
                nn.ReLU(),
                nn.Linear(8, config.n_points),
                nn.Sigmoid()
            )
        else:
            # Original refinement for larger inputs
            self.refine = nn.Sequential(
                nn.Linear(config.n_points + 2, 64),
                nn.ReLU(),
                nn.Linear(64, config.n_points),
                nn.Sigmoid()
            )
        
        # Learnable physics parameters
        self.guinier_weight = nn.Parameter(torch.tensor(1.0))
        self.porod_weight = nn.Parameter(torch.tensor(1.0))
        
    def forward(self, x, mask):
        batch_size = x.shape[0]
        
        # For small inputs, skip physics scores
        if self.config.n_points <= 10:
            # Simple residual enhancement
            refined = self.refine(x)
            return x + 0.1 * refined
        
        # Original implementation for larger inputs
        if x.shape[1] != self.config.n_points:
            if x.shape[1] < self.config.n_points:
                x = F.pad(x, (0, self.config.n_points - x.shape[1]))
            else:
                x = x[:, :self.config.n_points]
        
        # Calculate physics features
        guinier_score = self.calculate_guinier_score(x)
        porod_score = self.calculate_porod_score(x)
        
        # Ensure mask has the same shape as x
        if mask.shape[1] != x.shape[1]:
            if mask.shape[1] < x.shape[1]:
                mask = F.pad(mask, (0, x.shape[1] - mask.shape[1]))
            else:
                mask = mask[:, :x.shape[1]]
        
        # Concatenate with physics scores
        physics_features = torch.cat([
            x,
            guinier_score.unsqueeze(1),
            porod_score.unsqueeze(1)
        ], dim=1)
        
        # Refine
        refined = self.refine(physics_features)
        
        # Ensure masked regions are preserved from input
        output = x * (1 - mask) + refined * mask
        
        return output
    
    def calculate_guinier_score(self, x):
        """Calculate Guinier law compliance"""
        # For small inputs, use all features
        if x.shape[1] <= 10:
            return torch.mean(x, dim=1)
        return torch.mean(x[:, :50], dim=1)
    
    def calculate_porod_score(self, x):
        """Calculate Porod law compliance"""
        # For small inputs, use all features
        if x.shape[1] <= 10:
            return torch.mean(x, dim=1)
        return torch.mean(x[:, -50:], dim=1)


class PhysicsInformedClassifier(TorchModel):
    """Minimal physics-informed neural network for small inputs"""
    
    def __init__(self, config: Optional[Dict[str, Any]] = None):
        # Initialize parent class FIRST
        super().__init__(config)
        
        # Ensure learning_rate is properly set from config with type conversion
        self.learning_rate = safe_float(config.get('learning_rate', 0.001), 0.001)
        
        input_dim = safe_int(config.get('input_dim', 4), 4)
        
        # Adjust hidden dimension based on input size
        if input_dim <= 10:
            hidden_dim = 16
        else:
            hidden_dim = safe_int(config.get('hidden_dim', 128), 128)
        
        # Create model config for physics layer
        model_config = ModelConfig(n_points=input_dim)
        
        # Physics enhancement
        self.physics_layer = PhysicsRefinementLayer(model_config)
        
        # Simple classifier
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(hidden_dim, 2)
        )
    
    def forward(self, x):
        """Forward pass with physics enhancement"""
        # Create dummy mask (no masking for classification)
        mask = torch.zeros_like(x)
        
        # Apply physics refinement
        x_refined = self.physics_layer(x, mask)
        
        # Classify
        return self.net(x_refined)
    
    def train(self, X: np.ndarray, y: np.ndarray) -> None:
        """Train with physics constraints"""
        # Setup losses and optimizer from config
        num_classes = len(np.unique(y))
        self._setup_losses(num_classes)
        
        # Need to add physics layer parameters to optimizer
        all_params = list(self.net.parameters()) + list(self.physics_layer.parameters())
        self.optimizer = OptimizerFactory.create_optimizer(
            self.optimizer_config, 
            all_params, 
            self.learning_rate
        )
        
        if self.scheduler_config:
            self.scheduler = SchedulerFactory.create_scheduler(
                self.scheduler_config, 
                self.optimizer
            )
        
        self.net.to(self.device)
        self.physics_layer.to(self.device)
        self.net.train()
        self.physics_layer.train()
        
        dataloader = self._prepare_data(X, y)
        
        for epoch in range(self.epochs):
            epoch_loss = 0
            for batch_X, batch_y in dataloader:
                batch_X = batch_X.to(self.device)
                batch_y = batch_y.to(self.device)
                
                self.optimizer.zero_grad()
                outputs = self.forward(batch_X)
                loss = self.criterion(outputs, batch_y)
                loss.backward()
                self.optimizer.step()
                
                epoch_loss += loss.item()
            
            if (epoch + 1) % 10 == 0:
                avg_loss = epoch_loss / len(dataloader)
                logger.debug(f"Epoch [{epoch+1}/{self.epochs}], Loss: {avg_loss:.4f}")
        
        self.fitted = True
        logger.info(f"{self.model_name} trained successfully")
    
    def predict(self, X: np.ndarray) -> np.ndarray:
        """Make predictions with physics refinement"""
        if not self.fitted:
            raise ValueError("Model must be trained before prediction")
        
        self.net.eval()
        self.physics_layer.eval()
        X_tensor = torch.FloatTensor(X).to(self.device)
        
        with torch.no_grad():
            outputs = self.forward(X_tensor)
            _, predicted = torch.max(outputs.data, 1)
        
        return predicted.cpu().numpy()


# Register models with factory
ModelFactory.register_model('advanced_contrastive', AdvancedContrastiveModel)
ModelFactory.register_model('physics_informed', PhysicsInformedClassifier)
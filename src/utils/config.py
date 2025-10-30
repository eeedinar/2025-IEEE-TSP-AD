"""Configuration management for ML framework."""

import yaml
import copy
from typing import Dict, Optional, Any, Union
from pathlib import Path


class Config:
    """
    YAML configuration loader that preserves all keys while expanding references.
    
    String references (e.g., loss: 'focal_loss') are expanded to full configs
    from their respective sections. All other values remain untouched.
    """
    
    def __init__(self, config_path: Union[str, Path]):
        """Load configuration from YAML file."""
        self.config_path = Path(config_path)
        with open(self.config_path, 'r') as f:
            self.config = yaml.safe_load(f)
    
    def get_model_config(self, model_name: str) -> Dict[str, Any]:
        """
        Get model configuration with expanded references.
        
        Args:
            model_name: Model name from 'models' section
            
        Returns:
            Deep copy of model config with string references expanded
            
        Raises:
            ValueError: If model not found
        """
        models = self.config.get('models', {})
        if model_name not in models:
            raise ValueError(f"Model '{model_name}' not found")
        
        # Deep copy preserves all nested structures including early_stopping
        config = copy.deepcopy(models[model_name])
        
        # Expand string references to named configurations
        for key in ['loss', 'optimizer', 'scheduler']:
            if isinstance(config.get(key), str):
                config[key] = self._get_named_config(key + 's', config[key])
        
        # Handle multi_loss references
        if 'multi_loss' in config:
            multi = config['multi_loss']
            if isinstance(multi.get('primary'), str):
                multi['primary'] = self._get_named_config('losses', multi['primary'])
            if 'auxiliary' in multi:
                multi['auxiliary'] = [
                    self._get_named_config('losses', aux) if isinstance(aux, str) else aux
                    for aux in multi['auxiliary']
                ]
        
        return config
    
    def _get_named_config(self, section: str, name: str) -> Optional[Dict[str, Any]]:
        """
        Get named configuration from a section.
        
        Args:
            section: Config section name (e.g., 'losses', 'optimizers')
            name: Configuration name
            
        Returns:
            Config dict with 'type' field added, or minimal config if not found
        """
        configs = self.config.get(section, {})
        
        if name not in configs:
            # Return None for schedulers, minimal config for others
            return None if section == 'schedulers' else {'type': name}
        
        config = copy.deepcopy(configs[name])
        config['type'] = name
        return config
    
    # Simple getters for other sections
    def get_data_config(self) -> Dict[str, Any]:
        """Get data configuration."""
        return self.config.get('data', {})
    
    def get_training_config(self) -> Dict[str, Any]:
        """Get training configuration."""
        return self.config.get('training', {})
    
    def get_evaluation_config(self) -> Dict[str, Any]:
        """Get evaluation configuration."""
        return self.config.get('evaluation', {})
    
    def get_visualization_config(self) -> Dict[str, Any]:
        """Get visualization configuration."""
        return self.config.get('visualization', {})
    
    def get_output_config(self) -> Dict[str, Any]:
        """Get output configuration."""
        return self.config.get('output', {})
    
    # Direct access methods for factories
    def get_loss_config(self, name: str) -> Dict[str, Any]:
        """Get loss configuration by name."""
        return self._get_named_config('losses', name) or {'type': name}
    
    def get_optimizer_config(self, name: str) -> Dict[str, Any]:
        """Get optimizer configuration by name."""
        return self._get_named_config('optimizers', name) or {'type': name}
    
    def get_scheduler_config(self, name: str) -> Optional[Dict[str, Any]]:
        """Get scheduler configuration by name."""
        return self._get_named_config('schedulers', name)
    
    def __getitem__(self, key: str) -> Any:
        """Allow dict-like access to config."""
        return self.config[key]
    
    def get(self, key: str, default: Any = None) -> Any:
        """Get configuration value with default."""
        return self.config.get(key, default)


class ModelConfigBuilder:
    """Simple builder for creating model configurations programmatically."""
    
    def __init__(self):
        self.config = {}
    
    def set(self, **kwargs) -> 'ModelConfigBuilder':
        """Set any configuration values."""
        self.config.update(kwargs)
        return self
    
    def set_loss(self, loss_name: str, **loss_params) -> 'ModelConfigBuilder':
        """Set loss configuration."""
        self.config['loss'] = {'type': loss_name, **loss_params}
        return self
    
    def set_optimizer(self, optimizer_name: str, **optimizer_params) -> 'ModelConfigBuilder':
        """Set optimizer configuration."""
        self.config['optimizer'] = {'type': optimizer_name, **optimizer_params}
        return self
    
    def set_scheduler(self, scheduler_name: str, **scheduler_params) -> 'ModelConfigBuilder':
        """Set scheduler configuration."""
        self.config['scheduler'] = {'type': scheduler_name, **scheduler_params}
        return self
    
    def build(self) -> Dict[str, Any]:
        """Return the built configuration."""
        return copy.deepcopy(self.config)
"""
Configuration module for mathematics-for-ai.
"""

import yaml
import os
from pathlib import Path
from typing import Any, Optional


class Config:
    """Configuration manager for the library."""
    
    _instance = None
    _config = {}
    
    def __new__(cls):
        if cls._instance is None:
            cls._instance = super().__new__(cls)
        return cls._instance
    
    def __init__(self):
        if not self._config:
            self._load_default_config()
    
    def _load_default_config(self):
        """Load default configuration."""
        config_path = Path(__file__).parent.parent.parent / 'config' / 'config.yaml'
        
        if config_path.exists():
            with open(config_path, 'r') as f:
                self._config = yaml.safe_load(f)
        else:
            self._config = self._get_default_config()
    
    def _get_default_config(self) -> dict:
        """Get default configuration values."""
        return {
            'random_seed': 42,
            'precision': 'float64',
            'optimization': {
                'default_learning_rate': 0.001,
                'max_iterations': 1000,
                'tolerance': 1e-6,
                'early_stopping': True,
                'patience': 50
            },
            'models': {
                'linear_regression': {
                    'fit_intercept': True,
                    'method': 'gradient_descent'
                },
                'neural_network': {
                    'weight_init': 'he',
                    'optimizer': 'adam'
                }
            },
            'visualization': {
                'figsize': (10, 8),
                'dpi': 300,
                'animation_fps': 30
            }
        }
    
    def get(self, key: str, default: Any = None) -> Any:
        """Get configuration value by dot-notation key."""
        keys = key.split('.')
        value = self._config
        
        for k in keys:
            if isinstance(value, dict) and k in value:
                value = value[k]
            else:
                return default
        
        return value
    
    def set(self, key: str, value: Any):
        """Set configuration value."""
        keys = key.split('.')
        config = self._config
        
        for k in keys[:-1]:
            if k not in config:
                config[k] = {}
            config = config[k]
        
        config[keys[-1]] = value
    
    def update(self, config_dict: dict):
        """Update configuration with dictionary."""
        self._update_nested(self._config, config_dict)
    
    def _update_nested(self, base: dict, update: dict):
        """Recursively update nested dictionary."""
        for key, value in update.items():
            if key in base and isinstance(base[key], dict) and isinstance(value, dict):
                self._update_nested(base[key], value)
            else:
                base[key] = value
    
    def reset(self):
        """Reset to default configuration."""
        self._config = self._get_default_config()
    
    @property
    def random_seed(self) -> int:
        return self._config.get('random_seed', 42)
    
    @random_seed.setter
    def random_seed(self, value: int):
        self._config['random_seed'] = value
    
    @property
    def learning_rate(self) -> float:
        return self._config.get('optimization', {}).get('default_learning_rate', 0.001)
    
    @property
    def max_iterations(self) -> int:
        return self._config.get('optimization', {}).get('max_iterations', 1000)


# Global configuration instance
config = Config()


def get_config(key: str, default: Any = None) -> Any:
    """Get configuration value."""
    return config.get(key, default)


def set_config(key: str, value: Any):
    """Set configuration value."""
    config.set(key, value)


def load_config(filepath: str):
    """Load configuration from YAML file."""
    with open(filepath, 'r') as f:
        config_dict = yaml.safe_load(f)
    config.update(config_dict)


def save_config(filepath: str):
    """Save current configuration to YAML file."""
    with open(filepath, 'w') as f:
        yaml.dump(config._config, f, default_flow_style=False)

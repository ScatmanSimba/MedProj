"""
Configuration loader for the project.

This module provides functions for loading and validating configuration files.
"""

import yaml
from pathlib import Path
from typing import Dict, Any, Optional
import logging
from .schemas import FeatureConfig

logger = logging.getLogger(__name__)

def load_feature_config(config_path: Optional[Path] = None) -> FeatureConfig:
    """Load and validate feature configuration.
    
    Args:
        config_path: Optional path to config file. If None, uses default path.
        
    Returns:
        Validated FeatureConfig object
        
    Raises:
        ValueError: If config file is invalid or missing required fields
        FileNotFoundError: If config file does not exist
    """
    if config_path is None:
        config_path = Path('configs/feature_config.yaml')
    
    if not config_path.exists():
        raise FileNotFoundError(f"Config file not found at {config_path}")
    
    try:
        with open(config_path, 'r') as f:
            config_dict = yaml.safe_load(f)
        
        # Validate config using Pydantic model
        config = FeatureConfig(**config_dict)
        logger.info(f"Successfully loaded and validated config from {config_path}")
        return config
        
    except Exception as e:
        logger.error(f"Error loading config from {config_path}: {e}")
        raise

def save_feature_config(config: FeatureConfig, config_path: Optional[Path] = None) -> None:
    """Save feature configuration to file.
    
    Args:
        config: FeatureConfig object to save
        config_path: Optional path to save config file. If None, uses default path.
        
    Raises:
        ValueError: If config is invalid
        IOError: If file cannot be written
    """
    if config_path is None:
        config_path = Path('configs/feature_config.yaml')
    
    try:
        # Convert to dict and save as YAML
        config_dict = config.dict()
        with open(config_path, 'w') as f:
            yaml.dump(config_dict, f, default_flow_style=False, sort_keys=False)
        
        logger.info(f"Successfully saved config to {config_path}")
        
    except Exception as e:
        logger.error(f"Error saving config to {config_path}: {e}")
        raise

def get_default_config() -> FeatureConfig:
    """Get default feature configuration.
    
    Returns:
        FeatureConfig object with default values
    """
    return FeatureConfig(
        confidence_weights={
            'base': 0.3,
            'consistency': 0.3,
            'causality': 0.3,
            'temporal': 0.3,
            'effect': 0.4,
            'uncertainty_penalty': 0.2,
            'side_effect_boost': 0.1,
            'cluster_size': 0.2,
            'dimension': 1.0
        },
        causal_confidence_weights={
            'base': 0.3,
            'pattern': 0.3,
            'temporal': 0.2,
            'event': 0.2
        },
        dimension_weights={
            'activation': 1.0,
            'emotional': 1.0,
            'metabolic': 1.0
        },
        confidence_thresholds={
            'floor': 0.2,
            'ceiling': 0.95,
            'min_signal': 0.15
        },
        proximity={
            'decay_factor': 0.8,
            'max_distance': 10
        },
        cross_sentence={
            'window_size': 2,
            'decay_factor': 0.7,
            'min_confidence': 0.3,
            'max_distance': 5
        },
        temporal_recency_weights={
            'current': 1.0,
            'recent': 0.8,
            'past': 0.5,
            'distant': 0.2
        },
        coreference={
            'enabled': True,
            'max_distance': 5,
            'min_confidence': 0.6,
            'pronouns': [
                "it", "this", "that", "they", "them", "these", "those",
                "its", "itself", "one", "ones", "which", "who", "whom", "whose"
            ],
            'medication_specific': {
                'enabled': True,
                'min_confidence': 0.7,
                'max_distance': 3,
                'context_window': 2
            },
            'logging': {
                'level': "DEBUG",
                'track_resolutions': True,
                'track_confidence': True
            }
        },
        uncertainty_weights={
            'epistemic': 0.6,
            'aleatoric': 0.4
        },
        cache={
            'causality_cache_size': 10000,
            'fuzzy_cache_size': 5000,
            'temporal_cache_size': 5000,
            'emoji_cache_size': 1000
        },
        word2vec={
            'enabled': False,
            'model_path': Path('data/models/word2vec-google-news-300.bin'),
            'version': "1.0",
            'memory_map': True
        },
        symptom_matcher={
            'lexicon_path': Path('data/lexicons/symptom_lexicon.yaml'),
            'fuzzy_threshold': 0.85,
            'max_fuzzy_terms': 1000,
            'cache': {
                'fuzzy_cache_size': 5000
            }
        },
        temporal_parser={
            'max_distance': 10,
            'decay_factor': 0.8,
            'temporal_patterns': [
                {'pattern': "started taking", 'weight': 0.9},
                {'pattern': "stopped taking", 'weight': 0.9},
                {'pattern': "been on", 'weight': 0.8},
                {'pattern': "tried", 'weight': 0.7}
            ]
        },
        emoji_processor={
            'max_distance': 5,
            'decay_factor': 0.9,
            'emoji_categories': {
                'activation': ["😴", "😫", "😤"],
                'emotional': ["😊", "😢", "😡"],
                'metabolic': ["😋", "🤢", "😴"]
            }
        },
        causal_patterns={
            'direct': [
                {'pattern': "makes me feel", 'weight': 0.9},
                {'pattern': "causes me to", 'weight': 0.9},
                {'pattern': "results in", 'weight': 0.8}
            ],
            'indirect': [
                {'pattern': "seems to help with", 'weight': 0.7},
                {'pattern': "appears to reduce", 'weight': 0.7},
                {'pattern': "might be helping", 'weight': 0.6}
            ]
        },
        debug={
            'log_level': "INFO",
            'detailed_confidence': False,
            'cache_stats': False
        },
        logging={
            'log_dir': Path('logs'),
            'module_levels': {
                'features.symptom_matcher': "INFO",
                'features.response_attribution': "INFO",
                'features.temporal_parser': "INFO",
                'features.emoji_processor': "INFO",
                'models.confidence_aware_training': "INFO",
                'models.main_pipeline': "INFO",
                'evaluation.evaluate_confidence': "INFO"
            }
        }
    ) 
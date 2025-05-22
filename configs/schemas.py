"""
Configuration schemas for the project.

This module defines Pydantic models for validating configuration files.
"""

from typing import Dict, List, Optional, Union, Any
from pydantic import BaseModel, Field, validator
from pathlib import Path

class ConfidenceWeights(BaseModel):
    """Confidence weights configuration."""
    base: float = Field(ge=0.0, le=1.0)
    consistency: float = Field(ge=0.0, le=1.0)
    causality: float = Field(ge=0.0, le=1.0)
    temporal: float = Field(ge=0.0, le=1.0)
    effect: float = Field(ge=0.0, le=1.0)
    uncertainty_penalty: float = Field(ge=0.0, le=1.0)
    side_effect_boost: float = Field(ge=0.0, le=1.0)
    cluster_size: float = Field(ge=0.0, le=1.0)
    dimension: float = Field(ge=0.0, le=1.0)

class CausalConfidenceWeights(BaseModel):
    """Causal confidence weights configuration."""
    base: float = Field(ge=0.0, le=1.0)
    pattern: float = Field(ge=0.0, le=1.0)
    temporal: float = Field(ge=0.0, le=1.0)
    event: float = Field(ge=0.0, le=1.0)

class DimensionWeights(BaseModel):
    """Dimension weights configuration."""
    activation: float = Field(ge=0.0, le=1.0)
    emotional: float = Field(ge=0.0, le=1.0)
    metabolic: float = Field(ge=0.0, le=1.0)

class ConfidenceThresholds(BaseModel):
    """Confidence thresholds configuration."""
    floor: float = Field(ge=0.0, le=1.0)
    ceiling: float = Field(ge=0.0, le=1.0)
    min_signal: float = Field(ge=0.0, le=1.0)

class ProximitySettings(BaseModel):
    """Proximity settings configuration."""
    decay_factor: float = Field(ge=0.0, le=1.0)
    max_distance: int = Field(ge=1)

class CrossSentenceSettings(BaseModel):
    """Cross-sentence settings configuration."""
    window_size: int = Field(ge=1)
    decay_factor: float = Field(ge=0.0, le=1.0)
    min_confidence: float = Field(ge=0.0, le=1.0)
    max_distance: int = Field(ge=1)

class TemporalRecencyWeights(BaseModel):
    """Temporal recency weights configuration."""
    current: float = Field(ge=0.0, le=1.0)
    recent: float = Field(ge=0.0, le=1.0)
    past: float = Field(ge=0.0, le=1.0)
    distant: float = Field(ge=0.0, le=1.0)

class CoreferenceSettings(BaseModel):
    """Coreference settings configuration."""
    enabled: bool
    max_distance: int = Field(ge=1)
    min_confidence: float = Field(ge=0.0, le=1.0)
    pronouns: List[str]
    medication_specific: Dict[str, Any]
    logging: Dict[str, Any]

class UncertaintyWeights(BaseModel):
    """Uncertainty weights configuration."""
    epistemic: float = Field(ge=0.0, le=1.0)
    aleatoric: float = Field(ge=0.0, le=1.0)

class CacheSettings(BaseModel):
    """Cache settings configuration."""
    causality_cache_size: int = Field(ge=1)
    fuzzy_cache_size: int = Field(ge=1)
    temporal_cache_size: int = Field(ge=1)
    emoji_cache_size: int = Field(ge=1)
    compression_threshold: int = Field(default=1024, ge=0)  # 1KB default
    warmup_size: int = Field(default=100, ge=0)
    eviction_policy: str = Field(default="lru_freq", regex="^(lru|lru_freq)$")
    track_stats: bool = Field(default=True)
    compression_enabled: bool = Field(default=True)
    warmup_enabled: bool = Field(default=True)

class Word2VecSettings(BaseModel):
    """Word2Vec settings configuration."""
    enabled: bool
    model_path: Optional[Path]
    version: str
    memory_map: bool

class SymptomMatcherSettings(BaseModel):
    """Symptom matcher settings configuration."""
    lexicon_path: Path
    fuzzy_threshold: float = Field(ge=0.0, le=1.0)
    max_fuzzy_terms: int = Field(ge=1)
    cache: Dict[str, Any]

class TemporalParserSettings(BaseModel):
    """Temporal parser settings configuration."""
    max_distance: int = Field(ge=1)
    decay_factor: float = Field(ge=0.0, le=1.0)
    temporal_patterns: List[Dict[str, Any]]

class EmojiProcessorSettings(BaseModel):
    """Emoji processor settings configuration."""
    max_distance: int = Field(ge=1)
    decay_factor: float = Field(ge=0.0, le=1.0)
    emoji_categories: Dict[str, List[str]]

class CausalPatterns(BaseModel):
    """Causal patterns configuration."""
    direct: List[Dict[str, Any]]
    indirect: List[Dict[str, Any]]

class DebugSettings(BaseModel):
    """Debug settings configuration."""
    log_level: str
    detailed_confidence: bool
    cache_stats: bool

class LoggingSettings(BaseModel):
    """Logging settings configuration."""
    log_dir: Path
    module_levels: Dict[str, str]

class FeatureConfig(BaseModel):
    """Main feature configuration model."""
    confidence_weights: ConfidenceWeights
    causal_confidence_weights: CausalConfidenceWeights
    dimension_weights: DimensionWeights
    confidence_thresholds: ConfidenceThresholds
    proximity: ProximitySettings
    cross_sentence: CrossSentenceSettings
    temporal_recency_weights: TemporalRecencyWeights
    coreference: CoreferenceSettings
    uncertainty_weights: UncertaintyWeights
    cache: CacheSettings
    word2vec: Word2VecSettings
    symptom_matcher: SymptomMatcherSettings
    temporal_parser: TemporalParserSettings
    emoji_processor: EmojiProcessorSettings
    causal_patterns: CausalPatterns
    debug: DebugSettings
    logging: LoggingSettings

    @validator('logging')
    def validate_log_levels(cls, v):
        """Validate log levels."""
        valid_levels = {'DEBUG', 'INFO', 'WARNING', 'ERROR', 'CRITICAL'}
        for module, level in v.module_levels.items():
            if level.upper() not in valid_levels:
                raise ValueError(f"Invalid log level '{level}' for module '{module}'")
        return v

    @validator('word2vec')
    def validate_model_path(cls, v):
        """Validate Word2Vec model path."""
        if v.enabled and v.model_path and not v.model_path.exists():
            raise ValueError(f"Word2Vec model path does not exist: {v.model_path}")
        return v

    @validator('symptom_matcher')
    def validate_lexicon_path(cls, v):
        """Validate symptom lexicon path."""
        if not v.lexicon_path.exists():
            raise ValueError(f"Symptom lexicon path does not exist: {v.lexicon_path}")
        return v 
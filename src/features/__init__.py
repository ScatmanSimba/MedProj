"""Feature extraction package for psychiatric medication response prediction.

This package provides modules for extracting and processing features from text,
including symptom matching, temporal parsing, emoji processing, and response attribution.
"""

from src.features.symptom_matcher import SymptomMatcher
from src.features.temporal_parser import TemporalParser
from src.features.emoji_processor import EmojiProcessor
from src.features.response_attribution import ResponseAttributor

__all__ = [
    'SymptomMatcher',
    'TemporalParser',
    'EmojiProcessor',
    'ResponseAttributor'
] 
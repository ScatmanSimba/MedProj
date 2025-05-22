"""
Psychiatric Medication Response Predictor

This package contains modules for:
- Data collection from Reddit
- Medication name extraction and normalization
- Feature engineering
- Model training and evaluation
"""

__version__ = '0.1.0'
__author__ = 'MedProj Team'

# Import main classes for easier access
from .snapshot import RedditDataSnapshot
from .med_dictionary1803 import MedDictionary
from .features import (
    SymptomMatcher,
    TemporalParser,
    EmojiProcessor,
    ResponseAttributor
)

__all__ = [
    'RedditDataSnapshot',
    'MedDictionary',
    'SymptomMatcher',
    'TemporalParser',
    'EmojiProcessor',
    'ResponseAttributor'
] 
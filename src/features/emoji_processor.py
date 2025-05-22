"""Emoji processing module for psychiatric medication response prediction.

This module provides functionality for processing emoji signals in text.
"""

from typing import List, Dict, Any, Optional, Tuple, Union
import spacy
from spacy.tokens import Doc, Token, Span
import logging
from collections import OrderedDict
from functools import lru_cache
import hashlib
import re
from datetime import datetime, timedelta
import threading
from lru import LRUCache

logger = logging.getLogger(__name__)

class EmojiProcessor:
    """Processor for emoji signals in text."""
    
    def __init__(self, config: Dict[str, Any], debug: bool = False):
        """Initialize the emoji processor.
        
        Args:
            config: Configuration dictionary
            debug: Whether to enable debug logging
        """
        self.config = config
        self.debug = debug
        self.logger = logging.getLogger(__name__)
        
        # Initialize spaCy
        try:
            self.nlp = spacy.load("en_core_web_sm")
            self.logger.info("Successfully loaded spaCy model")
        except Exception as e:
            self.logger.error(f"Failed to load spaCy model: {e}")
            raise
        
        # Initialize caches
        cache_config = config.get('cache_settings', {})
        self.cache = LRUCache(
            maxsize=cache_config.get('emoji_cache_size', 5000),
            compression_threshold=cache_config.get('compression_threshold', 1024),
            warmup_size=cache_config.get('warmup_size', 100),
            eviction_policy=cache_config.get('eviction_policy', 'lru_freq')
        )
        
        # Initialize locks
        self.cache_lock = threading.RLock()
        
        # Initialize emoji mappings
        self.emoji_mappings = self._load_emoji_mappings()
        
        # Initialize user emoji maps
        self.user_emoji_maps = {
            'activation': {},
            'emotional': {},
            'metabolic': {}
        }
        
        # Load user emoji maps if available
        if 'user_emoji_maps' in config:
            self.user_emoji_maps.update(config['user_emoji_maps'])
    
    def extract_all_emoji_signals(self, text_or_doc: Union[str, Doc]) -> List[Dict[str, Any]]:
        """Extract all emoji signals from text.
        
        Args:
            text_or_doc: Input text or spaCy Doc object
            
        Returns:
            List of emoji signals with their properties
        """
        # Convert string to Doc if needed
        if isinstance(text_or_doc, str):
            doc = self.nlp(text_or_doc)
        else:
            doc = text_or_doc
        
        # Check cache first
        cache_key = hash(doc.text)
        with self.cache_lock:
            if cache_key in self.cache:
                return self.cache[cache_key]
        
        # Extract emoji signals
        signals = []
        for token in doc:
            if token._.is_emoji:
                # Get emoji properties
                emoji_info = self._get_emoji_info(token)
                if emoji_info:
                    signals.append(emoji_info)
        
        # Cache results
        with self.cache_lock:
            self.cache[cache_key] = signals
        
        return signals
    
    def get_attribution_ready_emoji_map(self, doc: Doc, medications: List[str]) -> Tuple[bool, Dict[str, Dict[str, List[Dict[str, Any]]]]]:
        """Get emoji signals organized by medication and dimension.
        
        Args:
            doc: spaCy Doc object
            medications: List of medications
            
        Returns:
            Tuple of (has_emoji, emoji_signals_by_med)
        """
        # Extract all emoji signals
        signals = self.extract_all_emoji_signals(doc)
        
        if not signals:
            return False, {}
        
        # Organize signals by medication and dimension
        emoji_signals = {med: {
            'activation': [],
            'emotional': [],
            'metabolic': []
        } for med in medications}
        
        # Process each signal
        for signal in signals:
            # Find closest medication
            closest_med = self._find_closest_medication(doc, signal['position'][0], medications)
            if closest_med:
                # Add signal to appropriate dimension
                dimension = signal['dimension']
                if dimension in emoji_signals[closest_med]:
                    emoji_signals[closest_med][dimension].append(signal)
        
        return True, emoji_signals
    
    def _get_emoji_info(self, token: Token) -> Optional[Dict[str, Any]]:
        """Get information about an emoji token.
        
        Args:
            token: Emoji token
            
        Returns:
            Dictionary with emoji information or None if not found
        """
        emoji_text = token.text
        
        # Check user maps first
        for dimension in ['activation', 'emotional', 'metabolic']:
            if emoji_text in self.user_emoji_maps[dimension]:
                return {
                    'text': emoji_text,
                    'dimension': dimension,
                    'polarity': self.user_emoji_maps[dimension][emoji_text]['polarity'],
                    'confidence': self.user_emoji_maps[dimension][emoji_text]['confidence'],
                    'position': (token.i, token.i + 1)
                }
        
        # Check default mappings
        for dimension, mappings in self.emoji_mappings.items():
            if emoji_text in mappings:
                return {
                    'text': emoji_text,
                    'dimension': dimension,
                    'polarity': mappings[emoji_text]['polarity'],
                    'confidence': mappings[emoji_text]['confidence'],
                    'position': (token.i, token.i + 1)
                }
        
        return None
    
    def _find_closest_medication(self, doc: Doc, token_idx: int, medications: List[str]) -> Optional[str]:
        """Find the medication closest to a token.
        
        Args:
            doc: spaCy Doc
            token_idx: Token index
            medications: List of medications
            
        Returns:
            Closest medication name or None
        """
        closest_med = None
        min_distance = float('inf')
        
        for med in medications:
            # Find medication in text
            for ent in doc.ents:
                if ent.label_ == "MEDICATION" and ent.text.lower() == med.lower():
                    # Calculate distance
                    distance = min(abs(token_idx - ent.start), abs(token_idx - ent.end))
                    if distance < min_distance:
                        min_distance = distance
                        closest_med = med
        
        return closest_med
    
    def _load_emoji_mappings(self) -> Dict[str, Dict[str, Dict[str, float]]]:
        """Load emoji mappings from configuration.
        
        Returns:
            Dictionary mapping dimensions to emoji mappings
        """
        mappings = {
            'activation': {},
            'emotional': {},
            'metabolic': {}
        }
        
        # Load from config
        if 'emoji_mappings' in self.config:
            for dimension, emoji_map in self.config['emoji_mappings'].items():
                if dimension in mappings:
                    mappings[dimension].update(emoji_map)
        
        return mappings
    
    def clear_cache(self) -> None:
        """Clear the emoji processing cache."""
        with self.cache_lock:
            self.cache.clear()
    
    def get_cache_stats(self) -> Dict[str, Any]:
        """Get cache statistics.
        
        Returns:
            Dictionary with cache statistics
        """
        return self.cache.get_stats()
    
    def __del__(self):
        """Cleanup when object is destroyed."""
        self.clear_cache() 
"""Symptom matching module for psychiatric medication response prediction.

This module provides functionality for matching symptoms in text using various
matching strategies including exact, fuzzy, and semantic matching.
"""

from typing import List, Dict, Any, Optional, Set, Tuple, Union
import spacy
from spacy.tokens import Doc, Token, Span
from spacy.matcher import Matcher, PhraseMatcher
import logging
from collections import OrderedDict
from functools import lru_cache
import yaml
from pathlib import Path
import pickle
import os
import numpy as np
import re
import unicodedata
from tqdm import tqdm
import threading
from gensim.models import Word2Vec
from ..config.config_loader import load_feature_config
from ..utils.cache_utils import EnhancedLRUCache

logger = logging.getLogger(__name__)

# Module-level singleton for Word2Vec model with thread safety
_word2vec_model = None
_word2vec_lock = threading.Lock()
_word2vec_loading = False

def get_word2vec_model(use_word2vec: bool = True, model_path: Optional[str] = None) -> Optional['KeyedVectors']:
    """Get or load the Word2Vec model.
    
    Args:
        use_word2vec: Whether to use Word2Vec for semantic matching
        model_path: Optional path to pre-downloaded model file
        
    Returns:
        Loaded Word2Vec model or None if use_word2vec is False
    """
    if not use_word2vec:
        return None
        
    global _word2vec_model, _word2vec_loading
    
    with _word2vec_lock:
        if _word2vec_model is None and not _word2vec_loading:
            _word2vec_loading = True
            try:
                import gensim.downloader as api
                from gensim.models import KeyedVectors
                
                if model_path and os.path.exists(model_path):
                    logger.info(f"Loading Word2Vec model from {model_path}")
                    _word2vec_model = KeyedVectors.load_word2vec_format(model_path, binary=True)
                else:
                    logger.info("Downloading Word2Vec model...")
                    _word2vec_model = api.load('word2vec-google-news-300')
                
                logger.info("Word2Vec model loaded successfully")
            except Exception as e:
                logger.warning(f"Failed to load Word2Vec model: {e}")
                _word2vec_model = None
            finally:
                _word2vec_loading = False
    
    return _word2vec_model

# Load dimension mappings from YAML
def load_dimension_mappings() -> Dict[str, Dict[str, Any]]:
    """Load dimension mappings from YAML configuration.
    
    Returns:
        Dictionary of dimension mappings
    """
    config = load_feature_config()
    return config.symptom_dimensions

# Load semantic expansions from YAML
def load_semantic_expansions() -> Dict[str, Dict[str, List[str]]]:
    """Load semantic expansions from YAML configuration.
    
    Returns:
        Dictionary of semantic expansions
    """
    config = load_feature_config()
    return config.semantic_expansions

class SymptomMatcher:
    """Matches symptoms in text using word embeddings and pattern matching."""
    
    def __init__(self, config: Dict[str, Any], debug: bool = False):
        """Initialize the symptom matcher.
        
        Args:
            config: Configuration dictionary
            debug: Whether to enable debug logging
        """
        self.config = config
        self.debug = debug
        self.logger = logging.getLogger(__name__)
        
        # Initialize caches
        cache_config = config.get('cache_settings', {})
        self.cache = EnhancedLRUCache(
            maxsize=cache_config.get('fuzzy_cache_size', 5000),
            compression_threshold=cache_config.get('compression_threshold', 1024),
            warmup_size=cache_config.get('warmup_size', 100),
            eviction_policy=cache_config.get('eviction_policy', 'lru_freq')
        )
        
        # Initialize vector cache for Word2Vec
        self.vector_cache = EnhancedLRUCache(
            maxsize=cache_config.get('vector_cache_size', 10000),
            compression_threshold=cache_config.get('compression_threshold', 1024),
            warmup_size=cache_config.get('warmup_size', 100),
            eviction_policy=cache_config.get('eviction_policy', 'lru_freq')
        )
        
        # Initialize locks
        self.cache_lock = threading.RLock()
        self.vector_cache_lock = threading.RLock()
        
        # Load components
        self.lexicon = self._load_lexicon()
        self.patterns = self._compile_patterns()
        
        # Initialize Word2Vec model if enabled
        self._word2vec_model = None
        if config.get('use_word2vec', False):
            self._word2vec_model = get_word2vec_model(
                use_word2vec=True,
                model_path=config.get('word2vec_model_path')
            )
            # Pre-compute vectors for lexicon
            self._precompute_lexicon_vectors()
    
    def _precompute_lexicon_vectors(self) -> None:
        """Pre-compute Word2Vec vectors for lexicon entries."""
        if not self.word2vec_model:
            return
            
        for symptom in self.lexicon:
            words = symptom.lower().split()
            try:
                # Get vectors for all words
                vectors = [self.word2vec_model.wv[word] for word in words 
                         if word in self.word2vec_model.wv]
                if vectors:
                    # Store average vector
                    self.vector_cache[symptom] = np.mean(vectors, axis=0)
            except Exception as e:
                self.logger.debug(f"Error pre-computing vector for {symptom}: {e}")
    
    def _get_symptom_vector(self, text: str) -> Optional[np.ndarray]:
        """Get Word2Vec vector for text, using cache if available.
        
        Args:
            text: Input text
            
        Returns:
            Word2Vec vector or None if not available
        """
        if not self.word2vec_model:
            return None
            
        # Check vector cache first
        with self.vector_cache_lock:
            if text in self.vector_cache:
                return self.vector_cache[text]
        
        # Compute vector
        words = text.lower().split()
        try:
            vectors = [self.word2vec_model.wv[word] for word in words 
                     if word in self.word2vec_model.wv]
            if vectors:
                vector = np.mean(vectors, axis=0)
                # Cache result
                with self.vector_cache_lock:
                    self.vector_cache[text] = vector
                return vector
        except Exception as e:
            self.logger.debug(f"Error computing vector for {text}: {e}")
        
        return None
    
    def _batch_compute_similarity(self, text: str, 
                                potential_matches: List[Dict[str, Any]]) -> List[float]:
        """Compute similarity scores for multiple matches in batch.
        
        Args:
            text: Input text
            potential_matches: List of potential matches
            
        Returns:
            List of similarity scores
        """
        if not self.word2vec_model:
            return [0.0] * len(potential_matches)
            
        # Get vector for input text
        text_vector = self._get_symptom_vector(text)
        if text_vector is None:
            return [0.0] * len(potential_matches)
        
        # Get vectors for all matches
        match_vectors = []
        for match in potential_matches:
            vector = self._get_symptom_vector(match['text'])
            match_vectors.append(vector if vector is not None else text_vector)
        
        # Compute similarities in batch
        similarities = []
        for vector in match_vectors:
            if vector is not None:
                similarity = np.dot(text_vector, vector) / (
                    np.linalg.norm(text_vector) * np.linalg.norm(vector)
                )
                similarities.append(float(similarity))
            else:
                similarities.append(0.0)
        
        return similarities
    
    def match_symptoms(self, text: str) -> List[Dict[str, Any]]:
        """Match symptoms in text.
        
        Args:
            text: Input text
            
        Returns:
            List of matched symptoms with scores and spans
        """
        # Initialize results
        matches = []
        used_spans = set()
        
        # First try exact matches
        for symptom, pattern_info in self.patterns.items():
            exact_pattern = pattern_info['exact']
            for match in exact_pattern.finditer(text):
                start, end = match.span()
                if not any(s <= start < e or s < end <= e for s, e in used_spans):
                    matches.append({
                        'text': symptom,
                        'score': 1.0,
                        'start': start,
                        'end': end,
                        'info': pattern_info['info']
                    })
                    used_spans.add((start, end))
        
        # Then try fuzzy matches on unused spans
        unused_spans = self._find_unused_spans(text, used_spans)
        for start, end in unused_spans:
            span_text = text[start:end]
            if not span_text.strip():
                continue
                
            # Try fuzzy matching against all symptoms
            best_score = 0.0
            best_match = None
            
            for symptom, pattern_info in self.patterns.items():
                score = self._calculate_fuzzy_score(span_text, symptom)
                if score > best_score and score >= self.config.get('fuzzy_threshold', 0.8):
                    best_score = score
                    best_match = (symptom, pattern_info)
            
            if best_match:
                symptom, pattern_info = best_match
                matches.append({
                    'text': symptom,
                    'score': best_score,
                    'start': start,
                    'end': end,
                    'info': pattern_info['info']
                })
                used_spans.add((start, end))
        
        return matches
    
    @property
    def word2vec_model(self) -> Optional[Word2Vec]:
        """Get the Word2Vec model.
        
        Returns:
            Word2Vec model or None if not enabled
        """
        return self._word2vec_model
    
    def clear_cache(self) -> None:
        """Clear the symptom matching cache."""
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
        self._word2vec_model = None

    def _load_lexicon(self) -> Dict[str, Dict[str, Any]]:
        """Load symptom lexicon from YAML file.
        
        Returns:
            Dictionary mapping symptoms to their information
        """
        lexicon_path = self.config.get('lexicon_path', 'data/lexicons/symptom_lexicon.yaml')
        
        try:
            with open(lexicon_path, 'r') as f:
                lexicon = yaml.safe_load(f)
            
            # Validate lexicon structure
            for symptom, info in lexicon.items():
                if not isinstance(info, dict):
                    raise ValueError(f"Invalid symptom info format for {symptom}")
                if 'dimension' not in info:
                    raise ValueError(f"Missing dimension for {symptom}")
                if 'polarity' not in info:
                    raise ValueError(f"Missing polarity for {symptom}")
            
            logger.info(f"Loaded {len(lexicon)} symptoms from lexicon")
            return lexicon
            
        except Exception as e:
            logger.error(f"Error loading lexicon: {e}")
            return {}
    
    def _compile_patterns(self) -> None:
        """Compile patterns for symptom matching."""
        self.patterns = {}
        
        for symptom, info in self.lexicon.items():
            # Create pattern for exact match only
            exact_pattern = re.compile(r'\b' + re.escape(symptom) + r'\b', re.IGNORECASE)
            
            self.patterns[symptom] = {
                'exact': exact_pattern,
                'info': info
            }
    
    def _find_unused_spans(self, text: str, used_spans: Set[Tuple[int, int]]) -> List[Tuple[int, int]]:
        """Find spans of text that haven't been used in matches.
        
        Args:
            text: Input text
            used_spans: Set of (start, end) tuples for used spans
            
        Returns:
            List of (start, end) tuples for unused spans
        """
        # Sort used spans by start position
        sorted_spans = sorted(used_spans)
        
        # Find gaps between used spans
        unused_spans = []
        last_end = 0
        
        for start, end in sorted_spans:
            if start > last_end:
                unused_spans.append((last_end, start))
            last_end = max(last_end, end)
        
        # Add final span if needed
        if last_end < len(text):
            unused_spans.append((last_end, len(text)))
        
        return unused_spans
    
    def _calculate_fuzzy_score(self, text: str, symptom: str) -> float:
        """Calculate fuzzy matching score between text and symptom using rapidfuzz.
        
        Args:
            text: Input text
            symptom: Symptom to match against
            
        Returns:
            Score between 0 and 1
        """
        # Check cache
        cache_key = f"{text}|{symptom}"
        with self.cache_lock:
            if cache_key in self.cache:
                return self.cache[cache_key]
        
        try:
            from rapidfuzz import process, fuzz
            
            # Use token_sort_ratio for better handling of word order
            score = fuzz.token_sort_ratio(text.lower(), symptom.lower()) / 100.0
            
            # Cache result
            with self.cache_lock:
                self.cache[cache_key] = score
            
            return score
        except ImportError:
            # Fallback to word overlap if rapidfuzz not available
            text_words = set(text.lower().split())
            symptom_words = set(symptom.lower().split())
            
            # Calculate word overlap
            common_words = text_words & symptom_words
            total_words = text_words | symptom_words
            
            score = len(common_words) / len(total_words) if total_words else 0.0
            
            # Cache result
            with self.cache_lock:
                self.cache[cache_key] = score
            
            return score
    
    def clear_caches(self) -> None:
        """Clear all caches."""
        self.fuzzy_cache.clear()
    
    def _warmup_cache(self) -> None:
        """Warm up cache with common symptoms."""
        common_symptoms = [
            ("anxiety", {"dimension": "anxiety", "weight": 0.9}),
            ("depression", {"dimension": "depression", "weight": 0.9}),
            ("insomnia", {"dimension": "sleep", "weight": 0.8}),
            ("fatigue", {"dimension": "energy", "weight": 0.8}),
            ("mood swings", {"dimension": "mood", "weight": 0.9}),
            ("concentration", {"dimension": "cognition", "weight": 0.8})
        ]
        self.fuzzy_cache.warmup(common_symptoms) 
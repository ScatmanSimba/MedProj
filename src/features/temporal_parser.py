"""Temporal parsing module for psychiatric medication response prediction.

This module provides functionality for parsing temporal information in text,
including medication status and duration.
"""

from typing import List, Dict, Any, Optional, Set, Tuple, Union
import spacy
from spacy.tokens import Doc, Token, Span
from spacy.matcher import Matcher
import logging
from collections import OrderedDict
from functools import lru_cache
import hashlib
import re
from datetime import datetime, timedelta
import threading
from lru import LRUCache

logger = logging.getLogger(__name__)

class TemporalStatusEngine:
    """Engine for determining temporal status of medications."""
    
    def __init__(self, config: Dict[str, Any]):
        """Initialize the temporal status engine.
        
        Args:
            config: Configuration dictionary
        """
        self.config = config
        self.logger = logging.getLogger(__name__)
        
        # Initialize spaCy
        try:
            self.nlp = spacy.load("en_core_web_sm")
            self.logger.info("Successfully loaded spaCy model")
        except Exception as e:
            self.logger.error(f"Failed to load spaCy model: {e}")
            raise
        
        # Initialize temporal patterns
        self.temporal_patterns = self._load_temporal_patterns()
        
        # Initialize temporal weights
        self.temporal_weights = {
            'current': 1.0,
            'past': 0.8,
            'prospective': 0.6,
            'unknown': 0.3
        }
        
        # Initialize temporal markers
        self.temporal_markers = {
            'current': ['now', 'currently', 'taking', 'on', 'using'],
            'past': ['stopped', 'quit', 'discontinued', 'used to', 'tried'],
            'prospective': ['plan to', 'going to', 'will start', 'about to']
        }
    
    def get_temporal_status(self, doc: Doc) -> Dict[str, Any]:
        """Get temporal status of medications in text.
        
        Args:
            doc: spaCy Doc object
            
        Returns:
            Dictionary with temporal status information for each medication:
            {
                'medication': {
                    'status': str,  # 'current', 'past', 'prospective', or 'unknown'
                    'confidence': float,  # Confidence score [0,1]
                    'evidence': str,  # Text evidence for the status
                    'position': tuple,  # (start, end) position in text
                    'duration': str,  # Duration string if available
                    'start_date': datetime,  # Start date if available
                    'end_date': datetime,  # End date if available
                    'transition': str,  # Transition type if available
                    'secondary_status': str,  # Secondary status if available
                    'mentions': list,  # List of all temporal mentions
                    'most_recent': datetime,  # Most recent mention date
                    'recency_days': int,  # Days since most recent mention
                    'recency_confidence': float,  # Confidence in recency [0,1]
                    'conflicting_mentions': list  # List of conflicting mentions
                }
            }
        """
        # Find medication entities
        medications = []
        for ent in doc.ents:
            if ent.label_ == "MEDICATION":
                medications.append(ent)
        
        if not medications:
            return {}
        
        # Initialize status dictionary
        status = {}
        
        for med in medications:
            # Default status
            med_status = {
                'status': 'unknown',
                'confidence': 0.3,
                'evidence': None,
                'position': (med.start, med.end),
                'duration': None,
                'start_date': None,
                'end_date': None,
                'transition': None,
                'secondary_status': None,
                'mentions': [],
                'most_recent': None,
                'recency_days': None,
                'recency_confidence': 0.0,
                'conflicting_mentions': []
            }
            
            # Check for temporal patterns
            for pattern in self.temporal_patterns:
                matches = pattern['matcher'](doc)
                for match_id, start, end in matches:
                    if start <= med.start and end >= med.end:
                        # Update status based on pattern
                        pattern_type = pattern['type']
                        confidence = pattern['weight']
                        
                        if pattern_type in self.temporal_weights:
                            # Check for conflicts
                            if med_status['status'] != 'unknown' and med_status['status'] != pattern_type:
                                med_status['conflicting_mentions'].append({
                                    'text': doc[start:end].text,
                                    'type': pattern_type,
                                    'confidence': confidence,
                                    'position': (start, end)
                                })
                                # Lower confidence when there are conflicts
                                confidence *= 0.8
                            
                            med_status['status'] = pattern_type
                            med_status['confidence'] = max(med_status['confidence'], confidence)
                            med_status['evidence'] = doc[start:end].text
                            
                            # Track mention
                            med_status['mentions'].append({
                                'text': doc[start:end].text,
                                'type': pattern_type,
                                'confidence': confidence,
                                'position': (start, end)
                            })
            
            # Calculate recency confidence
            if med_status['duration']:
                duration_info = self._parse_duration_to_days(med_status['duration'])
                if duration_info:
                    med_status['recency_days'] = duration_info['duration_days']
                    # Calculate recency confidence based on temporal recency weights
                    if duration_info['duration_days'] <= 7:  # Within a week
                        med_status['recency_confidence'] = self.temporal_weights['current']
                    elif duration_info['duration_days'] <= 30:  # Within a month
                        med_status['recency_confidence'] = self.temporal_weights['past'] * 0.9
                    elif duration_info['duration_days'] <= 180:  # Within 6 months
                        med_status['recency_confidence'] = self.temporal_weights['past'] * 0.7
                    else:
                        med_status['recency_confidence'] = self.temporal_weights['past'] * 0.5
            
            # Update status dictionary
            status[med.text] = med_status
        
        return status
    
    def _load_temporal_patterns(self) -> List[Dict[str, Any]]:
        """Load temporal patterns from configuration.
        
        Returns:
            List of temporal pattern dictionaries
        """
        patterns = []
        
        # Add current medication patterns
        current_patterns = [
            "taking MEDICATION",
            "on MEDICATION",
            "currently taking MEDICATION",
            "using MEDICATION",
            "MEDICATION is working",
            "MEDICATION helps"
        ]
        
        for pattern in current_patterns:
            patterns.append({
                'type': 'current',
                'pattern': pattern,
                'weight': 0.9,
                'matcher': self._create_matcher(pattern)
            })
        
        # Add past medication patterns
        past_patterns = [
            "stopped MEDICATION",
            "quit MEDICATION",
            "discontinued MEDICATION",
            "used to take MEDICATION",
            "tried MEDICATION",
            "MEDICATION didn't work"
        ]
        
        for pattern in past_patterns:
            patterns.append({
                'type': 'past',
                'pattern': pattern,
                'weight': 0.8,
                'matcher': self._create_matcher(pattern)
            })
        
        # Add prospective medication patterns
        prospective_patterns = [
            "plan to take MEDICATION",
            "going to start MEDICATION",
            "will start MEDICATION",
            "about to start MEDICATION",
            "considering MEDICATION"
        ]
        
        for pattern in prospective_patterns:
            patterns.append({
                'type': 'prospective',
                'pattern': pattern,
                'weight': 0.7,
                'matcher': self._create_matcher(pattern)
            })
        
        return patterns
    
    def _create_matcher(self, pattern: str) -> Matcher:
        """Create a spaCy matcher for a pattern.
        
        Args:
            pattern: Pattern string
            
        Returns:
            spaCy Matcher object
        """
        matcher = Matcher(self.nlp.vocab)
        
        # Convert pattern to spaCy pattern
        pattern_parts = pattern.split()
        matcher_pattern = []
        
        for part in pattern_parts:
            if part == "MEDICATION":
                matcher_pattern.append({"ENT_TYPE": "MEDICATION"})
            else:
                matcher_pattern.append({"LOWER": part.lower()})
        
        matcher.add(pattern, [matcher_pattern])
        return matcher

class TemporalParser:
    """Parser for temporal information in text."""
    
    def __init__(self, config: Dict[str, Any], debug: bool = False):
        """Initialize the temporal parser.
        
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
            maxsize=cache_config.get('temporal_cache_size', 5000),
            compression_threshold=cache_config.get('compression_threshold', 1024),
            warmup_size=cache_config.get('warmup_size', 100),
            eviction_policy=cache_config.get('eviction_policy', 'lru_freq')
        )
        
        # Initialize locks
        self.cache_lock = threading.RLock()
        
        # Initialize duration mappings
        self.duration_mappings = {
            'several': 3,
            'few': 3,
            'couple of': 2,
            'handful of': 5,
            'short': 1,
            'long': 30,
            'brief': 1,
            'extended': 30
        }
        
        # Load patterns
        self.patterns = self._load_patterns()
        
        # Initialize temporal status engine
        self.status_engine = TemporalStatusEngine(config)
    
    def _load_patterns(self) -> Dict[str, List[re.Pattern]]:
        """Load temporal patterns from configuration.
        
        Returns:
            Dictionary mapping pattern types to compiled regex patterns
        """
        patterns = {
            'duration': [],
            'date': [],
            'relative': []
        }
        
        # Duration patterns
        duration_patterns = [
            r'\b(\d+)\s*(day|days|week|weeks|month|months|year|years)\b',
            r'\b(several|few|couple of)\s*(day|days|week|weeks|month|months|year|years)\b',
            r'\b(short|long)\s*(while|time)\b'
        ]
        
        # Date patterns
        date_patterns = [
            r'\b(january|february|march|april|may|june|july|august|september|october|november|december)\s+(\d{1,2})(?:st|nd|rd|th)?\s*,?\s*(\d{4})?\b',
            r'\b(\d{1,2})[/-](\d{1,2})[/-](\d{2,4})\b',
            r'\b(today|yesterday|tomorrow)\b'
        ]
        
        # Relative time patterns
        relative_patterns = [
            r'\b(\d+)\s*(day|days|week|weeks|month|months|year|years)\s*(ago|from now)\b',
            r'\b(last|next)\s*(week|month|year)\b',
            r'\b(recently|lately|nowadays)\b'
        ]
        
        # Compile patterns
        for pattern in duration_patterns:
            patterns['duration'].append(re.compile(pattern, re.IGNORECASE))
        
        for pattern in date_patterns:
            patterns['date'].append(re.compile(pattern, re.IGNORECASE))
        
        for pattern in relative_patterns:
            patterns['relative'].append(re.compile(pattern, re.IGNORECASE))
        
        return patterns
    
    def parse_temporal_info(self, text_or_doc: Union[str, Doc]) -> Dict[str, Any]:
        """Parse temporal information from text.
        
        Args:
            text_or_doc: Input text or spaCy Doc object
            
        Returns:
            Dictionary with parsed temporal information:
            {
                'medications': {
                    'medication_name': {
                        'status': str,  # 'current', 'past', 'prospective', or 'unknown'
                        'confidence': float,  # Confidence score [0,1]
                        'evidence': str,  # Text evidence for the status
                        'position': tuple,  # (start, end) position in text
                        'duration': str,  # Duration string if available
                        'start_date': datetime,  # Start date if available
                        'end_date': datetime,  # End date if available
                        'transition': str,  # Transition type if available
                        'secondary_status': str,  # Secondary status if available
                        'mentions': list,  # List of all temporal mentions
                        'most_recent': datetime,  # Most recent mention date
                        'recency_days': int,  # Days since most recent mention
                        'recency_confidence': float,  # Confidence in recency [0,1]
                        'conflicting_mentions': list  # List of conflicting mentions
                    }
                },
                'durations': list,  # List of duration mentions
                'dates': list,  # List of date mentions
                'relative_times': list  # List of relative time mentions
            }
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
        
        # Get temporal status from engine
        temporal_status = self.status_engine.get_temporal_status(doc)
        
        # Parse additional temporal information
        temporal_info = {
            'medications': temporal_status,
            'durations': self._parse_durations(doc),
            'dates': self._parse_dates(doc),
            'relative_times': self._parse_relative_times(doc)
        }
        
        # Cache results
        with self.cache_lock:
            self.cache[cache_key] = temporal_info
        
        return temporal_info
    
    def _parse_durations(self, doc: Doc) -> List[Dict[str, Any]]:
        """Parse duration information from text.
        
        Args:
            doc: spaCy Doc object
            
        Returns:
            List of parsed durations
        """
        durations = []
        for pattern in self.patterns['duration']:
            matches = pattern.finditer(doc.text)
            for match in matches:
                duration = self._parse_duration_to_days(match.group())
                if duration is not None:
                    durations.append({
                        'text': match.group(),
                        'days': duration,
                        'start': match.start(),
                        'end': match.end()
                    })
        return durations
    
    def _parse_dates(self, doc: Doc) -> List[Dict[str, Any]]:
        """Parse date information from text.
        
        Args:
            doc: spaCy Doc object
            
        Returns:
            List of parsed dates
        """
        dates = []
        for pattern in self.patterns['date']:
            matches = pattern.finditer(doc.text)
            for match in matches:
                date = self._parse_date_to_timestamp(match.group())
                if date is not None:
                    dates.append({
                        'text': match.group(),
                        'timestamp': date,
                        'start': match.start(),
                        'end': match.end()
                    })
        return dates
    
    def _parse_relative_times(self, doc: Doc) -> List[Dict[str, Any]]:
        """Parse relative time information from text.
        
        Args:
            doc: spaCy Doc object
            
        Returns:
            List of parsed relative times
        """
        relative_times = []
        for pattern in self.patterns['relative']:
            matches = pattern.finditer(doc.text)
            for match in matches:
                relative_time = self._parse_relative_time(match.group())
                if relative_time is not None:
                    relative_times.append({
                        'text': match.group(),
                        'offset': relative_time,
                        'start': match.start(),
                        'end': match.end()
                    })
        return relative_times
    
    def clear_cache(self) -> None:
        """Clear the temporal parsing cache."""
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

    def _add_temporal_patterns(self) -> None:
        """Add temporal patterns to the matcher using config-defined patterns."""
        # Clear existing patterns
        self.temporal_matcher = Matcher(self.nlp.vocab)
        
        # Store pattern weights for later lookup
        self.pattern_weights = {}
        
        # Add each pattern as a separate rule
        for pattern in self.config.get('temporal_patterns', []):
            pattern_text = pattern['pattern']
            weight = pattern.get('weight', 0.5)
            
            # Convert pattern text to spaCy pattern
            pattern_parts = pattern_text.split()
            matcher_pattern = []
            
            for part in pattern_parts:
                if part == "MEDICATION":
                    matcher_pattern.append({"ENT_TYPE": "MEDICATION"})
                else:
                    matcher_pattern.append({"LOWER": part.lower()})
            
            # Add pattern with unique ID
            pattern_id = f"TEMPORAL_{pattern_text.replace(' ', '_')}"
            self.temporal_matcher.add(pattern_id, [matcher_pattern])
            
            # Store weight for this pattern
            self.pattern_weights[pattern_id] = weight
            
            logger.debug(f"Added temporal pattern: {pattern_text} with weight {weight}")
    
    def _get_cache_key(self, doc: Doc) -> str:
        """Generate a cache key for document-level caching.
        
        Args:
            doc: spaCy Doc
            
        Returns:
            Cache key string
        """
        # Include document text for precise matching
        key_parts = [
            str(doc.text)
        ]
        return hashlib.md5('|'.join(key_parts).encode()).hexdigest()
    
    def _parse_duration(self, doc: Doc, start_idx: int, end_idx: int) -> Optional[Dict[str, Any]]:
        """Parse duration from text span.
        
        Args:
            doc: spaCy Doc
            start_idx: Start index of span
            end_idx: End index of span
            
        Returns:
            Dictionary with duration information or None if not found
        """
        # Get tokens in span
        tokens = doc[start_idx:end_idx]
        
        # Initialize duration
        amount = None
        unit = None
        unit_str = None
        
        # Find amount
        for token in tokens:
            # Check for numeric values
            if token.like_num:
                amount = float(token.text)
            # Check for duration mappings
            elif token.text.lower() in self.duration_mappings:
                amount = self.duration_mappings[token.text.lower()]
        
        # Find unit
        for token in tokens:
            if token.text.lower() in ['day', 'days']:
                unit = 1.0
                unit_str = 'days'
            elif token.text.lower() in ['week', 'weeks']:
                unit = 7.0
                unit_str = 'weeks'
            elif token.text.lower() in ['month', 'months']:
                unit = 30.0
                unit_str = 'months'
            elif token.text.lower() in ['year', 'years']:
                unit = 365.0
                unit_str = 'years'
        
        # Return duration if both amount and unit found
        if amount is not None and unit is not None:
            return {
                'duration': f"{amount} {unit_str}",
                'duration_days': amount * unit
            }
        return None
    
    def _get_dependency_path(self, doc: Doc, token1: Token, token2: Token) -> List[Token]:
        """Get dependency path between two tokens.
        
        Args:
            doc: spaCy Doc
            token1: First token
            token2: Second token
            
        Returns:
            List of tokens in dependency path
        """
        # Get paths to root for both tokens
        path1 = []
        path2 = []
        
        # Get path for token1
        current = token1
        while current.head != current:
            path1.append(current.head)
            current = current.head
        
        # Get path for token2
        current = token2
        while current.head != current:
            path2.append(current.head)
            current = current.head
        
        # Find common ancestor
        common_ancestor = None
        for t1 in path1:
            if t1 in path2:
                common_ancestor = t1
                break
        
        if common_ancestor is None:
            return []
        
        # Build path from token1 to common ancestor
        path = []
        current = token1
        while current != common_ancestor:
            path.append(current)
            current = current.head
        
        # Add common ancestor
        path.append(common_ancestor)
        
        # Add path from common ancestor to token2
        current = token2
        while current != common_ancestor:
            path.append(current)
            current = current.head
        
        return path
    
    def _is_related(self, doc: Doc, token1: Token, token2: Token) -> bool:
        """Check if two tokens are related in dependency tree.
        
        Args:
            doc: spaCy Doc
            token1: First token
            token2: Second token
            
        Returns:
            True if tokens are related
        """
        # Get dependency path
        path = self._get_dependency_path(doc, token1, token2)
        
        # Check if path exists and is not too long
        max_path_length = self.config.get('dependency', {}).get('max_path_length', 5)
        return len(path) > 0 and len(path) <= max_path_length
    
    def _detect_transition(self, doc: Doc, sent: Span) -> Optional[Dict[str, Any]]:
        """Detect medication transition in sentence.
        
        Args:
            doc: spaCy Doc
            sent: Sentence span
            
        Returns:
            Dictionary with transition information or None
        """
        # Find transition matches
        matches = self.temporal_matcher(sent)
        
        for match_id, start, end in matches:
            # Get transition type
            transition_type = self.nlp.vocab.strings[match_id]
            
            # Get medications involved
            med_ents = [ent for ent in sent.ents if ent.label_ == "MEDICATION"]
            
            if not med_ents:
                continue
            
            # Initialize transition info
            transition = {
                'type': transition_type,
                'medications': [],
                'direction': None
            }
            
            # Process based on transition type
            if transition_type == 'switch':
                # Find from and to medications
                from_med = None
                to_med = None
                
                # Look for "from" and "to" tokens
                for token in sent:
                    if token.text.lower() == 'from' and token.i + 1 < len(sent):
                        from_med = next((ent for ent in med_ents if ent.start == token.i + 1), None)
                    elif token.text.lower() == 'to' and token.i + 1 < len(sent):
                        to_med = next((ent for ent in med_ents if ent.start == token.i + 1), None)
                
                if from_med and to_med:
                    transition['medications'] = [from_med.text, to_med.text]
                    transition['direction'] = 'from_to'
                elif len(med_ents) >= 2:
                    # If no explicit from/to, use first two medications
                    transition['medications'] = [med_ents[0].text, med_ents[1].text]
                    transition['direction'] = 'from_to'
            
            elif transition_type in ['add', 'remove']:
                # Find target medication
                target_med = next((ent for ent in med_ents if ent.start > start), None)
                if target_med:
                    transition['medications'] = [target_med.text]
                    transition['direction'] = 'to' if transition_type == 'add' else 'from'
            
            if transition['medications']:
                return transition
        
        return None
    
    def parse_temporal(self, doc: spacy.tokens.Doc) -> List[Dict[str, Any]]:
        """Parse temporal status of medications in text.
        
        Args:
            doc: spaCy Doc
            
        Returns:
            List of temporal status dictionaries
        """
        # Check cache
        cache_key = self._get_cache_key(doc)
        if cache_key in self.temporal_cache:
            return self.temporal_cache[cache_key]
        
        # Find medication entities
        medications = []
        for ent in doc.ents:
            if ent.label_ == "MEDICATION":
                medications.append(ent)
        
        if not medications:
            return []
        
        # Find temporal patterns
        matches = self.temporal_matcher(doc)
        temporal_status = []
        
        for med in medications:
            # Default status
            status = {
                'medication': med.text,
                'status': 'current',
                'confidence': 0.5,
                'pattern_matches': []  # Track which patterns matched
            }
            
            # Check for temporal patterns
            for match_id, start, end in matches:
                if start <= med.start and end >= med.end:
                    pattern_text = doc[start:end].text
                    pattern_id = self.nlp.vocab.strings[match_id]
                    
                    # Get pattern weight
                    weight = self.pattern_weights.get(pattern_id, 0.5)
                    
                    # Update status based on pattern
                    if "started" in pattern_text or "taking" in pattern_text:
                        status['status'] = 'current'
                        status['confidence'] = max(status['confidence'], weight)
                    elif "stopped" in pattern_text or "quit" in pattern_text:
                        status['status'] = 'past'
                        status['confidence'] = max(status['confidence'], weight)
                    elif "tried" in pattern_text or "used" in pattern_text:
                        status['status'] = 'past'
                        status['confidence'] = max(status['confidence'], weight * 0.8)
                    
                    # Track pattern match
                    status['pattern_matches'].append({
                        'pattern': pattern_text,
                        'weight': weight,
                        'span': (start, end)
                    })
            
            temporal_status.append(status)
        
        # Update cache
        self._update_cache(cache_key, temporal_status)
        
        return temporal_status
    
    def _update_cache(self, cache_key: str, status: Dict[str, Any]) -> None:
        """Update cache with new status information.
        
        Args:
            cache_key: Cache key
            status: Dictionary with temporal status information
        """
        if len(self.temporal_cache) >= self.max_cache_size:
            self.temporal_cache.popitem(last=False)
        self.temporal_cache[cache_key] = status
    
    def clear_caches(self) -> None:
        """Clear all caches."""
        self.temporal_cache.clear()
    
    def get_cache_stats(self) -> Dict[str, Any]:
        """Get cache statistics.
        
        Returns:
            Dictionary with cache statistics
        """
        return {
            'temporal_cache': {
                'size': len(self.temporal_cache),
                'max_size': self.max_cache_size
            }
        }

    def _process_temporal_status(self, temporal_info: Dict[str, Any]) -> Dict[str, Any]:
        """Process temporal status information into a standardized format.
        
        Args:
            temporal_info: Dictionary with temporal information from parse_temporal_info
            
        Returns:
            Dictionary with processed temporal status:
            {
                'medications': {
                    'medication_name': {
                        'status': str,  # 'current', 'past', 'prospective', or 'unknown'
                        'confidence': float,  # Confidence score [0,1]
                        'evidence': str,  # Text evidence for the status
                        'position': tuple,  # (start, end) position in text
                        'duration': str,  # Duration string if available
                        'start_date': datetime,  # Start date if available
                        'end_date': datetime,  # End date if available
                        'transition': str,  # Transition type if available
                        'secondary_status': str,  # Secondary status if available
                        'mentions': list,  # List of all temporal mentions
                        'most_recent': datetime,  # Most recent mention date
                        'recency_days': int,  # Days since most recent mention
                        'recency_confidence': float,  # Confidence in recency [0,1]
                        'conflicting_mentions': list  # List of conflicting mentions
                    }
                }
            }
        """
        # The temporal_info is already in the correct format from parse_temporal_info
        return {
            'medications': temporal_info['medications']
        }

    def _parse_duration_to_days(self, duration_str: str) -> Optional[float]:
        """Convert a duration string to days.
        
        Args:
            duration_str: Duration string (e.g., '2 weeks', 'few days')
        Returns:
            Number of days as float, or None if parsing fails
        """
        import re
        duration_str = duration_str.lower().strip()
        # Try to match numeric duration
        match = re.match(r"(\d+(?:\.\d+)?)\s*(day|days|week|weeks|month|months|year|years)", duration_str)
        if match:
            amount = float(match.group(1))
            unit = match.group(2)
            if 'day' in unit:
                return amount
            elif 'week' in unit:
                return amount * 7
            elif 'month' in unit:
                return amount * 30
            elif 'year' in unit:
                return amount * 365
        # Try to match mapped duration
        for key, val in self.duration_mappings.items():
            if key in duration_str:
                # Find unit
                for unit in ['day', 'week', 'month', 'year']:
                    if unit in duration_str:
                        if unit == 'day':
                            return val
                        elif unit == 'week':
                            return val * 7
                        elif unit == 'month':
                            return val * 30
                        elif unit == 'year':
                            return val * 365
        return None

    def _parse_date_to_timestamp(self, date_str: str) -> Optional[float]:
        """Convert a date string to a Unix timestamp.
        
        Args:
            date_str: Date string in various formats
        Returns:
            Unix timestamp as float, or None if parsing fails
        """
        from datetime import datetime
        import re
        # Try common date formats
        formats = [
            "%Y-%m-%d",
            "%m/%d/%Y",
            "%d/%m/%Y",
            "%B %d, %Y",
            "%b %d, %Y",
        ]
        for fmt in formats:
            try:
                dt = datetime.strptime(date_str.strip(), fmt)
                return dt.timestamp()
            except ValueError:
                continue
        # Try relative dates (e.g., '3 days ago')
        rel_patterns = {
            r"(\d+)\s+days?\s+ago": lambda x: datetime.now() - timedelta(days=int(x)),
            r"(\d+)\s+weeks?\s+ago": lambda x: datetime.now() - timedelta(weeks=int(x)),
            r"(\d+)\s+months?\s+ago": lambda x: datetime.now() - timedelta(days=int(x)*30),
            r"(\d+)\s+years?\s+ago": lambda x: datetime.now() - timedelta(days=int(x)*365),
        }
        for pattern, func in rel_patterns.items():
            match = re.match(pattern, date_str.lower())
            if match:
                try:
                    dt = func(match.group(1))
                    return dt.timestamp()
                except Exception:
                    continue
        return None 
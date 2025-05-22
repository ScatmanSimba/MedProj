"""Response attribution module for psychiatric medication response prediction.

This module provides functionality for attributing response dimensions to specific
medications based on text analysis.
"""

from typing import List, Dict, Any, Optional, Tuple, Set
import spacy
from spacy.tokens import Doc, Token, Span
from spacy.matcher import Matcher
from spacy.pipeline import EntityRuler
import numpy as np
import logging
from pathlib import Path
import pandas as pd
import io
import hashlib
from collections import OrderedDict
import yaml
from tqdm import tqdm
import threading
from sklearn.isotonic import IsotonicRegression
from dataclasses import dataclass
from datetime import datetime

from src.features.symptom_matcher import SymptomMatcher
from src.features.temporal_parser import TemporalParser
from src.features.emoji_processor import EmojiProcessor
from src.med_dictionary1803 import MedDictionary
from ..config.config_loader import load_feature_config
from ..utils.cache_utils import EnhancedLRUCache

logger = logging.getLogger(__name__)

# Constants for signal filtering
MIN_SIGNAL_COUNT = 1
MIN_SIGNAL_STRENGTH = 0.15

@dataclass
class TokenMatch:
    """Information about a matched token."""
    text: str
    start: int
    end: int
    pos: str
    dep: str
    lemma: str
    proximity_weight: float
    causality_weight: float
    intensity_weight: float
    confidence: float
    ignored: bool = False
    ignore_reason: Optional[str] = None

@dataclass
class SymptomMatch:
    """Information about a matched symptom."""
    text: str
    canonical: str
    dimension: str
    polarity: float
    start: int
    end: int
    tokens: List[TokenMatch]
    ignored: bool = False
    ignore_reason: Optional[str] = None

class TokenAlignmentDebugger:
    """Debugger for token-level alignment in response attribution."""
    
    def __init__(self, debug_dir: Optional[str] = None):
        """Initialize the debugger.
        
        Args:
            debug_dir: Optional directory to store debug output
        """
        self.debug_dir = debug_dir
        self.matches: Dict[str, List[SymptomMatch]] = {}  # med_name -> matches
        self.ignored_matches: Dict[str, List[SymptomMatch]] = {}  # med_name -> ignored matches
        self.token_weights: Dict[str, Dict[str, float]] = {}  # token_id -> weights
        
        if debug_dir:
            Path(debug_dir).mkdir(parents=True, exist_ok=True)
    
    def add_match(self, med_name: str, symptom: Dict[str, Any], 
                 doc: Doc, ignored: bool = False, ignore_reason: Optional[str] = None) -> None:
        """Add a symptom match with token-level information.
        
        Args:
            med_name: Name of the medication
            symptom: Matched symptom information
            doc: spaCy Doc
            ignored: Whether this match was ignored
            ignore_reason: Reason for ignoring if ignored
        """
        # Get token matches
        tokens = []
        for i in range(symptom['position'][0], symptom['position'][1]):
            token = doc[i]
            # Calculate weights
            proximity_weight = self._calculate_proximity_weight(doc, i, doc[symptom['position'][0]:symptom['position'][1]])
            causality_weight = self._calculate_causal_confidence(doc, doc[symptom['position'][0]:symptom['position'][1]], token)
            intensity_weight = self._detect_intensity(doc, i, i+1)
            
            token_match = TokenMatch(
                text=token.text,
                start=token.idx,
                end=token.idx + len(token.text),
                pos=token.pos_,
                dep=token.dep_,
                lemma=token.lemma_,
                proximity_weight=proximity_weight,
                causality_weight=causality_weight,
                intensity_weight=intensity_weight,
                confidence=min(proximity_weight, causality_weight, intensity_weight)
            )
            tokens.append(token_match)
        
        # Create symptom match
        symptom_match = SymptomMatch(
            text=symptom['text'],
            canonical=symptom['canonical'],
            dimension=symptom['dimension'],
            polarity=symptom['polarity'],
            start=symptom['position'][0],
            end=symptom['position'][1],
            tokens=tokens,
            ignored=ignored,
            ignore_reason=ignore_reason
        )
        
        # Add to appropriate collection
        if ignored:
            if med_name not in self.ignored_matches:
                self.ignored_matches[med_name] = []
            self.ignored_matches[med_name].append(symptom_match)
        else:
            if med_name not in self.matches:
                self.matches[med_name] = []
            self.matches[med_name].append(symptom_match)
    
    def _calculate_proximity_weight(self, doc: Doc, token_idx: int, med_span: Span) -> float:
        """Calculate proximity weight between token and medication."""
        distance = min(abs(token_idx - med_span.start), abs(token_idx - med_span.end))
        decay_factor = 0.8  # Configurable
        max_distance = 10   # Configurable
        return max(0.0, min(1.0, decay_factor ** distance)) if distance <= max_distance else 0.0
    
    def _calculate_causal_confidence(self, doc: Doc, symptom_span: Span, token: Token) -> float:
        """Calculate causal confidence for a token."""
        # Check for causal patterns
        causal_patterns = {
            'causes': 0.9,
            'results in': 0.9,
            'leads to': 0.8,
            'triggers': 0.8,
            'makes': 0.7,
            'affects': 0.6
        }
        
        # Check token and its context
        confidence = 0.0
        for pattern, weight in causal_patterns.items():
            if pattern in token.text.lower():
                confidence = max(confidence, weight)
            # Check bigrams
            if token.i + 1 < len(doc):
                bigram = f"{token.text.lower()} {doc[token.i + 1].text.lower()}"
                if pattern in bigram:
                    confidence = max(confidence, weight)
        
        return confidence
    
    def _detect_intensity(self, doc: Doc, start_idx: int, end_idx: int) -> float:
        """Detect intensity modifiers around a span."""
        intensity = 1.0
        window_size = 3
        
        # Check tokens in window
        start = max(0, start_idx - window_size)
        end = min(len(doc), end_idx + window_size)
        
        for token in doc[start:end]:
            if token.text.lower() in ['very', 'extremely', 'incredibly', 'really', 'so']:
                intensity *= 1.5
            elif token.text.lower() in ['slightly', 'somewhat', 'a bit', 'kind of']:
                intensity *= 0.7
            elif token.text.lower() in ['not', 'no', 'never', 'none']:
                intensity *= -1.0
        
        return intensity
    
    def save_debug_output(self, filename: Optional[str] = None) -> None:
        """Save debug output to file.
        
        Args:
            filename: Optional filename, defaults to timestamp
        """
        if not self.debug_dir:
            return
        
        if filename is None:
            filename = f"token_alignment_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
        
        output = {
            'matches': {
                med: [
                    {
                        'text': match.text,
                        'canonical': match.canonical,
                        'dimension': match.dimension,
                        'polarity': match.polarity,
                        'start': match.start,
                        'end': match.end,
                        'tokens': [
                            {
                                'text': token.text,
                                'pos': token.pos,
                                'dep': token.dep,
                                'lemma': token.lemma,
                                'proximity_weight': token.proximity_weight,
                                'causality_weight': token.causality_weight,
                                'intensity_weight': token.intensity_weight,
                                'confidence': token.confidence
                            }
                            for token in match.tokens
                        ]
                    }
                    for match in matches
                ]
                for med, matches in self.matches.items()
            },
            'ignored_matches': {
                med: [
                    {
                        'text': match.text,
                        'canonical': match.canonical,
                        'dimension': match.dimension,
                        'polarity': match.polarity,
                        'start': match.start,
                        'end': match.end,
                        'ignore_reason': match.ignore_reason,
                        'tokens': [
                            {
                                'text': token.text,
                                'pos': token.pos,
                                'dep': token.dep,
                                'lemma': token.lemma,
                                'proximity_weight': token.proximity_weight,
                                'causality_weight': token.causality_weight,
                                'intensity_weight': token.intensity_weight,
                                'confidence': token.confidence
                            }
                            for token in match.tokens
                        ]
                    }
                    for match in matches
                ]
                for med, matches in self.ignored_matches.items()
            }
        }
        
        output_path = Path(self.debug_dir) / filename
        with open(output_path, 'w') as f:
            json.dump(output, f, indent=2)
        
        logger.info(f"Saved token alignment debug output to {output_path}")

class ResponseAttributor:
    """Attributor for medication responses with confidence scoring."""
    
    def __init__(self, config: Dict[str, Any], debug: bool = False):
        """Initialize the ResponseAttributor.
        
        Args:
            config: Configuration dictionary
            debug: Whether to enable debug logging
        """
        self.config = config
        self.debug = debug
        
        # Initialize spaCy with coreferee
        self.nlp = spacy.load("en_core_web_sm")
        self.nlp.add_pipe("coreferee")
        
        # Initialize components
        self.symptom_matcher = SymptomMatcher(config, debug)
        self.temporal_parser = TemporalParser(config, debug)
        self.emoji_processor = EmojiProcessor(config, debug)
        
        # Initialize confidence calibration
        self.calibration_models = {
            'symptom': IsotonicRegression(out_of_bounds='clip'),
            'temporal': IsotonicRegression(out_of_bounds='clip'),
            'causal': IsotonicRegression(out_of_bounds='clip'),
            'emoji': IsotonicRegression(out_of_bounds='clip'),
            'overall': IsotonicRegression(out_of_bounds='clip')
        }
        
        # Initialize confidence weights
        self.confidence_weights = self.config.get('confidence_weights', {
            'symptom': 0.4,
            'temporal': 0.3,
            'causal': 0.2,
            'emoji': 0.1
        })
        
        # Initialize caches
        self._confidence_cache = {}
        self._temporal_cache = {}
        
        # Initialize calibration state
        self.is_calibrated = False
        self.calibration_metrics = {}
        
        # Initialize caches with enhanced LRU implementation
        cache_config = config.get('cache_settings', {})
        self.causality_cache = EnhancedLRUCache(
            maxsize=cache_config.get('causality_cache_size', 10000),
            compression_threshold=cache_config.get('compression_threshold', 1024),
            warmup_size=cache_config.get('warmup_size', 100),
            eviction_policy=cache_config.get('eviction_policy', 'lru_freq')
        )
        
        # Initialize locks
        self.cache_lock = threading.RLock()
        
        # Warm up caches
        self._warmup_cache()
        
        # Track document cache
        self._doc_cache: Dict[str, Doc] = {}
        self._doc_cache_lock = threading.RLock()
        
        # Initialize matchers and patterns
        self._setup_medication_ner()
        self._add_causal_patterns()
        
        # Initialize token alignment debugger
        self.token_debugger = TokenAlignmentDebugger(
            debug_dir=config.get('debug_dir')
        ) if debug else None
    
    @property
    def nlp(self) -> spacy.Language:
        """Get the spaCy language model."""
        return self.nlp
    
    def _get_or_create_doc(self, text: str) -> Doc:
        """Get or create a spaCy Doc object for the given text.
        
        Args:
            text: Input text
            
        Returns:
            spaCy Doc object
        """
        # Check cache first
        with self._doc_cache_lock:
            if text in self._doc_cache:
                return self._doc_cache[text]
            
            # Create new doc
            doc = self.nlp(text)
            self._doc_cache[text] = doc
            return doc
    
    def attribute_responses(
        self,
        text: str,
        medications: List[Dict[str, Any]],
        confidence_threshold: float = 0.5
    ) -> Dict[str, Any]:
        """Attribute responses to medications with confidence scoring.
        
        Args:
            text: Input text
            medications: List of medication dictionaries
            confidence_threshold: Minimum confidence score
            
        Returns:
            Dictionary with attributed responses and confidence scores
        """
        # Get or create document
        doc = self._get_or_create_doc(text)
        
        # Resolve coreferences and get confidence scores
        resolved_doc, coref_confidence = self._resolve_coreferences(doc)
        
        # Get symptom matches from both original and resolved docs
        original_symptoms = self.symptom_matcher.match_symptoms(doc)
        resolved_symptoms = self.symptom_matcher.match_symptoms(resolved_doc)
        
        # Merge symptoms, preferring resolved ones but with confidence adjustment
        merged_symptoms = []
        seen_positions = set()
        
        # First add original symptoms
        for symptom in original_symptoms:
            start, end = symptom['position']
            merged_symptoms.append(symptom)
            seen_positions.add((start, end))
        
        # Then add resolved symptoms with confidence adjustment
        for symptom in resolved_symptoms:
            start, end = symptom['position']
            if (start, end) not in seen_positions:
                # Adjust confidence based on coreference confidence
                min_coref_conf = min(coref_confidence[i] for i in range(start, end))
                symptom['confidence'] = symptom.get('confidence', 1.0) * min_coref_conf
                merged_symptoms.append(symptom)
                seen_positions.add((start, end))
        
        # Get temporal information
        temporal_info = self.temporal_parser.parse_temporal_info(resolved_doc)
        
        # Get emoji signals
        emoji_signals = self.emoji_processor.process_emoji_signals(resolved_doc)
        
        # Attribute responses using merged symptoms
        attributed_responses = []
        for med in medications:
            med_name = med['name']
            med_span = med.get('span', None)
            
            # Find relevant symptoms
            med_symptoms = []
            for symptom in merged_symptoms:
                if self._is_symptom_related_to_medication(symptom, med_span, resolved_doc):
                    med_symptoms.append(symptom)
            
            # Calculate confidence scores
            confidence_scores = self._calculate_confidence_scores(
                med_symptoms,
                temporal_info,
                emoji_signals,
                med_span,
                resolved_doc
            )
            
            # Filter by confidence threshold
            if confidence_scores['overall'] >= confidence_threshold:
                attributed_responses.append({
                    'medication': med_name,
                    'symptoms': med_symptoms,
                    'confidence_scores': confidence_scores,
                    'temporal_info': temporal_info,
                    'emoji_signals': emoji_signals
                })
        
        return {
            'attributed_responses': attributed_responses,
            'confidence_threshold': confidence_threshold,
            'total_symptoms': len(merged_symptoms),
            'total_medications': len(medications)
        }
    
    def clear_caches(self) -> None:
        """Clear all caches."""
        with self.cache_lock:
            self.causality_cache.clear()
        with self._doc_cache_lock:
            self._doc_cache.clear()
        self.symptom_matcher.clear_cache()
        self.temporal_parser.clear_cache()
        self.emoji_processor.clear_cache()
    
    def get_cache_stats(self) -> Dict[str, Any]:
        """Get cache statistics.
        
        Returns:
            Dictionary with cache statistics
        """
        return {
            'causality_cache': self.causality_cache.get_stats(),
            'symptom_matcher': self.symptom_matcher.get_cache_stats(),
            'temporal_parser': self.temporal_parser.get_cache_stats(),
            'emoji_processor': self.emoji_processor.get_cache_stats(),
            'doc_cache_size': len(self._doc_cache)
        }

    def _setup_medication_ner(self) -> None:
        """Set up medication NER using MedDictionary patterns and SciSpacy fallback."""
        # Create EntityRuler if it doesn't exist
        if "entity_ruler" not in self.nlp.pipe_names:
            ruler = EntityRuler(self.nlp)
            self.nlp.add_pipe(ruler)
        
        # Get patterns from MedDictionary
        patterns = []
        for med_name, med_info in self.med_dict.items():
            # Add main name pattern
            patterns.append({
                "label": "MEDICATION",
                "pattern": med_name
            })
            
            # Add alternative names
            if 'alt_names' in med_info:
                for alt_name in med_info['alt_names']:
                    patterns.append({
                        "label": "MEDICATION",
                        "pattern": alt_name
                    })
            
            # Add common misspellings
            if 'misspellings' in med_info:
                for misspelling in med_info['misspellings']:
                    patterns.append({
                        "label": "MEDICATION",
                        "pattern": misspelling
                    })
        
        # Add patterns to ruler
        self.nlp.get_pipe("entity_ruler").add_patterns(patterns)
        
        logger.info(f"Added {len(patterns)} medication patterns to NER")

    def normalize_medication_name(self, med_name: str) -> str:
        """Normalize medication name to standard form.
        
        Args:
            med_name: Input medication name
            
        Returns:
            Normalized medication name
        """
        # Try to find in dictionary
        med_info = self.med_dict.get(med_name.lower())
        if med_info:
            return med_info['generic_name']
        
        # If not found, try fuzzy matching
        best_match = self.med_dict.find_closest_match(med_name)
        if best_match:
            return best_match['generic_name']
        
        # If still not found, return original
        return med_name

    def _add_causal_patterns(self) -> None:
        """Add causal patterns to the matcher using config-defined patterns."""
        # Clear existing patterns
        self.causal_matcher = Matcher(self.nlp.vocab)
        
        # Store pattern weights for later lookup
        self.pattern_weights = {
            'direct': {},
            'indirect': {}
        }
        
        # Direct causal patterns
        for pattern in self.config['causal_patterns']['direct']:
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
            pattern_id = f"DIRECT_{pattern_text.replace(' ', '_')}"
            self.causal_matcher.add(pattern_id, [matcher_pattern])
            
            # Store weight for this pattern
            self.pattern_weights['direct'][pattern_id] = weight
            
            logger.debug(f"Added direct causal pattern: {pattern_text} with weight {weight}")
        
        # Indirect causal patterns
        for pattern in self.config['causal_patterns']['indirect']:
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
            pattern_id = f"INDIRECT_{pattern_text.replace(' ', '_')}"
            self.causal_matcher.add(pattern_id, [matcher_pattern])
            
            # Store weight for this pattern
            self.pattern_weights['indirect'][pattern_id] = weight
            
            logger.debug(f"Added indirect causal pattern: {pattern_text} with weight {weight}")

    def _prune_overlapping_spans(self, matches: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Prune overlapping spans, keeping the longer/more specific ones.
        
        Args:
            matches: List of symptom matches with positions
            
        Returns:
            Pruned list of matches
        """
        if not matches:
            return []
            
        # Sort by span length (descending) and then by start position
        sorted_matches = sorted(
            matches,
            key=lambda x: (
                -(x['position'][1] - x['position'][0]),  # Longer spans first
                x['position'][0]  # Then by start position
            )
        )
        
        pruned_matches = []
        used_positions = set()
        
        for match in sorted_matches:
            start, end = match['position']
            # Check if this span overlaps with any used positions
            if not any(
                used_start <= start < used_end or used_start < end <= used_end
                for used_start, used_end in used_positions
            ):
                pruned_matches.append(match)
                used_positions.add((start, end))
        
        return pruned_matches

    def batch_attribute_responses(self, text: str, medications: List[str]) -> Dict[str, Any]:
        """Attribute response dimensions to medications in text (batch processing version).
        
        This is a legacy method focused on batch processing of responses.
        For single-instance attribution with confidence scoring, use attribute_responses().
        
        Args:
            text: Input text
            medications: List of medications
            
        Returns:
            Dictionary with response attributions including dimension scores and confidence
        """
        # Parse text with spaCy
        doc = self.nlp(text)
        
        # Only resolve coreferences in debug mode
        if self.debug:
            doc = self._resolve_coreferences(doc)
            logger.debug("Using coreference-resolved doc for debugging")
        
        # Normalize medication names
        normalized_meds = {}
        for med in medications:
            normalized = self.normalize_medication_name(med)
            normalized_meds[normalized.lower()] = normalized
        
        # Get temporal status
        temporal_status = self.temporal_parser.parse_temporal(doc)
        processed_temporal = self._process_temporal_status(temporal_status)
        
        logger.debug(f"Detected temporal status: {processed_temporal}")
        
        # Initialize evidence collection
        med_evidence = {med: {
            'activation': [],
            'emotional': [],
            'metabolic': []
        } for med in normalized_meds.values()}
        
        # Initialize signal counts
        signal_counts = {med: {
            'activation': 0,
            'emotional': 0,
            'metabolic': 0
        } for med in normalized_meds.values()}
        
        # Initialize signal strengths
        signal_strengths = {med: {
            'activation': 0.0,
            'emotional': 0.0,
            'metabolic': 0.0
        } for med in normalized_meds.values()}
        
        # Detect emoji signals
        emoji_present, emoji_signals = self.emoji_processor.process_emoji(doc, list(normalized_meds.values()))
        
        logger.debug(f"Emoji signals detected: {emoji_present}")
        if emoji_present:
            logger.debug(f"Emoji signals: {emoji_signals}")
        
        # Process each sentence with coreference resolution
        for sent in doc.sents:
            # Find medications in sentence
            med_ents = [ent for ent in sent.ents if ent.label_ == "MEDICATION"]
            
            # Get symptoms in sentence and prune overlapping spans
            symptoms = self.symptom_matcher.match_symptoms(sent)
            pruned_symptoms = self._prune_overlapping_spans(symptoms)
            
            # Process each medication in sentence
            for med_ent in med_ents:
                # Normalize medication name
                med_name = self.normalize_medication_name(med_ent.text)
                if med_name.lower() not in normalized_meds:
                    continue
                
                # Get temporal status for this medication
                med_temporal = processed_temporal.get(med_name, {})
                
                # Process symptoms
                for symptom in pruned_symptoms:
                    # Calculate evidence segment
                    evidence = self._create_evidence_segment(
                        doc, med_ent, symptom, med_temporal
                    )
                    
                    # Add to debugger if enabled
                    if self.token_debugger:
                        # Check if symptom should be ignored
                        ignored = False
                        ignore_reason = None
                        
                        if evidence['weighted_polarity'] < self.confidence_thresholds['min_signal']:
                            ignored = True
                            ignore_reason = "Below minimum signal threshold"
                        elif evidence['causality_score'] < self.confidence_thresholds['min_causality']:
                            ignored = True
                            ignore_reason = "Below minimum causality threshold"
                        
                        self.token_debugger.add_match(
                            med_name=med_name,
                            symptom=symptom,
                            doc=doc,
                            ignored=ignored,
                            ignore_reason=ignore_reason
                        )
                    
                    # Add to evidence collection if not ignored
                    if not ignored:
                        med_evidence[med_name][symptom['dimension']].append(evidence)
                        
                        # Update signal counts and strengths
                        signal_counts[med_name][symptom['dimension']] += 1
                        signal_strengths[med_name][symptom['dimension']] += evidence['weighted_polarity']
        
        # Calculate final dimension scores
        final_scores = {}
        final_confidence = {}
        for med in normalized_meds.values():
            final_scores[med] = {}
            final_confidence[med] = {}
            
            for dimension in ['activation', 'emotional', 'metabolic']:
                evidence = med_evidence[med][dimension]
                
                if not evidence:
                    final_scores[med][dimension] = 0.0
                    final_confidence[med][dimension] = 0.0
                    continue
                
                # Calculate weighted average of evidence segments
                total_weight = 0.0
                weighted_sum = 0.0
                confidence_sum = 0.0
                
                for segment in evidence:
                    # Calculate segment weight based on multiple factors
                    weight = (
                        segment['causality_score'] * self.confidence_weights['causality'] +
                        segment['proximity_weight'] * self.confidence_weights['proximity'] +
                        segment['contextual_confidence'] * self.confidence_weights['context']
                    )
                    
                    # Apply intensity modifier
                    weight *= segment['intensity']
                    
                    # Apply temporal recency weight if available
                    if med in processed_temporal:
                        weight *= processed_temporal[med]['recency_confidence']
                    
                    # Add to weighted sum
                    weighted_sum += segment['weighted_polarity'] * weight
                    total_weight += weight
                    confidence_sum += segment['contextual_confidence']
                
                # Calculate final score and confidence
                if total_weight > 0:
                    final_scores[med][dimension] = weighted_sum / total_weight
                    final_confidence[med][dimension] = confidence_sum / len(evidence)
                else:
                    final_scores[med][dimension] = 0.0
                    final_confidence[med][dimension] = 0.0
            
            # Apply signal strength filtering
            if (signal_counts[med]['activation'] < MIN_SIGNAL_COUNT or 
                signal_strengths[med]['activation'] < MIN_SIGNAL_STRENGTH):
                final_scores[med]['activation'] = 0.0
                final_confidence[med]['activation'] = 0.0
            if (signal_counts[med]['emotional'] < MIN_SIGNAL_COUNT or 
                signal_strengths[med]['emotional'] < MIN_SIGNAL_STRENGTH):
                final_scores[med]['emotional'] = 0.0
                final_confidence[med]['emotional'] = 0.0
            if (signal_counts[med]['metabolic'] < MIN_SIGNAL_COUNT or 
                signal_strengths[med]['metabolic'] < MIN_SIGNAL_STRENGTH):
                final_scores[med]['metabolic'] = 0.0
                final_confidence[med]['metabolic'] = 0.0
        
        # Save debug output if enabled
        if self.token_debugger:
            self.token_debugger.save_debug_output()
        
        return {
            'medication_responses': med_evidence,
            'dimension_scores': final_scores,
            'dimension_confidence': final_confidence,
            'temporal_status': processed_temporal,
            'emoji_signals': emoji_signals if emoji_present else None,
            'signal_counts': signal_counts,
            'signal_strengths': signal_strengths
        }

    def _create_evidence_segment(self, doc: Doc, med_ent: Span, 
                               symptom: Dict[str, Any], 
                               temporal_status: Optional[Dict[str, Any]]) -> Dict[str, Any]:
        """Create an evidence segment for a symptom.
        
        Args:
            doc: spaCy Doc
            med_ent: Medication entity span
            symptom: Matched symptom information
            temporal_status: Temporal status information
            
        Returns:
            Dictionary with evidence segment information
        """
        # Calculate causality score
        causality_score = self._calculate_causal_confidence(
            doc, med_ent, doc[symptom['position'][0]:symptom['position'][1]]
        )
        
        # Calculate proximity weight
        proximity_weight = self._calculate_proximity_weight(
            doc, symptom['position'][0], med_ent
        )
        
        # Calculate intensity
        intensity = self._detect_intensity(
            doc, symptom['position'][0], symptom['position'][1]
        )
        
        # Calculate contextual confidence
        contextual_confidence = self._calculate_contextual_confidence(
            doc, med_ent, [symptom], temporal_status
        )[symptom['dimension']]
        
        # Calculate weighted polarity
        weighted_polarity = (
            symptom['polarity'] * 
            causality_score * 
            proximity_weight * 
            intensity * 
            contextual_confidence
        )
        
        return {
            'type': 'symptom',
            'text': symptom['text'],
            'canonical': symptom['canonical'],
            'polarity': symptom['polarity'],
            'position': symptom['position'],
            'causality_score': causality_score,
            'proximity_weight': proximity_weight,
            'intensity': intensity,
            'contextual_confidence': contextual_confidence,
            'weighted_polarity': weighted_polarity
        }

    def _calculate_emoji_causality(self, doc: Doc, med_ent: Span, 
                                 emoji_signal: Dict[str, Any]) -> float:
        """Calculate causality score for emoji signal.
        
        Args:
            doc: spaCy Doc
            med_ent: Medication entity span
            emoji_signal: Emoji signal information
            
        Returns:
            Causality score between 0 and 1
        """
        # Get emoji token
        emoji_token = doc[emoji_signal['position'][0]]
        
        # Calculate dependency distance
        distance = self._get_dependency_distance(med_ent.root, emoji_token)
        
        # Convert distance to causality score
        max_distance = 5
        if distance > max_distance:
            return 0.0
        
        raw_confidence = max(0.0, 1.0 - (distance / max_distance))
        
        # Apply calibration if available
        if self.is_calibrated and 'emoji' in self.calibration_models:
            # Reshape for sklearn
            X = np.array([raw_confidence]).reshape(-1, 1)
            calibrated_confidence = float(self.calibration_models['emoji'].predict(X)[0])
            return calibrated_confidence
        
        return raw_confidence

    def _calculate_proximity_weight(self, doc: Doc, token_idx: int, med_ent: spacy.tokens.Span) -> float:
        """Calculate proximity weight between token and medication.
        
        Args:
            doc: spaCy Doc
            token_idx: Token index
            med_ent: Medication entity span
            
        Returns:
            Proximity weight between 0 and 1
        """
        # Calculate token distance
        distance = min(abs(token_idx - med_ent.start), abs(token_idx - med_ent.end))
        
        # Get decay factor and max distance from config
        decay_factor = self.config['proximity']['decay_factor']
        max_distance = self.config['proximity']['max_distance']
        
        # Return 0 if beyond max distance
        if distance > max_distance:
            return 0.0
        
        # Calculate weight using exponential decay
        weight = decay_factor ** distance
        
        # Normalize to [0,1] range
        return max(0.0, min(1.0, weight))

    def _detect_intensity(self, doc: Doc, start_idx: int, end_idx: int) -> float:
        """Detect intensity modifiers around a span.
        
        Args:
            doc: spaCy Doc
            start_idx: Start index of span
            end_idx: End index of span
            
        Returns:
            Intensity multiplier
        """
        intensity = 1.0
        
        # Check tokens in window
        window_size = 3
        start = max(0, start_idx - window_size)
        end = min(len(doc), end_idx + window_size)
        
        for token in doc[start:end]:
            # Check for intensifiers
            if token.text.lower() in ['very', 'extremely', 'incredibly', 'really', 'so']:
                intensity *= 1.5
            elif token.text.lower() in ['slightly', 'somewhat', 'a bit', 'kind of']:
                intensity *= 0.7
            elif token.text.lower() in ['not', 'no', 'never', 'none']:
                intensity *= -1.0
        
        return intensity

    def _get_cache_key(self, doc: Doc, sent: Span, med_ent: Span) -> str:
        """Generate a cache key for sentence-level caching.
        
        Args:
            doc: spaCy Doc
            sent: Sentence span
            med_ent: Medication entity span
            
        Returns:
            Cache key string
        """
        # Include sentence text and medication position for precise matching
        key_parts = [
            str(doc),
            str(sent.start_char),
            str(sent.end_char),
            str(med_ent.start_char),
            str(med_ent.end_char)
        ]
        return hashlib.md5('|'.join(key_parts).encode()).hexdigest()

    def _calculate_causal_confidence(self, doc: Doc, med_ent: Span, 
                                   symptom_span: Span) -> float:
        """Calculate causal confidence using multiple factors.
        
        Args:
            doc: spaCy Doc
            med_ent: Medication entity span
            symptom_span: Symptom span
            
        Returns:
            Confidence score between 0 and 1
        """
        # Get root tokens
        med_root = med_ent.root
        symptom_root = symptom_span.root
        
        # Calculate dependency distance and path quality
        distance, path_quality = self._get_dependency_distance(med_root, symptom_root)
        
        # Convert distance to confidence score
        max_distance = self.cross_sentence_max_distance
        if distance > max_distance:
            return 0.0
        
        base_confidence = max(0.0, 1.0 - (distance / max_distance))
        base_confidence *= path_quality  # Apply path quality modifier
        
        # Check for causal patterns
        matches = self.causal_matcher(doc)
        pattern_confidence = 0.0
        
        for match_id, start, end in matches:
            match_span = doc[start:end]
            # Use spaCy's span overlap functionality for more precise matching
            if match_span.has_overlap(med_ent):
                # Get pattern type and text
                pattern_id = self.nlp.vocab.strings[match_id]
                pattern_type = "direct" if pattern_id.startswith("DIRECT_") else "indirect"
                
                # Get pattern weight
                pattern_confidence = max(pattern_confidence, 
                                      self.pattern_weights[pattern_type].get(pattern_id, 0.0))
                
                logger.debug(f"Matched causal pattern: {pattern_id} with confidence {pattern_confidence}")
        
        # Check temporal ordering
        temporal_confidence = self._check_temporal_ordering(doc, med_ent, symptom_span)
        
        # Check event ordering
        event_confidence = self._check_event_ordering(doc, med_ent, symptom_span)
        
        # Combine confidences with weights
        weights = self.causal_confidence_weights
        
        final_confidence = (
            weights['base'] * base_confidence +
            weights['pattern'] * pattern_confidence +
            weights['temporal'] * temporal_confidence +
            weights['event'] * event_confidence
        )
        
        # Apply uncertainty penalty if needed
        if self._has_uncertainty_markers(doc, med_ent, symptom_span):
            final_confidence *= 0.8
        
        return max(0.0, min(1.0, final_confidence))

    def _check_temporal_ordering(self, doc: Doc, med_ent: Span, symptom_span: Span) -> float:
        """Check temporal ordering between medication and symptom.
        
        Args:
            doc: spaCy Doc
            med_ent: Medication entity span
            symptom_span: Symptom span
            
        Returns:
            Confidence score between 0 and 1
        """
        # Get temporal markers with weights
        temporal_markers = {
            'before': {
                'markers': ['before', 'prior to', 'earlier than', 'preceding'],
                'weight': 0.2
            },
            'after': {
                'markers': ['after', 'since', 'following', 'subsequent to'],
                'weight': 0.8
            },
            'during': {
                'markers': ['during', 'while', 'whilst', 'throughout'],
                'weight': 0.6
            },
            'immediate': {
                'markers': ['immediately', 'right after', 'straight after'],
                'weight': 0.9
            },
            'delayed': {
                'markers': ['eventually', 'later', 'after some time'],
                'weight': 0.4
            }
        }
        
        # Initialize confidence
        confidence = 0.5  # Default neutral confidence
        
        # Get tokens between medication and symptom
        start = min(med_ent.end, symptom_span.start)
        end = max(med_ent.start, symptom_span.end)
        between_tokens = doc[start:end]
        
        # Check for temporal markers
        found_markers = []
        for marker_type, marker_info in temporal_markers.items():
            for token in between_tokens:
                if token.text.lower() in marker_info['markers']:
                    found_markers.append((marker_type, marker_info['weight']))
        
        if found_markers:
            # Use highest weight marker
            confidence = max(m[1] for m in found_markers)
            
            # Check for negation
            for token in between_tokens:
                if token.text.lower() in ['not', 'no', 'never']:
                    confidence = 1.0 - confidence
                    break
        
        # Check for relative time frames
        time_frames = {
            'short': ['day', 'week', 'month'],
            'medium': ['month', 'year'],
            'long': ['year', 'years']
        }
        
        for frame_type, frames in time_frames.items():
            for token in between_tokens:
                if token.text.lower() in frames:
                    if frame_type == 'short':
                        confidence *= 1.2
                    elif frame_type == 'medium':
                        confidence *= 1.0
                    else:
                        confidence *= 0.8
        
        return max(0.0, min(1.0, confidence))

    def _check_event_ordering(self, doc: Doc, med_ent: Span, symptom_span: Span) -> float:
        """Check event ordering between medication and symptom.
        
        Args:
            doc: spaCy Doc
            med_ent: Medication entity span
            symptom_span: Symptom span
            
        Returns:
            Confidence score between 0 and 1
        """
        # Get event markers with weights and categories
        event_markers = {
            'direct_cause': {
                'markers': ['causes', 'caused', 'results in', 'led to', 'triggers', 'triggered'],
                'weight': 0.9
            },
            'indirect_cause': {
                'markers': ['contributes to', 'contributed to', 'plays a role in', 'influences'],
                'weight': 0.7
            },
            'prevent': {
                'markers': ['prevents', 'prevented', 'stops', 'stopped', 'blocks', 'blocked'],
                'weight': 0.8
            },
            'help': {
                'markers': ['helps', 'helped', 'improves', 'improved', 'alleviates', 'alleviated'],
                'weight': 0.6
            },
            'worsen': {
                'markers': ['worsens', 'worsened', 'exacerbates', 'exacerbated', 'aggravates'],
                'weight': 0.7
            }
        }
        
        # Initialize confidence
        confidence = 0.5  # Default neutral confidence
        
        # Get tokens between medication and symptom
        start = min(med_ent.end, symptom_span.start)
        end = max(med_ent.start, symptom_span.end)
        between_tokens = doc[start:end]
        
        # Check for event markers
        found_markers = []
        for marker_type, marker_info in event_markers.items():
            # Check for multi-word markers
            for i in range(len(between_tokens) - 1):
                phrase = ' '.join(t.text.lower() for t in between_tokens[i:i+2])
                if phrase in marker_info['markers']:
                    found_markers.append((marker_type, marker_info['weight']))
            
            # Check single tokens
            for token in between_tokens:
                if token.text.lower() in marker_info['markers']:
                    found_markers.append((marker_type, marker_info['weight']))
        
        if found_markers:
            # Use highest weight marker
            confidence = max(m[1] for m in found_markers)
            
            # Check for negation and uncertainty
            for token in between_tokens:
                if token.text.lower() in ['not', 'no', 'never']:
                    confidence = 1.0 - confidence
                    break
                elif token.text.lower() in ['maybe', 'perhaps', 'possibly', 'might']:
                    confidence *= 0.7
                    break
        
        # Check for intensity modifiers
        intensity_modifiers = {
            'high': ['significantly', 'dramatically', 'substantially', 'greatly'],
            'medium': ['somewhat', 'moderately', 'slightly'],
            'low': ['barely', 'hardly', 'scarcely']
        }
        
        for intensity, modifiers in intensity_modifiers.items():
            for token in between_tokens:
                if token.text.lower() in modifiers:
                    if intensity == 'high':
                        confidence *= 1.2
                    elif intensity == 'medium':
                        confidence *= 1.0
                    else:
                        confidence *= 0.8
        
        return max(0.0, min(1.0, confidence))

    def _has_uncertainty_markers(self, doc: Doc, med_ent: Span, symptom_span: Span) -> bool:
        """Check for uncertainty markers between medication and symptom.
        
        Args:
            doc: spaCy Doc
            med_ent: Medication entity span
            symptom_span: Symptom span
            
        Returns:
            True if uncertainty markers are present
        """
        uncertainty_markers = {
            'maybe', 'perhaps', 'possibly', 'might', 'could', 'may',
            'seems', 'appears', 'looks like', 'not sure', 'unsure',
            'think', 'believe', 'guess', 'suppose'
        }
        
        # Get tokens between medication and symptom
        start = min(med_ent.end, symptom_span.start)
        end = max(med_ent.start, symptom_span.end)
        between_tokens = doc[start:end]
        
        # Check for uncertainty markers
        for token in between_tokens:
            if token.text.lower() in uncertainty_markers:
                return True
            
            # Check for multi-word markers
            if token.i + 1 < len(doc):
                phrase = ' '.join([token.text.lower(), doc[token.i + 1].text.lower()])
                if phrase in uncertainty_markers:
                    return True
        
        return False

    def _get_dependency_distance(self, token1: Token, token2: Token) -> Tuple[int, float]:
        """Calculate dependency distance and path quality between two tokens.
        
        Args:
            token1: First token
            token2: Second token
            
        Returns:
            Tuple of (distance, path_quality)
        """
        # Get path from token1 to token2
        path = []
        current = token1
        
        # Track path quality factors
        path_quality = 1.0
        has_causal_dep = False
        has_temporal_dep = False
        
        while current != token2 and current.head != current:
            path.append(current)
            
            # Check dependency type
            if current.dep_ in ['nsubj', 'dobj', 'iobj']:
                path_quality *= 1.2  # Strong syntactic connection
            elif current.dep_ in ['prep', 'pobj']:
                path_quality *= 0.8  # Weaker connection
            
            # Check for causal dependencies
            if current.dep_ in ['ccomp', 'xcomp', 'advcl']:
                has_causal_dep = True
                path_quality *= 1.3
            
            # Check for temporal dependencies
            if current.dep_ in ['advmod', 'npadvmod'] and current.text.lower() in [
                'before', 'after', 'during', 'while', 'since'
            ]:
                has_temporal_dep = True
                path_quality *= 1.2
            
            current = current.head
        
        if current == token2:
            # Apply quality modifiers
            if has_causal_dep:
                path_quality *= 1.5
            if has_temporal_dep:
                path_quality *= 1.3
            
            return len(path), min(1.0, path_quality)
        
        # If no direct path, try reverse
        path = []
        current = token2
        path_quality = 1.0
        has_causal_dep = False
        has_temporal_dep = False
        
        while current != token1 and current.head != current:
            path.append(current)
            
            # Check dependency type
            if current.dep_ in ['nsubj', 'dobj', 'iobj']:
                path_quality *= 1.2
            elif current.dep_ in ['prep', 'pobj']:
                path_quality *= 0.8
            
            # Check for causal dependencies
            if current.dep_ in ['ccomp', 'xcomp', 'advcl']:
                has_causal_dep = True
                path_quality *= 1.3
            
            # Check for temporal dependencies
            if current.dep_ in ['advmod', 'npadvmod'] and current.text.lower() in [
                'before', 'after', 'during', 'while', 'since'
            ]:
                has_temporal_dep = True
                path_quality *= 1.2
            
            current = current.head
        
        if current == token1:
            # Apply quality modifiers
            if has_causal_dep:
                path_quality *= 1.5
            if has_temporal_dep:
                path_quality *= 1.3
            
            return len(path), min(1.0, path_quality)
        
        # If no path found, return large distance and low quality
        return 100, 0.1

    def _calculate_contextual_confidence(self, doc: Doc, med_ent: spacy.tokens.Span,
                                       symptoms: List[Dict[str, Any]], 
                                       temporal_status: Dict[str, Any]) -> Dict[str, Any]:
        """Calculate contextual confidence scores.
        
        Args:
            doc: spaCy Doc
            med_ent: Medication entity span
            symptoms: List of matched symptoms
            temporal_status: Temporal status information
            
        Returns:
            Dictionary of confidence scores by dimension
        """
        confidence_scores = {
            'activation': 0.0,
            'emotional': 0.0,
            'metabolic': 0.0
        }
        
        # Initialize detailed confidence breakdown if debug is enabled
        if self.debug:
            confidence_details = {
                'activation': {'base': 0.0, 'causality': 0.0, 'temporal': 0.0, 'total': 0.0},
                'emotional': {'base': 0.0, 'causality': 0.0, 'temporal': 0.0, 'total': 0.0},
                'metabolic': {'base': 0.0, 'causality': 0.0, 'temporal': 0.0, 'total': 0.0}
            }
        
        # Group symptoms by dimension
        dimension_symptoms = {
            'activation': [],
            'emotional': [],
            'metabolic': []
        }
        
        for symptom in symptoms:
            dimension = symptom['dimension']
            if dimension in dimension_symptoms:
                dimension_symptoms[dimension].append(symptom)
        
        # Calculate confidence for each dimension
        for dimension, dim_symptoms in dimension_symptoms.items():
            if not dim_symptoms:
                continue
            
            # Base confidence from number of symptoms
            base_confidence = min(1.0, len(dim_symptoms) * 0.3)
            
            # Calculate causal confidence for each symptom
            causal_confidences = []
            for symptom in dim_symptoms:
                symptom_span = doc[symptom['position'][0]:symptom['position'][1]]
                causal_conf = self._calculate_causal_confidence(doc, med_ent, symptom_span)
                causal_confidences.append(causal_conf)
            
            causal_confidence = max(causal_confidences) if causal_confidences else 0.0
            
            # Temporal confidence
            temporal_confidence = 0.5
            if temporal_status:
                if temporal_status['status'] == 'current':
                    temporal_confidence = 0.9
                elif temporal_status['status'] == 'past':
                    temporal_confidence = 0.7
            
            # Calculate raw confidence scores
            raw_scores = {
                'base': base_confidence,
                'causality': causal_confidence,
                'temporal': temporal_confidence
            }
            
            # Apply calibration if available
            calibrated_scores = self._apply_calibration(raw_scores)
            
            # Calculate final confidence
            confidence = (
                self.confidence_weights['base'] * calibrated_scores['base'] +
                self.confidence_weights['causality'] * calibrated_scores['causality'] +
                self.confidence_weights['temporal'] * calibrated_scores['temporal']
            )
            
            # Apply dimension weight
            confidence *= self.dimension_weights[dimension]
            
            # Apply thresholds
            confidence = max(self.confidence_thresholds['floor'],
                           min(self.confidence_thresholds['ceiling'], confidence))
            
            confidence_scores[dimension] = confidence
            
            # Store detailed breakdown if debug is enabled
            if self.debug:
                confidence_details[dimension] = {
                    'base': calibrated_scores['base'],
                    'causality': calibrated_scores['causality'],
                    'temporal': calibrated_scores['temporal'],
                    'total': confidence
                }
        
        # Add detailed confidence breakdown to results if debug is enabled
        if self.debug:
            confidence_scores['confidence_details'] = confidence_details
        
        return confidence_scores

    def process_dataframe(self, df: pd.DataFrame, text_col: str, med_col: str, 
                         temporal_col: Optional[str] = None) -> pd.DataFrame:
        """Process a DataFrame of texts and medications.
        
        Args:
            df: Input DataFrame
            text_col: Column containing text
            med_col: Column containing medications
            temporal_col: Optional column containing temporal status
            
        Returns:
            DataFrame with response attribution results
        """
        result_df = df.copy()
        
        # Process each row with optional progress bar
        attribution_results = []
        iterator = tqdm(df.iterrows(), total=len(df)) if self.debug else df.iterrows()
        
        for _, row in iterator:
            # Get medications
            medications = row[med_col]
            if not isinstance(medications, list):
                medications = [medications]
            
            # Get temporal status if available
            temporal_status = row[temporal_col] if temporal_col else None
            
            # Attribute responses
            results = self.attribute_responses(row[text_col], medications)
            
            # Add temporal status if provided
            if temporal_status:
                results['temporal_status'] = temporal_status
            
            attribution_results.append(results)
        
        # Add results to DataFrame
        result_df['response_dimension_scores'] = [r['dimension_scores'] for r in attribution_results]
        result_df['response_dimension_confidence'] = [r['dimension_confidence'] for r in attribution_results]
        result_df['medication_responses'] = [r['medication_responses'] for r in attribution_results]
        result_df['temporal_status'] = [r['temporal_status'] for r in attribution_results]
        result_df['emoji_signals'] = [r['emoji_signals'] for r in attribution_results]
        
        return result_df

    def attribute_with_uncertainty(self, text: str, medications: List[str], 
                                 n_samples: int = 10, dropout_rate: float = 0.1) -> Dict[str, Any]:
        """Attribute responses with uncertainty estimates using Monte Carlo Dropout.
        
        Args:
            text: Input text
            medications: List of medications
            n_samples: Number of Monte Carlo samples
            dropout_rate: Dropout rate for uncertainty estimation
            
        Returns:
            Dictionary with response attributions and uncertainty estimates
        """
        # Store original dropout rates
        original_dropout_rates = {}
        for pipe in self.nlp.pipe_names:
            if hasattr(self.nlp.get_pipe(pipe), 'model'):
                original_dropout_rates[pipe] = self.nlp.get_pipe(pipe).model.dropout
        
        # Enable dropout for uncertainty estimation
        for pipe in self.nlp.pipe_names:
            if hasattr(self.nlp.get_pipe(pipe), 'model'):
                self.nlp.get_pipe(pipe).model.dropout = dropout_rate
        
        try:
            # Collect samples
            samples = []
            for _ in range(n_samples):
                # Get base attribution with dropout enabled
                results = self.attribute_responses(text, medications)
                samples.append(results)
            
            # Calculate statistics
            uncertainty = {}
            for med in medications:
                med_uncertainty = {}
                for dimension in ['activation', 'emotional', 'metabolic']:
                    # Get scores across samples
                    scores = [s['dimension_scores'][med][dimension] for s in samples]
                    confidences = [s['dimension_confidence'][med][dimension] for s in samples]
                    
                    # Calculate mean and variance
                    mean_score = np.mean(scores)
                    score_variance = np.var(scores)
                    mean_confidence = np.mean(confidences)
                    
                    # Calculate uncertainty components
                    epistemic_uncertainty = score_variance  # Model uncertainty
                    aleatoric_uncertainty = 1.0 - mean_confidence  # Data uncertainty
                    
                    # Combine uncertainties with configurable weights
                    uncertainty_weights = self.config['uncertainty_weights']
                    
                    total_uncertainty = (
                        uncertainty_weights['epistemic'] * epistemic_uncertainty +
                        uncertainty_weights['aleatoric'] * aleatoric_uncertainty
                    )
                    
                    # Store uncertainty breakdown
                    med_uncertainty[dimension] = {
                        'total': total_uncertainty,
                        'epistemic': epistemic_uncertainty,
                        'aleatoric': aleatoric_uncertainty,
                        'mean_score': mean_score,
                        'score_std': np.std(scores),
                        'mean_confidence': mean_confidence
                    }
                
                uncertainty[med] = med_uncertainty
            
            # Use mean scores and confidences from samples
            final_results = {
                'medication_responses': samples[0]['medication_responses'],
                'dimension_scores': {},
                'dimension_confidence': {},
                'temporal_status': samples[0]['temporal_status'],
                'emoji_signals': samples[0]['emoji_signals'],
                'signal_counts': samples[0]['signal_counts'],
                'signal_strengths': samples[0]['signal_strengths'],
                'uncertainty': uncertainty
            }
            
            # Calculate mean scores and confidences
            for med in medications:
                final_results['dimension_scores'][med] = {}
                final_results['dimension_confidence'][med] = {}
                for dimension in ['activation', 'emotional', 'metabolic']:
                    scores = [s['dimension_scores'][med][dimension] for s in samples]
                    confidences = [s['dimension_confidence'][med][dimension] for s in samples]
                    
                    final_results['dimension_scores'][med][dimension] = np.mean(scores)
                    final_results['dimension_confidence'][med][dimension] = np.mean(confidences)
            
            return final_results
            
        finally:
            # Restore original dropout rates
            for pipe in self.nlp.pipe_names:
                if hasattr(self.nlp.get_pipe(pipe), 'model'):
                    self.nlp.get_pipe(pipe).model.dropout = original_dropout_rates[pipe]

    def get_uncertainty_breakdown(self, uncertainty: Dict[str, Any]) -> Dict[str, Any]:
        """Get detailed breakdown of uncertainty sources.
        
        Args:
            uncertainty: Uncertainty estimates from attribute_with_uncertainty
            
        Returns:
            Dictionary with uncertainty breakdown
        """
        breakdown = {}
        for med, med_uncertainty in uncertainty.items():
            breakdown[med] = {}
            for dimension, dim_uncertainty in med_uncertainty.items():
                # Calculate relative contributions
                total = dim_uncertainty['total']
                if total > 0:
                    epistemic_ratio = dim_uncertainty['epistemic'] / total
                    aleatoric_ratio = dim_uncertainty['aleatoric'] / total
                else:
                    epistemic_ratio = 0.0
                    aleatoric_ratio = 0.0
                
                breakdown[med][dimension] = {
                    'total_uncertainty': total,
                    'epistemic_ratio': epistemic_ratio,
                    'aleatoric_ratio': aleatoric_ratio,
                    'score_variability': dim_uncertainty['score_std'],
                    'confidence_level': dim_uncertainty['mean_confidence']
                }
        
        return breakdown

    def export_attribution(self, results: Dict[str, Any], format: str = 'json') -> str:
        """Export attribution results to specified format.
        
        Args:
            results: Attribution results
            format: Export format ('json' or 'csv')
            
        Returns:
            Exported results as string
        """
        if format == 'json':
            import json
            return json.dumps(results, indent=2)
        elif format == 'csv':
            # Convert to DataFrame
            rows = []
            for med, scores in results['dimension_scores'].items():
                row = {
                    'medication': med,
                    'activation_score': scores['activation'],
                    'emotional_score': scores['emotional'],
                    'metabolic_score': scores['metabolic'],
                    'activation_confidence': results['dimension_confidence'][med]['activation'],
                    'emotional_confidence': results['dimension_confidence'][med]['emotional'],
                    'metabolic_confidence': results['dimension_confidence'][med]['metabolic']
                }
                rows.append(row)
            
            df = pd.DataFrame(rows)
            
            # Export to CSV
            output = io.StringIO()
            df.to_csv(output, index=False)
            return output.getvalue()
        else:
            raise ValueError(f"Unsupported format: {format}")

    def _warmup_cache(self) -> None:
        """Warm up cache with common patterns."""
        common_patterns = [
            ("makes me feel", {"pattern": "makes me feel", "weight": 0.9}),
            ("causes me to", {"pattern": "causes me to", "weight": 0.9}),
            ("results in", {"pattern": "results in", "weight": 0.8}),
            ("seems to help with", {"pattern": "seems to help with", "weight": 0.7}),
            ("appears to reduce", {"pattern": "appears to reduce", "weight": 0.7}),
            ("might be helping", {"pattern": "might be helping", "weight": 0.6})
        ]
        self.causality_cache.warmup(common_patterns)

    def _resolve_coreferences(self, doc: Doc) -> Tuple[Doc, Dict[int, float]]:
        """Resolve coreferences in the document using coreferee.
        
        Args:
            doc: spaCy Doc to process
            
        Returns:
            Tuple of (resolved Doc, confidence scores for each token)
        """
        # Get coreference clusters
        clusters = doc._.coref_clusters
        
        # Create a copy of the doc to modify
        resolved_doc = doc.copy()
        
        # Track confidence scores for each token
        token_confidence = {i: 1.0 for i in range(len(doc))}
        
        # Replace pronouns with their antecedents
        for cluster in clusters:
            # Get the main mention (usually the first one)
            main_mention = cluster.main
            main_text = main_mention.text
            
            # Calculate base confidence for this cluster
            cluster_confidence = self._calculate_cluster_confidence(cluster)
            
            # Replace all mentions in the cluster with the main mention
            for mention in cluster.mentions:
                if mention != main_mention:
                    # Only replace if the mention is a pronoun
                    if mention.root.pos_ == "PRON":
                        # Get the span to replace
                        start = mention.start
                        end = mention.end
                        
                        # Replace the pronoun with the main mention
                        resolved_doc[start:end] = resolved_doc.vocab[main_text]
                        
                        # Update confidence scores for replaced tokens
                        for i in range(start, end):
                            token_confidence[i] = cluster_confidence
        
        return resolved_doc, token_confidence

    def _calculate_cluster_confidence(self, cluster) -> float:
        """Calculate confidence score for a coreference cluster.
        
        Args:
            cluster: Coreference cluster
            
        Returns:
            Confidence score between 0 and 1
        """
        # Base confidence factors
        confidence = 1.0
        
        # Penalize clusters with many mentions (more likely to be wrong)
        num_mentions = len(cluster.mentions)
        if num_mentions > 3:
            confidence *= 0.8
        
        # Penalize clusters with long distance between mentions
        max_distance = max(
            abs(m1.start - m2.start)
            for m1 in cluster.mentions
            for m2 in cluster.mentions
        )
        if max_distance > 10:
            confidence *= 0.9
        
        # Penalize clusters with different sentence mentions
        sentence_ids = {m.sent.start for m in cluster.mentions}
        if len(sentence_ids) > 1:
            confidence *= 0.85
        
        return max(0.0, min(1.0, confidence))

    def calibrate_confidence(self, gold_data: pd.DataFrame) -> Dict[str, Any]:
        """Calibrate confidence scores using gold standard data.
        
        Args:
            gold_data: DataFrame with gold standard annotations
            
        Returns:
            Dictionary with calibration metrics
        """
        logger.info("Starting confidence calibration")
        
        # Get calibration data for each component
        calibration_data = self._get_calibration_data(gold_data)
        
        # Calibrate each component
        for component, model in self.calibration_models.items():
            if component in calibration_data:
                X = calibration_data[component]['scores'].values.reshape(-1, 1)
                y = calibration_data[component]['true_scores'].values
                
                # Fit calibration model
                model.fit(X, y)
                
                # Calculate calibration metrics
                metrics = self._calculate_calibration_metrics(X, y, model)
                self.calibration_metrics[component] = metrics
                
                logger.info(f"{component} calibration metrics:")
                logger.info(f"  Brier score: {metrics['brier_score']:.4f}")
                logger.info(f"  Calibration error: {metrics['calibration_error']:.4f}")
                logger.info(f"  Confidence correlation: {metrics['confidence_correlation']:.4f}")
        
        self.is_calibrated = True
        return self.calibration_metrics

    def _get_calibration_data(self, gold_data: pd.DataFrame) -> Dict[str, Dict[str, np.ndarray]]:
        """Get calibration data for each component.
        
        Args:
            gold_data: DataFrame with gold standard annotations
            
        Returns:
            Dictionary mapping components to their calibration data
        """
        calibration_data = {}
        
        # Get symptom calibration data
        if 'symptom_confidence' in gold_data.columns and 'symptom_gold' in gold_data.columns:
            calibration_data['symptom'] = {
                'scores': gold_data['symptom_confidence'].values,
                'true_scores': gold_data['symptom_gold'].values
            }
        
        # Get temporal calibration data
        if 'temporal_confidence' in gold_data.columns and 'temporal_gold' in gold_data.columns:
            calibration_data['temporal'] = {
                'scores': gold_data['temporal_confidence'].values,
                'true_scores': gold_data['temporal_gold'].values
            }
        
        # Get causal calibration data
        if 'causal_confidence' in gold_data.columns and 'causal_gold' in gold_data.columns:
            calibration_data['causal'] = {
                'scores': gold_data['causal_confidence'].values,
                'true_scores': gold_data['causal_gold'].values
            }
        
        # Get emoji calibration data
        if 'emoji_confidence' in gold_data.columns and 'emoji_gold' in gold_data.columns:
            calibration_data['emoji'] = {
                'scores': gold_data['emoji_confidence'].values,
                'true_scores': gold_data['emoji_gold'].values
            }
        
        # Get overall calibration data
        if all(col in gold_data.columns for col in ['symptom_confidence', 'temporal_confidence', 
                                                   'causal_confidence', 'emoji_confidence', 'overall_gold']):
            # Calculate weighted average of component confidences
            overall_scores = (
                self.confidence_weights['symptom'] * gold_data['symptom_confidence'] +
                self.confidence_weights['temporal'] * gold_data['temporal_confidence'] +
                self.confidence_weights['causal'] * gold_data['causal_confidence'] +
                self.confidence_weights['emoji'] * gold_data['emoji_confidence']
            )
            
            calibration_data['overall'] = {
                'scores': overall_scores.values,
                'true_scores': gold_data['overall_gold'].values
            }
        
        return calibration_data

    def _calculate_calibration_metrics(self, X: np.ndarray, y: np.ndarray, 
                                     model: IsotonicRegression) -> Dict[str, float]:
        """Calculate calibration metrics.
        
        Args:
            X: Input confidence scores
            y: True scores
            model: Calibration model
            
        Returns:
            Dictionary with calibration metrics
        """
        # Get calibrated predictions
        y_pred = model.predict(X)
        
        # Calculate Brier score
        brier_score = np.mean((y_pred - y) ** 2)
        
        # Calculate calibration error
        n_bins = 10
        bin_edges = np.linspace(0, 1, n_bins + 1)
        bin_indices = np.digitize(y_pred, bin_edges) - 1
        
        calibration_error = 0
        for i in range(n_bins):
            mask = bin_indices == i
            if np.any(mask):
                pred_mean = np.mean(y_pred[mask])
                actual_mean = np.mean(y[mask])
                calibration_error += abs(pred_mean - actual_mean)
        
        calibration_error /= n_bins
        
        # Calculate confidence correlation
        confidence_correlation = np.corrcoef(y_pred, y)[0, 1]
        
        return {
            'brier_score': brier_score,
            'calibration_error': calibration_error,
            'confidence_correlation': confidence_correlation
        }

    def _apply_calibration(self, confidence_scores: Dict[str, float]) -> Dict[str, float]:
        """Apply calibration to confidence scores.
        
        Args:
            confidence_scores: Dictionary of raw confidence scores
            
        Returns:
            Dictionary of calibrated confidence scores
        """
        if not self.is_calibrated:
            logger.warning("Confidence scores not calibrated. Using raw scores.")
            return confidence_scores
        
        calibrated_scores = {}
        
        # Calibrate each component
        for component, score in confidence_scores.items():
            if component in self.calibration_models:
                # Reshape for sklearn
                X = np.array([score]).reshape(-1, 1)
                
                # Apply calibration
                calibrated_scores[component] = float(self.calibration_models[component].predict(X)[0])
            else:
                calibrated_scores[component] = score
        
        # Calculate overall confidence
        if all(component in calibrated_scores for component in self.confidence_weights):
            overall_confidence = sum(
                calibrated_scores[component] * weight
                for component, weight in self.confidence_weights.items()
            )
            calibrated_scores['overall'] = overall_confidence
        
        return calibrated_scores

    def save_calibration(self, path: str) -> None:
        """Save calibration models and metrics to disk.
        
        Args:
            path: Path to save calibration data
        """
        if not self.is_calibrated:
            raise ValueError("No calibration data to save - run calibrate_confidence first")
        
        import joblib
        import json
        
        # Create directory if it doesn't exist
        Path(path).mkdir(parents=True, exist_ok=True)
        
        # Save calibration models
        for component, model in self.calibration_models.items():
            model_path = Path(path) / f"{component}_model.joblib"
            joblib.dump(model, model_path)
        
        # Save calibration metrics
        metrics_path = Path(path) / "calibration_metrics.json"
        with open(metrics_path, 'w') as f:
            json.dump(self.calibration_metrics, f, indent=2)
        
        # Save calibration state
        state_path = Path(path) / "calibration_state.json"
        with open(state_path, 'w') as f:
            json.dump({
                'is_calibrated': self.is_calibrated,
                'confidence_weights': self.confidence_weights
            }, f, indent=2)
        
        logger.info(f"Calibration data saved to {path}")

    def load_calibration(self, path: str) -> None:
        """Load calibration models and metrics from disk.
        
        Args:
            path: Path to load calibration data from
        """
        import joblib
        import json
        
        # Load calibration models
        for component in self.calibration_models:
            model_path = Path(path) / f"{component}_model.joblib"
            if model_path.exists():
                self.calibration_models[component] = joblib.load(model_path)
        
        # Load calibration metrics
        metrics_path = Path(path) / "calibration_metrics.json"
        if metrics_path.exists():
            with open(metrics_path, 'r') as f:
                self.calibration_metrics = json.load(f)
        
        # Load calibration state
        state_path = Path(path) / "calibration_state.json"
        if state_path.exists():
            with open(state_path, 'r') as f:
                state = json.load(f)
                self.is_calibrated = state['is_calibrated']
                self.confidence_weights = state['confidence_weights']
        
        logger.info(f"Calibration data loaded from {path}")

    def evaluate_confidence_vs_accuracy(self, gold_data: pd.DataFrame, 
                                      n_tiers: int = 5) -> Dict[str, Any]:
        """Evaluate how well confidence scores correlate with prediction accuracy.
        
        Args:
            gold_data: DataFrame with gold standard annotations
            n_tiers: Number of confidence tiers to evaluate
            
        Returns:
            Dictionary with evaluation metrics
        """
        logger.info("Starting confidence vs accuracy evaluation")
        
        # Initialize results storage
        tier_results = {
            'activation': {'tiers': [], 'accuracy': [], 'coverage': []},
            'emotional': {'tiers': [], 'accuracy': [], 'coverage': []},
            'metabolic': {'tiers': [], 'accuracy': [], 'coverage': []}
        }
        
        # Calculate tier boundaries
        tier_boundaries = np.linspace(0, 1, n_tiers + 1)
        
        # Process each dimension
        for dimension in ['activation', 'emotional', 'metabolic']:
            # Get predictions and gold scores
            pred_scores = gold_data[f'{dimension}_score'].values
            gold_scores = gold_data[f'{dimension}_gold'].values
            confidences = gold_data[f'{dimension}_confidence'].values
            
            # Calculate metrics for each tier
            for i in range(n_tiers):
                lower_bound = tier_boundaries[i]
                upper_bound = tier_boundaries[i + 1]
                
                # Get predictions in this confidence tier
                tier_mask = (confidences >= lower_bound) & (confidences < upper_bound)
                tier_preds = pred_scores[tier_mask]
                tier_gold = gold_scores[tier_mask]
                
                if len(tier_preds) > 0:
                    # Calculate accuracy (using RMSE for continuous scores)
                    accuracy = 1.0 - np.sqrt(np.mean((tier_preds - tier_gold) ** 2))
                    
                    # Calculate coverage
                    coverage = len(tier_preds) / len(pred_scores)
                    
                    # Store results
                    tier_results[dimension]['tiers'].append(f"{lower_bound:.2f}-{upper_bound:.2f}")
                    tier_results[dimension]['accuracy'].append(accuracy)
                    tier_results[dimension]['coverage'].append(coverage)
                else:
                    # No predictions in this tier
                    tier_results[dimension]['tiers'].append(f"{lower_bound:.2f}-{upper_bound:.2f}")
                    tier_results[dimension]['accuracy'].append(None)
                    tier_results[dimension]['coverage'].append(0.0)
        
        # Calculate overall metrics
        overall_metrics = self._calculate_overall_metrics(tier_results)
        
        # Add calibration plots
        calibration_plots = self._generate_calibration_plots(gold_data)
        
        return {
            'tier_results': tier_results,
            'overall_metrics': overall_metrics,
            'calibration_plots': calibration_plots
        }
    
    def _calculate_overall_metrics(self, tier_results: Dict[str, Any]) -> Dict[str, float]:
        """Calculate overall confidence calibration metrics.
        
        Args:
            tier_results: Results from tier-based evaluation
            
        Returns:
            Dictionary with overall metrics
        """
        metrics = {}
        
        for dimension in ['activation', 'emotional', 'metabolic']:
            # Get non-None accuracy values
            accuracies = [acc for acc in tier_results[dimension]['accuracy'] 
                        if acc is not None]
            coverages = tier_results[dimension]['coverage']
            
            if accuracies:
                # Calculate weighted average accuracy
                weighted_acc = np.average(accuracies, weights=coverages)
                
                # Calculate confidence-accuracy correlation
                tiers = np.array([float(t.split('-')[0]) for t in 
                                tier_results[dimension]['tiers']])
                valid_mask = ~np.isnan(accuracies)
                if np.sum(valid_mask) > 1:
                    correlation = np.corrcoef(tiers[valid_mask], 
                                            np.array(accuracies)[valid_mask])[0, 1]
                else:
                    correlation = None
                
                metrics[dimension] = {
                    'weighted_accuracy': weighted_acc,
                    'confidence_accuracy_correlation': correlation,
                    'total_coverage': sum(coverages)
                }
            else:
                metrics[dimension] = {
                    'weighted_accuracy': None,
                    'confidence_accuracy_correlation': None,
                    'total_coverage': 0.0
                }
        
        return metrics
    
    def _generate_calibration_plots(self, gold_data: pd.DataFrame) -> Dict[str, Any]:
        """Generate calibration plots for confidence evaluation.
        
        Args:
            gold_data: DataFrame with gold standard annotations
            
        Returns:
            Dictionary with plot data
        """
        plots = {}
        
        for dimension in ['activation', 'emotional', 'metabolic']:
            # Get predictions and gold scores
            pred_scores = gold_data[f'{dimension}_score'].values
            gold_scores = gold_data[f'{dimension}_gold'].values
            confidences = gold_data[f'{dimension}_confidence'].values
            
            # Calculate calibration curve
            n_bins = 10
            bin_edges = np.linspace(0, 1, n_bins + 1)
            bin_indices = np.digitize(confidences, bin_edges) - 1
            
            calibration_curve = []
            for i in range(n_bins):
                mask = bin_indices == i
                if np.any(mask):
                    pred_mean = np.mean(pred_scores[mask])
                    actual_mean = np.mean(gold_scores[mask])
                    confidence_mean = np.mean(confidences[mask])
                    calibration_curve.append({
                        'confidence': confidence_mean,
                        'predicted': pred_mean,
                        'actual': actual_mean,
                        'count': np.sum(mask)
                    })
            
            plots[dimension] = {
                'calibration_curve': calibration_curve,
                'perfect_calibration': [{'x': 0, 'y': 0}, {'x': 1, 'y': 1}]
            }
        
        return plots
    
    def validate_confidence_tiers(self, gold_data: pd.DataFrame, 
                                min_confidence: float = 0.7) -> Dict[str, Any]:
        """Validate that higher confidence predictions have better accuracy.
        
        Args:
            gold_data: DataFrame with gold standard annotations
            min_confidence: Minimum confidence threshold for high-confidence tier
            
        Returns:
            Dictionary with validation results
        """
        logger.info("Starting confidence tier validation")
        
        results = {}
        
        for dimension in ['activation', 'emotional', 'metabolic']:
            # Get predictions and gold scores
            pred_scores = gold_data[f'{dimension}_score'].values
            gold_scores = gold_data[f'{dimension}_gold'].values
            confidences = gold_data[f'{dimension}_confidence'].values
            
            # Split into high and low confidence
            high_conf_mask = confidences >= min_confidence
            low_conf_mask = ~high_conf_mask
            
            # Calculate accuracy for each tier
            high_conf_acc = 1.0 - np.sqrt(np.mean(
                (pred_scores[high_conf_mask] - gold_scores[high_conf_mask]) ** 2
            )) if np.any(high_conf_mask) else None
            
            low_conf_acc = 1.0 - np.sqrt(np.mean(
                (pred_scores[low_conf_mask] - gold_scores[low_conf_mask]) ** 2
            )) if np.any(low_conf_mask) else None
            
            # Calculate coverage
            high_conf_coverage = np.mean(high_conf_mask)
            low_conf_coverage = np.mean(low_conf_mask)
            
            results[dimension] = {
                'high_confidence': {
                    'accuracy': high_conf_acc,
                    'coverage': high_conf_coverage,
                    'count': np.sum(high_conf_mask)
                },
                'low_confidence': {
                    'accuracy': low_conf_acc,
                    'coverage': low_conf_coverage,
                    'count': np.sum(low_conf_mask)
                }
            }
            
            # Log results
            logger.info(f"{dimension} dimension validation:")
            logger.info(f"  High confidence (≥{min_confidence}):")
            logger.info(f"    Accuracy: {high_conf_acc:.3f}")
            logger.info(f"    Coverage: {high_conf_coverage:.1%}")
            logger.info(f"  Low confidence (<{min_confidence}):")
            logger.info(f"    Accuracy: {low_conf_acc:.3f}")
            logger.info(f"    Coverage: {low_conf_coverage:.1%}")
        
        return results
"""
Medication Temporal Graph Construction Module.
(Initialization is robust, focus on NLP pattern effectiveness)
"""

import spacy
import networkx as nx
from spacy.matcher import Matcher, PhraseMatcher
from spacy.pipeline import EntityRuler
from spacy.tokens import Doc, Span
import pandas as pd
from typing import Dict, List, Tuple, Set, Any, Optional
import re
import json
from pathlib import Path
import logging
from logging.handlers import RotatingFileHandler
from datetime import datetime
import ast
import numpy as np
import sys
import threading

# Configure logging with thread safety
_logger_lock = threading.Lock()
logger = logging.getLogger("med_temporal")

def _setup_logger():
    """Set up logger with thread-safe configuration."""
    with _logger_lock:
        if not logger.handlers:
            logger.setLevel(logging.DEBUG)
            
            # Console handler
            console_handler = logging.StreamHandler(sys.stdout)
            formatter = logging.Formatter(
                '%(asctime)s - %(name)s - %(levelname)s - %(module)s:%(funcName)s:%(lineno)d - %(message)s'
            )
            console_handler.setFormatter(formatter)
            logger.addHandler(console_handler)
            
            # File handler with rotation
            try:
                file_handler = RotatingFileHandler(
                    "med_temporal.log",
                    maxBytes=10*1024*1024,  # 10MB
                    backupCount=5,
                    encoding='utf-8'
                )
                file_handler.setFormatter(formatter)
                logger.addHandler(file_handler)
            except Exception as e:
                logger.warning(f"Could not set up file handler: {e}")

_setup_logger()

def fix_medications(med_value: Any) -> List[str]:
    """Fix and standardize medication value formats."""
    if isinstance(med_value, np.ndarray):
        return [str(item).strip() for item in med_value]
    if isinstance(med_value, str):
        if med_value.startswith('[') and med_value.endswith(']'):
            try:
                parsed_list = ast.literal_eval(med_value)
                if isinstance(parsed_list, list):
                    return [str(m).strip() for m in parsed_list]
            except (ValueError, SyntaxError):
                clean_meds = med_value.strip('[]')
                return [m.strip().strip("'").strip('"').strip() for m in clean_meds.split(',') if m.strip()]
        try:
            parsed_json = json.loads(med_value)
            if isinstance(parsed_json, list):
                return [str(m).strip() for m in parsed_json]
        except json.JSONDecodeError:
            return [m.strip() for m in med_value.split(',') if m.strip()]
    elif isinstance(med_value, list):
        return [str(m).strip() for m in med_value]
    elif pd.isna(med_value) or med_value is None or (hasattr(med_value, '__len__') and len(med_value) == 0) :
        return []
    else:
        try:
            return [str(med_value).strip()]
        except:
            logger.warning(f"fix_medications: Unexpected type for med_value: {type(med_value)}. Value: '{med_value}'. Returning empty list.")
            return []

class MedicationTemporalGraph:
    """Base class to create and analyze a graph of medication-effect-time relationships."""
    
    def __init__(self, model_name: str = "en_core_web_sm"):
        self.nlp = None
        self.matcher = None
        self.phrase_matcher = None
        self.temporal_verbs = {}
        self.time_markers = {}
        self.confidence_weights = {}
        
        logger.info(f"Base Init: Initializing MedicationTemporalGraph with model: {model_name}")
        try:
            self.nlp = spacy.load(model_name)
            logger.info(f"Base Init: Successfully loaded spaCy model '{model_name}' ({self.nlp.meta.get('lang', 'N/A')}_{self.nlp.meta.get('name', 'N/A')})")
        except Exception as e:
            logger.error(f"Base Init: CRITICAL - Failed to load spaCy model '{model_name}': {e}", exc_info=True)
            logger.warning("Base Init: Exiting __init__ early. self.nlp is None.")
            return 

        # Attempt to initialize Matcher
        try:
            logger.debug(f"Base Init: Attempting to create Matcher. self.nlp.vocab exists: {hasattr(self.nlp, 'vocab')}")
            self.matcher = Matcher(self.nlp.vocab)
            logger.info(f"Base Init: Matcher object CREATED. Type: {type(self.matcher)}. Length: {len(self.matcher)}")
            self._add_temporal_patterns()
            self._add_effect_patterns()
            final_matcher_len = len(self.matcher) if self.matcher is not None else -1
            logger.info(f"Base Init: After adding patterns, self.matcher length: {final_matcher_len}")
            if final_matcher_len == 0 and self.matcher is not None:
                 logger.warning("Base Init: self.matcher is an object but has 0 patterns. THIS WILL CAUSE W036 if used without patterns.")
        except Exception as e:
            logger.error(f"Base Init: CRITICAL - Error during Matcher creation or pattern addition: {e}", exc_info=True)
            self.matcher = None 

        # Attempt to initialize PhraseMatcher
        try:
            logger.debug(f"Base Init: Attempting to create PhraseMatcher. self.nlp.vocab exists: {hasattr(self.nlp, 'vocab')}")
            self.phrase_matcher = PhraseMatcher(self.nlp.vocab)
            logger.info(f"Base Init: PhraseMatcher object CREATED. Type: {type(self.phrase_matcher)}. Length: {len(self.phrase_matcher)}")
        except Exception as e:
            logger.error(f"Base Init: CRITICAL - Error during PhraseMatcher creation: {e}", exc_info=True)
            self.phrase_matcher = None 
        
        self.temporal_verbs = {
            'start': 'begin', 'begin': 'begin', 'commence': 'begin', 'started': 'begin', 'starting': 'begin',
            'stop': 'end', 'quit': 'end', 'discontinue': 'end', 'end': 'end', 'stopped': 'end', 'quitting': 'end',
            'take': 'ongoing', 'taking': 'ongoing', 'use': 'ongoing', 'using': 'ongoing',
            'try': 'tentative', 'tried': 'tentative', 'trying': 'tentative', 'attempt': 'tentative'
        }
        self.time_markers = {
            'past': ['yesterday', 'last week', 'last month', 'ago', 'previously', 'before', 'used to', 'had been'],
            'present': ['now', 'currently', 'today', 'present', 'am', 'is', 'are', 'have been', 'has been'],
            'future': ['tomorrow', 'next week', 'will', 'going to', 'about to', 'plan to', 'intend to']
        }
        self.confidence_weights = {
            'explicit_temporal': 0.4, 'direct_causality': 0.3,
            'proximity': 0.15, 'tense_consistency': 0.15
        }
        logger.debug("Base Init: MedicationTemporalGraph base initialization completed.")

    def _add_temporal_patterns(self) -> None:
        logger.debug(f"AddTemporalPatterns: Entered. self.matcher type: {type(self.matcher)}, Is None: {self.matcher is None}")
        if not isinstance(self.matcher, Matcher): 
            logger.warning(f"AddTemporalPatterns: self.matcher is not a valid Matcher object (type: {type(self.matcher)}). Cannot add temporal patterns.")
            return

        # Enhanced patterns with more variations and flexibility
        current_patterns = [
            # I am/I'm currently taking/using
            [{"LOWER": {"IN": ["i", "i'm", "im"]}}, {"LOWER": {"IN": ["currently", "now", "presently"]}, "OP": "?"}, 
             {"LOWER": {"IN": ["taking", "on", "using"]}}, {"OP": "*", "IS_PUNCT": False, "IS_SPACE": False}, 
             {"ENT_TYPE": "MEDICATION"}],
            
            # I've been on/taking/using
            [{"LOWER": {"IN": ["i", "i've", "ive"]}}, {"LOWER": {"IN": ["been"]}}, 
             {"LOWER": {"IN": ["on", "taking", "using"]}}, {"OP": "*", "IS_PUNCT": False, "IS_SPACE": False}, 
             {"ENT_TYPE": "MEDICATION"}],
            
            # Started [med] recently/today/yesterday/ago
            [{"LOWER": "started"}, {"OP": "*", "IS_PUNCT": False, "IS_SPACE": False}, 
             {"ENT_TYPE": "MEDICATION"}, {"OP": "*", "IS_PUNCT": False, "IS_SPACE": False}, 
             {"LOWER": {"IN": ["recently", "today", "yesterday", "ago", "days", "weeks", "months"]}}],
            
            # Been prescribed
            [{"LOWER": {"IN": ["was", "been"]}}, {"LOWER": "prescribed"}, 
             {"OP": "*", "IS_PUNCT": False, "IS_SPACE": False}, {"ENT_TYPE": "MEDICATION"}],
            
            # Currently prescribed
            [{"LOWER": {"IN": ["am", "is", "are"]}}, {"LOWER": "prescribed"}, 
             {"OP": "*", "IS_PUNCT": False, "IS_SPACE": False}, {"ENT_TYPE": "MEDICATION"}],
            
            # Taking [med] for [time]
            [{"LOWER": {"IN": ["taking", "using"]}}, {"OP": "*", "IS_PUNCT": False, "IS_SPACE": False}, 
             {"ENT_TYPE": "MEDICATION"}, {"LOWER": "for"}, {"OP": "*", "IS_PUNCT": False, "IS_SPACE": False}, 
             {"LOWER": {"IN": ["days", "weeks", "months", "years"]}}]
        ]

        past_patterns = [
            # I've stopped/discontinued/quit
            [{"LOWER": {"IN": ["i", "i've", "ive"]}}, 
             {"LOWER": {"IN": ["stopped", "discontinued", "quit", "dropped", "got off"]}}, 
             {"OP": "*", "IS_PUNCT": False, "IS_SPACE": False}, {"ENT_TYPE": "MEDICATION"}],
            
            # I tried/was on/took/used
            [{"LOWER": {"IN": ["i", "i've", "ive", "i'd", "id"]}}, 
             {"LOWER": {"IN": ["tried", "was on", "took", "used"]}}, 
             {"OP": "*", "IS_PUNCT": False, "IS_SPACE": False}, {"ENT_TYPE": "MEDICATION"}],
            
            # [med] didn't work/help
            [{"ENT_TYPE": "MEDICATION"}, {"OP": "*", "IS_PUNCT": False, "IS_SPACE": False}, 
             {"LOWER": {"IN": ["didn't", "didnt", "did not", "doesn't", "doesnt", "does not", "wasn't", "wasnt", "never"]}}, 
             {"LOWER": {"IN": ["work", "help", "suit me"]}}],
            
            # Stopped taking
            [{"LOWER": {"IN": ["stopped", "quit"]}}, {"LOWER": {"IN": ["taking", "using"]}}, 
             {"OP": "*", "IS_PUNCT": False, "IS_SPACE": False}, {"ENT_TYPE": "MEDICATION"}],
            
            # Used to take/be on
            [{"LOWER": "used"}, {"LOWER": "to"}, {"LOWER": {"IN": ["take", "be on"]}}, 
             {"OP": "*", "IS_PUNCT": False, "IS_SPACE": False}, {"ENT_TYPE": "MEDICATION"}]
        ]

        try:
            logger.debug(f"AddTemporalPatterns: About to add CURRENT_MED. Patterns count: {len(current_patterns)}")
            self.matcher.add("CURRENT_MED", current_patterns)
            logger.debug(f"AddTemporalPatterns: About to add PAST_MED. Patterns count: {len(past_patterns)}")
            self.matcher.add("PAST_MED", past_patterns)
            logger.info(f"AddTemporalPatterns: Successfully added patterns. Current matcher length: {len(self.matcher)}")
            
            # Log pattern details for debugging
            for pattern_list in current_patterns:
                pattern_str = " ".join(str(token) for token in pattern_list)
                logger.debug(f"AddTemporalPatterns: Added CURRENT_MED pattern: {pattern_str}")
            for pattern_list in past_patterns:
                pattern_str = " ".join(str(token) for token in pattern_list)
                logger.debug(f"AddTemporalPatterns: Added PAST_MED pattern: {pattern_str}")
            
        except Exception as e:
            logger.error(f"AddTemporalPatterns: CRITICAL - Error adding temporal patterns to matcher: {e}", exc_info=True)

    def _add_effect_patterns(self) -> None:
        logger.debug(f"AddEffectPatterns: Entered. self.matcher type: {type(self.matcher)}, Is None: {self.matcher is None}")
        if not isinstance(self.matcher, Matcher): 
            logger.warning(f"AddEffectPatterns: self.matcher is not a valid Matcher object (type: {type(self.matcher)}). Cannot add effect patterns.")
            return

        # Enhanced patterns with more variations
        direct_patterns = [
            # [med] makes/causes me feel
            [{"ENT_TYPE": "MEDICATION"}, {"OP": "*", "IS_PUNCT": False, "IS_SPACE": False}, 
             {"LOWER": {"IN": ["makes", "made", "causes", "caused", "is making", "was making", "gives", "gave"]}}, 
             {"LOWER": "me", "OP": "?"}, {"OP": "*", "IS_PUNCT": False, "IS_SPACE": False}, 
             {"POS": {"IN": ["ADJ", "VERB", "NOUN"]}}],
            
            # Because of/due to [med]
            [{"LOWER": {"IN": ["because of", "due to", "thanks to"]}}, 
             {"OP": "*", "IS_PUNCT": False, "IS_SPACE": False}, {"ENT_TYPE": "MEDICATION"}, 
             {"OP": "*", "IS_PUNCT": False, "IS_SPACE": False}, 
             {"LOWER": {"IN": ["i", "i'm", "im", "i've", "ive", "i'd", "id"]}}, 
             {"LOWER": {"IN": ["feel", "felt", "am", "feeling", "experiencing", "experienced", "have", "had", "got", "get"]}}],
            
            # After/since taking [med]
            [{"LOWER": {"IN": ["after", "since", "when"]}}, 
             {"LOWER": {"IN": ["starting", "taking", "using", "i started", "i took"]}}, 
             {"OP": "*", "IS_PUNCT": False, "IS_SPACE": False}, {"ENT_TYPE": "MEDICATION"}, 
             {"OP": "*", "IS_PUNCT": False, "IS_SPACE": False}, 
             {"LOWER": {"IN": ["i", "i'm", "im", "i've", "ive"]}}, 
             {"LOWER": {"IN": ["feel", "felt", "noticed", "notice", "started", "began", "begun", "got"]}}],
            
            # [med] helped with/improved
            [{"ENT_TYPE": "MEDICATION"}, {"OP": "*", "IS_PUNCT": False, "IS_SPACE": False}, 
             {"LOWER": {"IN": ["helped", "helps", "improved", "improves", "fixed", "fixes"]}}, 
             {"OP": "*", "IS_PUNCT": False, "IS_SPACE": False}, {"POS": {"IN": ["ADJ", "NOUN"]}}]
        ]

        side_effect_patterns = [
            # Standard side effect mentions
            [{"ENT_TYPE": "MEDICATION"}, {"OP": "*", "IS_PUNCT": False, "IS_SPACE": False}, 
             {"LOWER": {"IN": ["side", "adverse"]}}, {"LOWER": {"IN": ["effect", "effects", "reaction", "reactions"]}}],
            
            # Side effects of [med]
            [{"LOWER": {"IN": ["side", "adverse"]}}, {"LOWER": {"IN": ["effect", "effects", "reaction", "reactions"]}}, 
             {"LOWER": {"IN": ["of", "from"]}}, {"OP": "*", "IS_PUNCT": False, "IS_SPACE": False}, 
             {"ENT_TYPE": "MEDICATION"}],
            
            # [med] causes negative effects
            [{"ENT_TYPE": "MEDICATION"}, {"LOWER": {"IN": ["has", "gives", "gave", "causes", "caused"]}}, 
             {"LOWER": "me", "OP": "?"}, {"POS": {"IN": ["VERB", "ADJ", "NOUN"]}}],
            
            # Feeling [effect] from/on [med]
            [{"LOWER": {"IN": ["feeling", "feel"]}}, {"POS": "ADJ"}, 
             {"LOWER": {"IN": ["from", "on", "with", "after taking"]}}, {"ENT_TYPE": "MEDICATION"}],
            
            # [med] is making me feel
            [{"ENT_TYPE": "MEDICATION"}, {"LOWER": "is"}, {"LOWER": "making"}, 
             {"LOWER": "me", "OP": "?"}, {"LOWER": "feel"}, {"POS": "ADJ"}],
            
            # Experienced [effect] while on [med]
            [{"LOWER": {"IN": ["experienced", "noticed", "developed"]}}, 
             {"OP": "*", "IS_PUNCT": False, "IS_SPACE": False}, 
             {"LOWER": {"IN": ["while", "when", "after"]}}, 
             {"LOWER": {"IN": ["on", "taking"]}}, {"ENT_TYPE": "MEDICATION"}]
        ]

        try:
            logger.debug(f"AddEffectPatterns: About to add DIRECT_EFFECT. Patterns count: {len(direct_patterns)}")
            self.matcher.add("DIRECT_EFFECT", direct_patterns)
            logger.debug(f"AddEffectPatterns: About to add SIDE_EFFECT. Patterns count: {len(side_effect_patterns)}")
            self.matcher.add("SIDE_EFFECT", side_effect_patterns)
            logger.info(f"AddEffectPatterns: Successfully added patterns. Current matcher length: {len(self.matcher)}")
            
            # Log pattern details for debugging
            for pattern_list in direct_patterns:
                pattern_str = " ".join(str(token) for token in pattern_list)
                logger.debug(f"AddEffectPatterns: Added DIRECT_EFFECT pattern: {pattern_str}")
            for pattern_list in side_effect_patterns:
                pattern_str = " ".join(str(token) for token in pattern_list)
                logger.debug(f"AddEffectPatterns: Added SIDE_EFFECT pattern: {pattern_str}")
            
        except Exception as e:
            logger.error(f"AddEffectPatterns: CRITICAL - Error adding effect patterns to matcher: {e}", exc_info=True)

    def _register_medications(self, medications: List[str]) -> None:
        logger.debug("Base _register_medications called.")
        if not self.nlp:
            logger.warning("Base _register_medications: NLP model not available.")
            return
        
        if not isinstance(self.phrase_matcher, PhraseMatcher):
            logger.warning(f"Base _register_medications: self.phrase_matcher is not a valid PhraseMatcher (type: {type(self.phrase_matcher)}). Attempting to create.")
            try:
                self.phrase_matcher = PhraseMatcher(self.nlp.vocab)
                logger.info("Base _register_medications: PhraseMatcher re-created successfully.")
            except Exception as e:
                logger.error(f"Base _register_medications: CRITICAL - Failed to create PhraseMatcher: {e}", exc_info=True)
                self.phrase_matcher = None 
                return 

        if self.phrase_matcher:
            self.phrase_matcher = PhraseMatcher(self.nlp.vocab) 
            valid_meds = [med.strip().lower() for med in medications if med and isinstance(med, str) and med.strip()]
            phrase_patterns_docs = [self.nlp.make_doc(med) for med in valid_meds] 
            if phrase_patterns_docs:
                self.phrase_matcher.add("MEDICATION_PM_BASE", phrase_patterns_docs) 
                logger.debug(f"Base _register_medications: Added {len(phrase_patterns_docs)} patterns to PhraseMatcher. Length: {len(self.phrase_matcher)}")
            else:
                logger.debug("Base _register_medications: No valid medication strings for PhraseMatcher.")
        else:
            logger.warning("Base _register_medications: PhraseMatcher is None after creation attempt. Cannot add patterns.")


    def _build_graph(self, doc: Doc, medications: List[str]) -> nx.DiGraph:
        logger.debug("Base _build_graph called (minimal implementation).")
        G = nx.DiGraph()
        if not self.nlp: 
            logger.warning("Base _build_graph: NLP not available.")
            return G
        
        if not isinstance(self.phrase_matcher, PhraseMatcher):
            logger.warning(f"Base _build_graph: self.phrase_matcher is not valid (type: {type(self.phrase_matcher)}).")
            if self.phrase_matcher is None and self.nlp:
                try:
                    self.phrase_matcher = PhraseMatcher(self.nlp.vocab)
                    logger.info("Base _build_graph: PhraseMatcher created on-the-fly.")
                except Exception as e:
                    logger.error(f"Base _build_graph: CRITICAL - Failed to create PhraseMatcher on-the-fly: {e}", exc_info=True)
                    return G 
            else: 
                return G

        self._register_medications(medications) 
        
        medication_matches = []
        if self.phrase_matcher and len(self.phrase_matcher) > 0: # Ensure it has patterns
             medication_matches = self.phrase_matcher(doc)
        elif self.phrase_matcher is None:
            logger.warning("Base _build_graph: PhraseMatcher is None, cannot get matches.")
        elif len(self.phrase_matcher) == 0:
            logger.debug("Base _build_graph: PhraseMatcher is empty, no medication_matches will be found.")
            
        for match_id, start, end in medication_matches:
            span = doc[start:end]
            normalized_med = span.text.lower() 
            if not G.has_node(normalized_med): 
                G.add_node(normalized_med, type="medication")
        logger.debug(f"Base _build_graph created graph with {len(G.nodes())} medication nodes (from PhraseMatcher).")
        return G
    
    def _extract_tense(self, span: Span) -> str:
        text_lower = span.text.lower()
        if any(marker in text_lower for marker in self.time_markers.get('future', [])): return "future"
        if any(marker in text_lower for marker in self.time_markers.get('present', [])): return "present"
        if any(marker in text_lower for marker in self.time_markers.get('past', [])): return "past"
        root = span.root
        for token in root.subtree:
            if token.pos_ == "VERB":
                if token.tag_ in ["VBP", "VBZ"]: return "present" 
                if token.tag_ == "VBG": 
                    for child in token.children:
                        if child.dep_ == "aux":
                            if child.lemma_ == "be" and child.tag_ in ["VBP", "VBZ", "VB"]: return "present" 
                            if child.lemma_ == "be" and child.tag_ == "VBD": return "past" 
                            if child.lemma_ == "have" and child.tag_ in ["VBP", "VBZ"]: return "present" 
                            if child.lemma_ == "have" and child.tag_ == "VBD": return "past" 
                    if token.head.lemma_ == "be" and token.head.tag_ in ["VBP", "VBZ"]: return "present"
                    if token.head.lemma_ == "be" and token.head.tag_ == "VBD": return "past"
                    if token.dep_ == "ROOT" or token.dep_ == "ccomp" or token.dep_ == "xcomp": return "present"
                if token.tag_ == "VBD": return "past"    
                if token.tag_ == "VBN": 
                     for child in token.children:
                        if child.dep_ == "aux":
                            if child.lemma_ == "have" and child.tag_ in ["VBP", "VBZ"]: return "present" 
                            if child.lemma_ == "have" and child.tag_ == "VBD": return "past"   
                            if child.lemma_ == "be" and child.tag_ in ["VBP", "VBZ", "VB"]: return "present" 
                            if child.lemma_ == "be" and child.tag_ == "VBD": return "past" 
                     if token.dep_ == "ROOT" and not any(c.dep_ == "aux" for c in token.children): return "past"
            if token.tag_ == "MD": 
                if token.lemma_ in ["will", "shall", "'ll"]: return "future"
                if token.lemma_ in ["would", "could", "might", "should"]: return "past" 
        if span.root.pos_ == "VERB":
            if span.root.tag_ in ["VBP", "VBZ", "VBG"]: return "present"
            if span.root.tag_ == "VBD": return "past"
        return "unknown"
    
    def _extract_verb_category(self, verb_lemma: Optional[str]) -> str:
        if verb_lemma is None: return "unknown"
        return self.temporal_verbs.get(verb_lemma.lower(), "other")
    
    def _calculate_confidence(self, span: Span, pattern_name: str, 
                            verb_category: str, tense: str) -> float:
        confidence = 0.3 
        if pattern_name in ["CURRENT_MED", "PAST_MED"]:
            confidence += self.confidence_weights.get('explicit_temporal', 0.3)
            if verb_category not in ["unknown", "other"]: 
                confidence += 0.15
        if pattern_name in ["DIRECT_EFFECT", "SIDE_EFFECT"]:
            confidence += self.confidence_weights.get('direct_causality', 0.25)
        if tense != "unknown":
            confidence += self.confidence_weights.get('tense_consistency', 0.1)
        return min(confidence, 1.0)
    
    def process_text(self, text: str, medications: List[str]) -> Dict[str, Any]:
        logger.debug(f"Base process_text called for text: '{text[:50]}...' and meds: {medications}")
        if not self.nlp:
            logger.error("Base process_text: NLP model not loaded. Returning empty results.")
            return {"graph": nx.DiGraph(), "temporal_status": {}, "confidence_scores": {}, "effect_attributions": {}}
        doc = self.nlp(text)
        graph = self._build_graph(doc, medications) 
        temporal_status_results = {med: "unknown" for med in medications}
        confidence_scores_results = {med: 0.0 for med in medications}
        effect_attributions_results = {med: [] for med in medications}
        if graph.nodes:
            for med_name in medications:
                if graph.has_node(med_name):
                    confidence_scores_results[med_name] = 0.1 
        return {
            "graph": graph, "temporal_status": temporal_status_results,
            "confidence_scores": confidence_scores_results, "effect_attributions": effect_attributions_results
        }

    def process_dataframe(self, df: pd.DataFrame, text_col: str, med_col: str) -> pd.DataFrame:
        logger.info(f"Base process_dataframe called for DataFrame with text_col='{text_col}', med_col='{med_col}'")
        if not self.nlp:
            logger.error("Base process_dataframe: NLP model not available. Returning df with empty result columns.")
            df['med_temporal_status'] = None; df['med_confidence'] = None; df['med_attributions'] = None
            df['overall_confidence'] = 0.0; df['medications_fixed'] = df[med_col].apply(fix_medications)
            return df

        result_df = df.copy()
        result_df['medications_fixed'] = result_df[med_col].apply(fix_medications)
        results_list = []
        for idx, row in result_df.iterrows(): 
            text = row[text_col] if pd.notna(row[text_col]) and isinstance(row[text_col], str) else ""
            meds_list = row['medications_fixed']
            if not isinstance(meds_list, list): meds_list = []
            try:
                processed_row_data = self.process_text(text, meds_list)
                results_list.append(processed_row_data)
            except Exception as e:
                logger.error(f"Base process_dataframe: Error processing row {idx}: {e}", exc_info=True)
                results_list.append({
                    'med_temporal_status': {m: "error" for m in meds_list}, 
                    'med_confidence': {m: 0.0 for m in meds_list}, 
                    'med_attributions': {m: [] for m in meds_list},
                    'overall_confidence': 0.0
                })
        result_df['med_temporal_status'] = [r['temporal_status'] for r in results_list]
        result_df['med_confidence'] = [r['confidence_scores'] for r in results_list]
        result_df['med_attributions'] = [r['effect_attributions'] for r in results_list]
        overall_confidences = []
        for r in results_list:
            conf_scores = [s for s in r.get('confidence_scores', {}).values() if isinstance(s, (int, float))]
            overall_confidences.append(sum(conf_scores) / len(conf_scores) if conf_scores else 0.0)
        result_df['overall_confidence'] = overall_confidences
        logger.info("Base process_dataframe: Processing complete.")
        return result_df


class FixedMedicationTemporalGraph(MedicationTemporalGraph):
    """Enhanced version using EntityRuler."""
    
    def __init__(self, model_name: str = "en_core_web_sm"):
        logger.info(f"Fixed Init: Initializing FixedMedicationTemporalGraph with model: {model_name}")
        super().__init__(model_name) 
        
        if self.nlp is not None: 
            try:
                if "entity_ruler" not in self.nlp.pipe_names:
                    self.nlp.add_pipe("entity_ruler", config={"overwrite_ents": True})
                    logger.info("Fixed Init: Added EntityRuler to spaCy pipeline.")
                else:
                    logger.info("Fixed Init: EntityRuler already exists in pipeline.")
            except Exception as e:
                logger.error(f"Fixed Init: Failed during EntityRuler setup: {e}", exc_info=True)

            # "Second chance" for self.matcher
            if self.matcher is None: 
                logger.warning("Fixed Init: self.matcher is None after base init. Attempting reinitialization.")
                try:
                    logger.debug(f"Fixed Init (Second Chance Matcher): Attempting to create Matcher. self.nlp.vocab exists: {hasattr(self.nlp, 'vocab')}")
                    self.matcher = Matcher(self.nlp.vocab) 
                    logger.info(f"Fixed Init (Second Chance Matcher): Successfully reinitialized self.matcher. Type: {type(self.matcher)}")
                    
                    logger.debug(f"Fixed Init (Second Chance Matcher): BEFORE _add_temporal_patterns. Type: {type(self.matcher)}, Is None: {self.matcher is None}")
                    self._add_temporal_patterns() 
                    logger.debug(f"Fixed Init (Second Chance Matcher): AFTER _add_temporal_patterns. Type: {type(self.matcher)}, Is None: {self.matcher is None}, Length: {len(self.matcher) if self.matcher else -1}")

                    logger.debug(f"Fixed Init (Second Chance Matcher): BEFORE _add_effect_patterns. Type: {type(self.matcher)}, Is None: {self.matcher is None}")
                    self._add_effect_patterns()   
                    logger.debug(f"Fixed Init (Second Chance Matcher): AFTER _add_effect_patterns. Type: {type(self.matcher)}, Is None: {self.matcher is None}, Length: {len(self.matcher) if self.matcher else -1}")
                    
                    re_added_pattern_count = len(self.matcher) if self.matcher is not None else -1
                    logger.info(f"Fixed Init (Second Chance Matcher): Matcher length after re-adding patterns: {re_added_pattern_count}")
                    if re_added_pattern_count == 0 and self.matcher is not None:
                        logger.warning("Fixed Init (Second Chance Matcher): Matcher is empty even after re-adding patterns.")
                except Exception as e:
                    logger.error(f"Fixed Init (Second Chance Matcher): CRITICAL - Error during matcher re-initialization or re-adding patterns: {e}", exc_info=True)
                    self.matcher = None 

            # "Second chance" for self.phrase_matcher
            if self.phrase_matcher is None and self.nlp is not None: 
                logger.warning("Fixed Init: self.phrase_matcher is None after base init. Attempting reinitialization.")
                try:
                    logger.debug(f"Fixed Init (Second Chance PhraseMatcher): Attempting to create PhraseMatcher. self.nlp.vocab exists: {hasattr(self.nlp, 'vocab')}")
                    self.phrase_matcher = PhraseMatcher(self.nlp.vocab)
                    logger.info(f"Fixed Init (Second Chance PhraseMatcher): Successfully reinitialized self.phrase_matcher. Type: {type(self.phrase_matcher)}")
                except Exception as e:
                    logger.error(f"Fixed Init (Second Chance PhraseMatcher): CRITICAL - Error reinitializing self.phrase_matcher: {e}", exc_info=True)
                    self.phrase_matcher = None 
        else: 
            logger.error("Fixed Init: NLP model is None (failed in base class). Cannot perform subclass-specific NLP setup.")

        logger.debug(f"Fixed Init: FixedMedicationTemporalGraph initialization complete. Matcher type: {type(self.matcher)}, PhraseMatcher type: {type(self.phrase_matcher)}")
        if self.matcher is not None:
            logger.debug(f"Fixed Init: Final matcher pattern count: {len(self.matcher)}")
        else:
            logger.warning("Fixed Init: Final self.matcher is None.")
        if self.phrase_matcher is not None:
            logger.debug(f"Fixed Init: Final PhraseMatcher pattern count: {len(self.phrase_matcher)}")
        else:
            logger.warning("Fixed Init: Final self.phrase_matcher is None.")


    def _register_medications(self, medications: List[str]) -> None:
        """Register medications for entity recognition using EntityRuler and PhraseMatcher."""
        if not medications: 
            logger.debug("RegisterMeds (Fixed): No medications provided.")
            return
        if not self.nlp:
            logger.warning("RegisterMeds (Fixed): NLP model not available.")
            return
            
        # --- EntityRuler Setup ---
        entity_ruler_patterns = []
        for med_raw in medications:
            if not med_raw or not isinstance(med_raw, str) or not med_raw.strip():
                continue
            med_original_case = med_raw.strip() 
            med_lower = med_original_case.lower()
            pattern_id = med_original_case 
            
            if " " in med_lower: 
                words = med_lower.split()
                entity_ruler_patterns.append({"label": "MEDICATION", "pattern": [{"LOWER": word} for word in words], "id": pattern_id})
            else: 
                entity_ruler_patterns.append({"label": "MEDICATION", "pattern": [{"LOWER": med_lower}], "id": pattern_id})
        
        if entity_ruler_patterns:
            try:
                if "entity_ruler" in self.nlp.pipe_names:
                    ruler = self.nlp.get_pipe("entity_ruler")
                    ruler.add_patterns(entity_ruler_patterns) 
                    logger.info(f"RegisterMeds (Fixed): Added {len(entity_ruler_patterns)} patterns to EntityRuler. Total patterns in ruler now: {len(ruler)}")
                else:
                    logger.warning("RegisterMeds (Fixed): EntityRuler component not found. Cannot add patterns.")
            except Exception as e:
                logger.error(f"RegisterMeds (Fixed): Error during EntityRuler pattern registration: {e}", exc_info=True)
        else:
            logger.debug("RegisterMeds (Fixed): No valid patterns generated to add to EntityRuler.")

        # --- PhraseMatcher Setup (always create a fresh one for this call) ---
        # This ensures PhraseMatcher only has patterns for the current 'medications' list for this specific process_text call.
        current_call_phrase_matcher = None # Use a local variable
        try:
            logger.debug("RegisterMeds (Fixed): Attempting to create/re-create PhraseMatcher for current call.")
            current_call_phrase_matcher = PhraseMatcher(self.nlp.vocab) # Create a new, fresh instance
            
            valid_meds_for_pm = [med.strip().lower() for med in medications if med and isinstance(med, str) and med.strip()]
            phrase_patterns_docs = [self.nlp.make_doc(med) for med in valid_meds_for_pm]
            
            if phrase_patterns_docs:
                current_call_phrase_matcher.add("MEDICATION_PM", phrase_patterns_docs) 
                logger.debug(f"RegisterMeds (Fixed): Added {len(phrase_patterns_docs)} patterns to fresh PhraseMatcher for this call. Length: {len(current_call_phrase_matcher)}")
            else:
                logger.debug("RegisterMeds (Fixed): No valid medication strings for PhraseMatcher patterns for this call.")
            
            # Assign to self.phrase_matcher AFTER it's successfully created and populated for this call
            self.phrase_matcher = current_call_phrase_matcher 
            logger.info(f"RegisterMeds (Fixed): self.phrase_matcher is now set. Type: {type(self.phrase_matcher)}, Length: {len(self.phrase_matcher)}")

        except Exception as e:
            logger.error(f"RegisterMeds (Fixed): CRITICAL - Error during PhraseMatcher creation or pattern addition: {e}", exc_info=True)
            self.phrase_matcher = None # Ensure instance attribute is None on any failure in this block
            logger.warning("RegisterMeds (Fixed): self.phrase_matcher set to None due to error in _register_medications.")
        
        # This final check is now redundant if the above try-except sets self.phrase_matcher to None on failure
        # if not self.phrase_matcher:
        #      logger.warning("RegisterMeds (Fixed): At end of method, self.phrase_matcher is None or failed creation.")
    
    def _normalize_medication_name(self, text: str, entity_id: Optional[str], medications: List[str]) -> Optional[str]:
        """Normalize a medication mention to its canonical form."""
        logger.debug(f"NormalizeMedName (Fixed): Starting normalization for text='{text}', entity_id='{entity_id}'")
        
        if entity_id and entity_id in medications: 
            logger.debug(f"NormalizeMedName (Fixed): Found exact match with entity_id='{entity_id}'")
            return entity_id
        
        text_lower = text.lower().strip()
        logger.debug(f"NormalizeMedName (Fixed): Normalized text to lowercase: '{text_lower}'")
        
        # First try exact match
        for med_canonical in medications:
            if med_canonical.lower() == text_lower:
                logger.debug(f"NormalizeMedName (Fixed): Found exact lowercase match: '{med_canonical}'")
                return med_canonical
        
        # Then try regex word boundary match
        sorted_meds_canonical = sorted(medications, key=len, reverse=True)
        for med_canonical in sorted_meds_canonical:
            med_canonical_lower = med_canonical.lower()
            if re.search(r"\b" + re.escape(med_canonical_lower) + r"\b", text_lower, re.IGNORECASE):
                logger.debug(f"NormalizeMedName (Fixed): Found word boundary match: '{med_canonical}' in '{text_lower}'")
                return med_canonical
            if med_canonical_lower in text_lower: 
                logger.debug(f"NormalizeMedName (Fixed): Found substring match: '{med_canonical}' in '{text_lower}'")
                return med_canonical
        
        logger.debug(f"NormalizeMedName (Fixed): No match found for '{text}' in medications list")
        return None
    
    def _build_graph(self, doc: Doc, medications: List[str]) -> nx.DiGraph:
        """Build a directed graph of medication-effect-time relationships."""
        G = nx.DiGraph()
        if not self.nlp:
            logger.error("BuildGraph (Fixed): NLP model not available.")
            return G

        logger.debug(f"BuildGraph (Fixed): Processing text: '{doc.text[:100]}...'")
        logger.debug(f"BuildGraph (Fixed): Looking for medications: {medications}")

        med_spans_map = {} 
        current_doc_ents = list(doc.ents) 
        processed_med_ents_from_ruler = []

        # Log initial entities from EntityRuler
        logger.debug(f"BuildGraph (Fixed): Initial entities from EntityRuler for MEDICATION: {[ent.text for ent in current_doc_ents if ent.label_ == 'MEDICATION']}")
        logger.debug(f"BuildGraph (Fixed): All initial entities: {[(ent.text, ent.label_) for ent in current_doc_ents]}")

        for ent in current_doc_ents:
            if ent.label_ == "MEDICATION":
                logger.debug(f"BuildGraph (Fixed): Processing EntityRuler MEDICATION entity: '{ent.text}' (ID: {ent.ent_id_})")
                normalized_med = self._normalize_medication_name(ent.text, ent.ent_id_, medications)
                if normalized_med:
                    logger.debug(f"BuildGraph (Fixed): Normalized '{ent.text}' to '{normalized_med}'")
                    med_spans_map[(ent.start_char, ent.end_char)] = normalized_med
                    processed_med_ents_from_ruler.append(ent)
                else:
                    logger.warning(f"BuildGraph (Fixed): Could not normalize medication entity '{ent.text}'")

        logger.debug(f"BuildGraph (Fixed): Meds found by EntityRuler & normalized: {[ent.text for ent in processed_med_ents_from_ruler]}")
        
        ruler_ent_char_spans = set((e.start_char, e.end_char) for e in processed_med_ents_from_ruler)
        additional_med_ents_from_pm = []

        # PhraseMatcher processing
        if isinstance(self.phrase_matcher, PhraseMatcher) and len(self.phrase_matcher) > 0: 
            logger.debug(f"BuildGraph (Fixed): Using PhraseMatcher with {len(self.phrase_matcher)} patterns")
            pm_matches = self.phrase_matcher(doc) 
            logger.debug(f"BuildGraph (Fixed): PhraseMatcher found {len(pm_matches)} matches")
            
            for match_id, token_start, token_end in pm_matches:
                pm_span = doc[token_start:token_end]
                pm_char_start, pm_char_end = pm_span.start_char, pm_span.end_char
                
                logger.debug(f"BuildGraph (Fixed): PhraseMatcher found: '{pm_span.text}' at positions {token_start}:{token_end}")
                
                is_overlapping = any(max(pm_char_start, r_char_start) < min(pm_char_end, r_char_end) 
                                   for r_char_start, r_char_end in ruler_ent_char_spans)
                
                if is_overlapping:
                    logger.debug(f"BuildGraph (Fixed): Skipping overlapping PhraseMatcher match: '{pm_span.text}'")
                    continue

                normalized_med = self._normalize_medication_name(pm_span.text, None, medications) 
                if normalized_med:
                    logger.debug(f"BuildGraph (Fixed): PhraseMatcher normalized '{pm_span.text}' to '{normalized_med}'")
                    new_med_span_obj = Span(doc, token_start, token_end, label="MEDICATION")
                    additional_med_ents_from_pm.append(new_med_span_obj)
                    med_spans_map[(pm_char_start, pm_char_end)] = normalized_med
                    ruler_ent_char_spans.add((pm_char_start, pm_char_end)) 
                else:
                    logger.warning(f"BuildGraph (Fixed): Could not normalize PhraseMatcher match '{pm_span.text}'")
        else:
            logger.warning(f"BuildGraph (Fixed): PhraseMatcher not available or empty (type: {type(self.phrase_matcher)})")

        logger.debug(f"BuildGraph (Fixed): Additional meds found by PhraseMatcher & normalized: {[ent.text for ent in additional_med_ents_from_pm]}")

        final_ents = [e for e in current_doc_ents if e.label_ != "MEDICATION"] 
        final_ents.extend(processed_med_ents_from_ruler) 
        final_ents.extend(additional_med_ents_from_pm)  

        try:
            valid_final_ents = [e for e in final_ents if isinstance(e, Span)]
            logger.debug(f"BuildGraph (Fixed): Setting final entities: {[(e.text, e.label_) for e in valid_final_ents]}")
            doc.set_ents(sorted(valid_final_ents, key=lambda e: e.start_char))
            logger.debug(f"BuildGraph (Fixed): Successfully set {len(doc.ents)} entities in doc")
            logger.debug(f"BuildGraph (Fixed): Final doc entities: {[(ent.text, ent.label_) for ent in doc.ents]}")
        except ValueError as e: 
            logger.error(f"BuildGraph (Fixed): Error setting entities in doc: {e}. Entities attempted: {final_ents}", exc_info=True)

        for med_name in set(med_spans_map.values()): 
            if not G.has_node(med_name):
                G.add_node(med_name, type="medication")
                logger.debug(f"BuildGraph (Fixed): Added medication node: {med_name}")
        
        if not isinstance(self.matcher, Matcher): 
            logger.warning(f"BuildGraph (Fixed): Matcher not available (type: {type(self.matcher)}). Cannot extract relations.")
            return G
        
        logger.debug(f"BuildGraph (Fixed): Using Matcher with {len(self.matcher)} patterns")
        relation_matches = self.matcher(doc) 
        logger.debug(f"BuildGraph (Fixed): Found {len(relation_matches)} relation matches")
        
        for match_id, token_start, token_end in relation_matches:
            pattern_name = self.nlp.vocab.strings[match_id]
            relation_span = doc[token_start:token_end]
            logger.debug(f"BuildGraph (Fixed): Processing relation match '{pattern_name}': '{relation_span.text}'")
            
            mentioned_meds_in_relation = []
            rel_char_start, rel_char_end = relation_span.start_char, relation_span.end_char

            for (med_char_start, med_char_end), med_name in med_spans_map.items():
                if med_char_start >= rel_char_start and med_char_end <= rel_char_end:
                    mentioned_meds_in_relation.append(med_name)
                    logger.debug(f"BuildGraph (Fixed): Found medication '{med_name}' in relation span")
            
            if not mentioned_meds_in_relation:
                logger.debug(f"BuildGraph (Fixed): No medications found in relation span, skipping")
                continue 
            
            tense = self._extract_tense(relation_span)
            logger.debug(f"BuildGraph (Fixed): Extracted tense '{tense}' for relation")
            
            main_verb_lemma = None
            for token in relation_span: 
                if token.pos_ == "VERB" and token.dep_ in ["ROOT", "xcomp", "ccomp", "advcl"]:
                    main_verb_lemma = token.lemma_
                    logger.debug(f"BuildGraph (Fixed): Found main verb '{main_verb_lemma}' in relation")
                    break
            verb_category = self._extract_verb_category(main_verb_lemma)
            logger.debug(f"BuildGraph (Fixed): Categorized verb as '{verb_category}'")
            
            relation_node_id = f"relation_{pattern_name}_{relation_span.start_char}_{relation_span.end_char}"
            G.add_node(relation_node_id, 
                       type="relation", pattern=pattern_name, text=relation_span.text,
                       tense=tense, verb_category=verb_category,
                       position=(relation_span.start_char, relation_span.end_char))
            logger.debug(f"BuildGraph (Fixed): Added relation node: {relation_node_id}")
            
            current_confidence = self._calculate_confidence(relation_span, pattern_name, verb_category, tense)
            logger.debug(f"BuildGraph (Fixed): Calculated confidence {current_confidence} for relation")

            for med_name_involved in set(mentioned_meds_in_relation): 
                rel_type_for_edge = "unknown"
                if pattern_name == "CURRENT_MED":
                    rel_type_for_edge = "current" if tense in ["present", "future"] else "past" if tense == "past" else "unknown"
                    if verb_category == "begin": rel_type_for_edge = "starting" if tense == "present" else "started" if tense == "past" else rel_type_for_edge
                    elif verb_category == "end": rel_type_for_edge = "stopping" if tense == "present" else "stopped" if tense == "past" else rel_type_for_edge
                elif pattern_name == "PAST_MED":
                    rel_type_for_edge = "past"
                    if verb_category == "tentative": rel_type_for_edge = "tried"
                elif pattern_name in ["DIRECT_EFFECT", "SIDE_EFFECT"]:
                    rel_type_for_edge = "side_effect" if pattern_name == "SIDE_EFFECT" else "effect"
                
                G.add_edge(med_name_involved, relation_node_id, type=rel_type_for_edge, confidence=current_confidence)
                logger.debug(f"BuildGraph (Fixed): Added edge from '{med_name_involved}' to relation with type '{rel_type_for_edge}' and confidence {current_confidence}")
        
        logger.debug(f"BuildGraph (Fixed): Final graph has {len(G.nodes())} nodes and {len(G.edges())} edges")
        return G
    
    def _extract_tense(self, span: Span) -> str:
        text_lower = span.text.lower()
        if any(marker in text_lower for marker in self.time_markers.get('future', [])): return "future"
        if any(marker in text_lower for marker in self.time_markers.get('present', [])): return "present"
        if any(marker in text_lower for marker in self.time_markers.get('past', [])): return "past"
        root = span.root
        for token in root.subtree:
            if token.pos_ == "VERB":
                if token.tag_ in ["VBP", "VBZ"]: return "present" 
                if token.tag_ == "VBG": 
                    for child in token.children:
                        if child.dep_ == "aux":
                            if child.lemma_ == "be" and child.tag_ in ["VBP", "VBZ", "VB"]: return "present" 
                            if child.lemma_ == "be" and child.tag_ == "VBD": return "past" 
                            if child.lemma_ == "have" and child.tag_ in ["VBP", "VBZ"]: return "present" 
                            if child.lemma_ == "have" and child.tag_ == "VBD": return "past" 
                    if token.head.lemma_ == "be" and token.head.tag_ in ["VBP", "VBZ"]: return "present"
                    if token.head.lemma_ == "be" and token.head.tag_ == "VBD": return "past"
                    if token.dep_ == "ROOT" or token.dep_ == "ccomp" or token.dep_ == "xcomp": return "present"
                if token.tag_ == "VBD": return "past"    
                if token.tag_ == "VBN": 
                     for child in token.children:
                        if child.dep_ == "aux":
                            if child.lemma_ == "have" and child.tag_ in ["VBP", "VBZ"]: return "present" 
                            if child.lemma_ == "have" and child.tag_ == "VBD": return "past"   
                            if child.lemma_ == "be" and child.tag_ in ["VBP", "VBZ", "VB"]: return "present" 
                            if child.lemma_ == "be" and child.tag_ == "VBD": return "past" 
                     if token.dep_ == "ROOT" and not any(c.dep_ == "aux" for c in token.children): return "past"
            if token.tag_ == "MD": 
                if token.lemma_ in ["will", "shall", "'ll"]: return "future"
                if token.lemma_ in ["would", "could", "might", "should"]: return "past" 
        if span.root.pos_ == "VERB":
            if span.root.tag_ in ["VBP", "VBZ", "VBG"]: return "present"
            if span.root.tag_ == "VBD": return "past"
        return "unknown"
    
    def _extract_verb_category(self, verb_lemma: Optional[str]) -> str:
        if verb_lemma is None: return "unknown"
        return self.temporal_verbs.get(verb_lemma.lower(), "other")
    
    def _calculate_confidence(self, span: Span, pattern_name: str, 
                            verb_category: str, tense: str) -> float:
        confidence = 0.3 
        if pattern_name in ["CURRENT_MED", "PAST_MED"]:
            confidence += self.confidence_weights.get('explicit_temporal', 0.3)
            if verb_category not in ["unknown", "other"]: 
                confidence += 0.15
        if pattern_name in ["DIRECT_EFFECT", "SIDE_EFFECT"]:
            confidence += self.confidence_weights.get('direct_causality', 0.25)
        if tense != "unknown":
            confidence += self.confidence_weights.get('tense_consistency', 0.1)
        return min(confidence, 1.0)
    
    def process_text(self, text: str, medications: List[str]) -> Dict[str, Any]:
        logger.debug(f"ProcessText (Fixed) called for text: '{text[:50]}...' and meds: {medications}")
        if not self.nlp:
            logger.error("ProcessText (Fixed): NLP model not loaded. Returning empty results.")
            default_status = {med: "unknown" for med in medications}
            default_confidence = {med: 0.0 for med in medications}
            default_attributions = {med: [] for med in medications}
            return {"graph": nx.DiGraph(), "temporal_status": default_status,
                    "confidence_scores": default_confidence, "effect_attributions": default_attributions}
        self._register_medications(medications) 
        doc = self.nlp(text) 
        graph = self._build_graph(doc, medications)
        temporal_status_results = {}
        confidence_scores_results = {}
        effect_attributions_results = {med: [] for med in medications}
        for med_canonical_name in medications: 
            if not graph.has_node(med_canonical_name):
                temporal_status_results[med_canonical_name] = "unknown"
                confidence_scores_results[med_canonical_name] = 0.0
                continue
            med_relations = []
            if med_canonical_name in graph: 
                for _, relation_node_id, edge_data in graph.out_edges(med_canonical_name, data=True):
                    if relation_node_id in graph:
                        relation_node_data = graph.nodes[relation_node_id]
                        med_relations.append({
                            "type": edge_data.get("type", "unknown"), 
                            "pattern": relation_node_data.get("pattern", "unknown"), 
                            "text": relation_node_data.get("text", ""), 
                            "tense": relation_node_data.get("tense", "unknown"), 
                            "confidence": edge_data.get("confidence", 0.0) 
                        })
                    else:
                        logger.warning(f"ProcessText (Fixed): Relation node ID '{relation_node_id}' not found in graph for med '{med_canonical_name}'. Skipping this edge.")
            if not med_relations:
                temporal_status_results[med_canonical_name] = "unknown"
                confidence_scores_results[med_canonical_name] = 0.0
            else:
                med_relations.sort(key=lambda r: (r["confidence"], 1 if r["pattern"] in ["CURRENT_MED", "PAST_MED"] else 0), reverse=True)
                best_relation_for_status = next((r for r in med_relations if r["pattern"] in ["CURRENT_MED", "PAST_MED"]), None)
                if best_relation_for_status:
                    temporal_status_results[med_canonical_name] = best_relation_for_status["type"]
                    confidence_scores_results[med_canonical_name] = best_relation_for_status["confidence"]
                else:
                    temporal_status_results[med_canonical_name] = "unknown"
                    confidence_scores_results[med_canonical_name] = 0.1 
            for rel in med_relations: 
                if rel["pattern"] in ["DIRECT_EFFECT", "SIDE_EFFECT"]:
                    effect_attributions_results[med_canonical_name].append({
                        "type": rel["type"], "text": rel["text"], "confidence": rel["confidence"]
                    })
        return {"graph": graph, "temporal_status": temporal_status_results,
                "confidence_scores": confidence_scores_results, "effect_attributions": effect_attributions_results}

    def process_dataframe(self, df: pd.DataFrame, text_col: str, med_col: str) -> pd.DataFrame:
        if not self.nlp:
            logger.error("ProcessDataFrame (Fixed): NLP model not available. Returning df with empty result columns.")
            result_df_on_error = df.copy()
            result_df_on_error['medications_fixed'] = result_df_on_error[med_col].apply(fix_medications)
            result_df_on_error['med_temporal_status'] = [{} for _ in range(len(result_df_on_error))]
            result_df_on_error['med_confidence'] =  [{} for _ in range(len(result_df_on_error))]
            result_df_on_error['med_attributions'] =  [{} for _ in range(len(result_df_on_error))]
            result_df_on_error['overall_confidence'] = 0.0
            return result_df_on_error

        result_df = df.copy()
        result_df['med_temporal_status'] = pd.Series([None]*len(result_df), dtype=object)
        result_df['med_confidence'] = pd.Series([None]*len(result_df), dtype=object)
        result_df['med_attributions'] = pd.Series([None]*len(result_df), dtype=object)
        result_df['overall_confidence'] = 0.0 
        
        logger.info(f"ProcessDataFrame (Fixed): Applying medication format fix to column '{med_col}'...")
        result_df['medications_fixed'] = result_df[med_col].apply(fix_medications)
        logger.info("ProcessDataFrame (Fixed): Medication format fix applied.")

        if not result_df.empty:
            logger.debug("ProcessDataFrame (Fixed): Sample of original vs fixed medications (first 5 rows or less):")
            for i in range(min(5, len(result_df))):
                original_val = result_df[med_col].iloc[i]
                fixed_val = result_df['medications_fixed'].iloc[i]
                logger.debug(f"Row {i}: Original='{original_val}' (type: {type(original_val)}) -> Fixed='{fixed_val}' (type: {type(fixed_val)})")
        
        total_rows = len(result_df)
        logger.info(f"ProcessDataFrame (Fixed): Starting to process {total_rows} rows...")

        for idx, row in result_df.iterrows():
            if (idx + 1) % 100 == 0 or idx == total_rows -1 : 
                 logger.info(f"Processing row {idx+1}/{total_rows}...")
            text = row[text_col] if pd.notna(row[text_col]) and isinstance(row[text_col], str) else ""
            meds_list = row['medications_fixed'] 
            if not isinstance(meds_list, list): 
                logger.warning(f"Row {idx}: 'medications_fixed' is not a list (type: {type(meds_list)}, value: {meds_list}). Defaulting to empty list.")
                meds_list = []
            try:
                results = self.process_text(text, meds_list)
                result_df.at[idx, 'med_temporal_status'] = results['temporal_status']
                result_df.at[idx, 'med_confidence'] = results['confidence_scores']
                result_df.at[idx, 'med_attributions'] = results['effect_attributions']
                if results['confidence_scores']: 
                    valid_conf_scores = [s for s in results['confidence_scores'].values() if isinstance(s, (int, float))]
                    if valid_conf_scores:
                        result_df.at[idx, 'overall_confidence'] = sum(valid_conf_scores) / len(valid_conf_scores)
            except Exception as e:
                logger.error(f"ProcessDataFrame (Fixed): Error processing row {idx} (Text: '{text[:50]}...', Meds: {meds_list}): {e}", exc_info=True)
                error_status = {m: "error" for m in meds_list} if meds_list else {}
                error_confidence = {m: 0.0 for m in meds_list} if meds_list else {}
                error_attributions = {m: [] for m in meds_list} if meds_list else {}
                result_df.at[idx, 'med_temporal_status'] = error_status
                result_df.at[idx, 'med_confidence'] = error_confidence
                result_df.at[idx, 'med_attributions'] = error_attributions
                result_df.at[idx, 'overall_confidence'] = 0.0
        logger.info("ProcessDataFrame (Fixed): DataFrame processing complete.")
        return result_df

    def match_emoji_patterns(self, doc: Doc, medications: List[str]) -> List[Tuple[str, str, float]]:
        if not self.nlp: 
            logger.warning("MatchEmoji (Fixed): NLP model not available.")
            return []
        results = []
        emoji_mappings = {
            r"([😀😁😂😃😄😅😆😊😋😎😍🤩😘😗😙😚🙂🤗])": ("emotional_valence", 0.8), 
            r"([👍💖✨🎉🎊🥳])": ("emotional_valence", 0.7), 
            r"([☹🙁😠😡😞😟😥😢😭😩😫😕😔])": ("emotional_valence", 0.2), 
            r"([👎])": ("emotional_valence", 0.3), 
            r"([😮😲😯🤯])": ("activation", 0.8),  
            r"([😐😑😶])": ("emotional_valence", 0.5),     
        }
        emoji_specific_matcher = Matcher(self.nlp.vocab)
        combined_emoji_regex = "|".join(emoji_mappings.keys())
        for med_canonical in medications:
            if not med_canonical or not isinstance(med_canonical, str): continue
            med_parts = [{"ORTH": med_part} for med_part in med_canonical.split()]
            pattern_med_then_emoji = med_parts + [{"IS_SPACE": True, "OP": "?"}, {"TEXT": {"REGEX": combined_emoji_regex}}]
            pattern_emoji_then_med = [{"TEXT": {"REGEX": combined_emoji_regex}}, {"IS_SPACE": True, "OP": "?"}] + med_parts
            emoji_specific_matcher.add(f"EMOJI_CTX_{med_canonical.upper()}", [pattern_med_then_emoji, pattern_emoji_then_med])
        matches = emoji_specific_matcher(doc)
        for match_id, start_token, end_token in matches:
            rule_id_str = self.nlp.vocab.strings[match_id] 
            med_name_from_rule = rule_id_str
            if "EMOJI_CTX_" in rule_id_str:
                med_name_from_rule = rule_id_str.split("EMOJI_CTX_")[-1]
            matched_span = doc[start_token:end_token]
            found_emoji_in_span, dimension_for_emoji, polarity_for_emoji = None, "unknown", 0.5
            for token in matched_span:
                for emoji_regex, (dim, pol) in emoji_mappings.items():
                    if re.fullmatch(emoji_regex, token.text):
                        found_emoji_in_span, dimension_for_emoji, polarity_for_emoji = token.text, dim, pol
                        break
                if found_emoji_in_span: break 
            if found_emoji_in_span:
                normalized_med = self._normalize_medication_name(med_name_from_rule.lower(), med_name_from_rule, medications)
                if normalized_med:
                     results.append((normalized_med, dimension_for_emoji, float(polarity_for_emoji)))
        return results

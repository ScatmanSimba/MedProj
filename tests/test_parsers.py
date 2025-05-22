"""Tests for the feature extraction parsers."""

import pytest
import spacy
from typing import Dict, Any
from src.features.symptom_matcher import SymptomMatcher
from src.features.temporal_parser import TemporalParser
from src.features.emoji_processor import EmojiProcessor

@pytest.fixture
def nlp():
    """Load spaCy model for testing."""
    return spacy.load("en_core_web_sm")

@pytest.fixture
def symptom_matcher(nlp):
    """Create SymptomMatcher instance for testing."""
    return SymptomMatcher(nlp)

@pytest.fixture
def temporal_parser(nlp):
    """Create TemporalParser instance for testing."""
    return TemporalParser(nlp)

@pytest.fixture
def emoji_processor(nlp):
    """Create EmojiProcessor instance for testing."""
    return EmojiProcessor(nlp)

def test_symptom_matcher_canonical(symptom_matcher, nlp):
    """Test symptom matcher with canonical cases."""
    test_cases = [
        (
            "The medication made me feel very anxious and depressed",
            {
                'anxiety': {'dimension': 'emotional', 'confidence': 0.9},
                'depression': {'dimension': 'emotional', 'confidence': 0.9}
            }
        ),
        (
            "I've been feeling tired and unmotivated since starting the medication",
            {
                'fatigue': {'dimension': 'activation', 'confidence': 0.8},
                'anhedonia': {'dimension': 'activation', 'confidence': 0.8}
            }
        ),
        (
            "The medication increased my appetite and caused weight gain",
            {
                'increased_appetite': {'dimension': 'metabolic', 'confidence': 0.9},
                'weight_gain': {'dimension': 'metabolic', 'confidence': 0.9}
            }
        )
    ]
    
    for text, expected_symptoms in test_cases:
        doc = nlp(text)
        matches = symptom_matcher.match_symptoms(doc)
        
        # Check that all expected symptoms are found
        found_symptoms = {m['canonical']: m for m in matches}
        for symptom, info in expected_symptoms.items():
            assert symptom in found_symptoms, f"Expected symptom {symptom} not found in '{text}'"
            match = found_symptoms[symptom]
            assert match['dimension'] == info['dimension'], \
                f"Expected dimension {info['dimension']} for {symptom}, got {match['dimension']}"
            assert match['confidence'] >= info['confidence'], \
                f"Confidence {match['confidence']} below expected {info['confidence']} for {symptom}"

def test_temporal_parser_canonical(temporal_parser, nlp):
    """Test temporal parser with canonical cases."""
    test_cases = [
        (
            "I've been taking the medication for 2 weeks",
            {
                'status': 'current',
                'duration': '2 weeks',
                'confidence': 0.9
            }
        ),
        (
            "I stopped taking the medication last month",
            {
                'status': 'past',
                'time': 'last month',
                'confidence': 0.9
            }
        ),
        (
            "I'm planning to start the medication next week",
            {
                'status': 'prospective',
                'time': 'next week',
                'confidence': 0.9
            }
        )
    ]
    
    for text, expected_status in test_cases:
        doc = nlp(text)
        status = temporal_parser.parse_temporal(doc)
        
        assert status, f"No temporal status found in '{text}'"
        assert status['status'] == expected_status['status'], \
            f"Expected status {expected_status['status']}, got {status['status']}"
        assert status['confidence'] >= expected_status['confidence'], \
            f"Confidence {status['confidence']} below expected {expected_status['confidence']}"

def test_emoji_processor_canonical(emoji_processor, nlp):
    """Test emoji processor with canonical cases."""
    test_cases = [
        (
            "The medication made me feel very 😊",
            {
                'dimension': 'emotional',
                'polarity': 1.0,
                'confidence': 0.9
            }
        ),
        (
            "I've been feeling 😴 since starting the medication",
            {
                'dimension': 'activation',
                'polarity': -1.0,
                'confidence': 0.8
            }
        ),
        (
            "The medication increased my appetite 🍽️",
            {
                'dimension': 'metabolic',
                'polarity': 1.0,
                'confidence': 0.7
            }
        )
    ]
    
    for text, expected_signal in test_cases:
        doc = nlp(text)
        emoji_present, signals = emoji_processor.process_emoji(doc, ["medication"])
        
        assert emoji_present, f"No emoji signals found in '{text}'"
        for med_signals in signals.values():
            for dimension_signals in med_signals.values():
                for signal in dimension_signals:
                    assert signal['dimension'] == expected_signal['dimension'], \
                        f"Expected dimension {expected_signal['dimension']}, got {signal['dimension']}"
                    assert signal['polarity'] == expected_signal['polarity'], \
                        f"Expected polarity {expected_signal['polarity']}, got {signal['polarity']}"
                    assert signal['confidence'] >= expected_signal['confidence'], \
                        f"Confidence {signal['confidence']} below expected {expected_signal['confidence']}"

def test_parser_integration(nlp, symptom_matcher, temporal_parser, emoji_processor):
    """Test integration of all parsers."""
    text = "I've been taking the medication for 2 weeks and feel very anxious 😰"
    doc = nlp(text)
    
    # Test symptom matching
    symptoms = symptom_matcher.match_symptoms(doc)
    assert any(s['canonical'] == 'anxiety' for s in symptoms)
    
    # Test temporal parsing
    temporal = temporal_parser.parse_temporal(doc)
    assert temporal['status'] == 'current'
    assert temporal['duration'] == '2 weeks'
    
    # Test emoji processing
    emoji_present, signals = emoji_processor.process_emoji(doc, ["medication"])
    assert emoji_present
    for med_signals in signals.values():
        assert any(s['dimension'] == 'emotional' for s in med_signals['emotional']) 
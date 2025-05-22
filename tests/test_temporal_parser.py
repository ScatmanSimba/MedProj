"""Tests for temporal parsing functionality."""

import pytest
import spacy
from spacy.tokens import Doc
from src.features.temporal_parser import TemporalParser

@pytest.fixture
def nlp():
    """Load spaCy model."""
    return spacy.load("en_core_web_sm")

@pytest.fixture
def parser(nlp):
    """Create TemporalParser instance."""
    return TemporalParser(nlp)

def test_current_status_detection(parser, nlp):
    """Test detection of current medication status."""
    # Test direct current status
    text = "I am currently taking Prozac and it's working well."
    doc = nlp(text)
    results = parser.parse_temporal(doc)
    assert len(results) == 1
    assert results[0]['status'] == 'current'
    assert results[0]['confidence'] >= 0.8
    
    # Test present tense
    text = "I take Lexapro daily and it helps with my anxiety."
    doc = nlp(text)
    results = parser.parse_temporal(doc)
    assert len(results) == 1
    assert results[0]['status'] == 'current'
    assert results[0]['confidence'] >= 0.6
    
    # Test direct medication reference
    text = "Zoloft is working great for me."
    doc = nlp(text)
    results = parser.parse_temporal(doc)
    assert len(results) == 1
    assert results[0]['status'] == 'current'
    assert results[0]['confidence'] >= 0.4

def test_past_status_detection(parser, nlp):
    """Test detection of past medication status."""
    # Test direct past status
    text = "I stopped taking Prozac last month."
    doc = nlp(text)
    results = parser.parse_temporal(doc)
    assert len(results) == 1
    assert results[0]['status'] == 'past'
    assert results[0]['confidence'] >= 0.8
    
    # Test past tense
    text = "I took Lexapro for 2 years but it didn't help."
    doc = nlp(text)
    results = parser.parse_temporal(doc)
    assert len(results) == 1
    assert results[0]['status'] == 'past'
    assert results[0]['confidence'] >= 0.6
    
    # Test past effects
    text = "Prozac helped with my depression."
    doc = nlp(text)
    results = parser.parse_temporal(doc)
    assert len(results) == 1
    assert results[0]['status'] == 'past'
    assert results[0]['confidence'] >= 0.4

def test_future_status_detection(parser, nlp):
    """Test detection of future medication status."""
    # Test direct future status
    text = "I will start taking Prozac next week."
    doc = nlp(text)
    results = parser.parse_temporal(doc)
    assert len(results) == 1
    assert results[0]['status'] == 'future'
    assert results[0]['confidence'] >= 0.8
    
    # Test future tense
    text = "I'm going to take Lexapro for my anxiety."
    doc = nlp(text)
    results = parser.parse_temporal(doc)
    assert len(results) == 1
    assert results[0]['status'] == 'future'
    assert results[0]['confidence'] >= 0.6
    
    # Test future expectations
    text = "Prozac should help with my depression."
    doc = nlp(text)
    results = parser.parse_temporal(doc)
    assert len(results) == 1
    assert results[0]['status'] == 'future'
    assert results[0]['confidence'] >= 0.4

def test_duration_detection(parser, nlp):
    """Test detection of medication duration."""
    # Test exact duration
    text = "I've been taking Prozac for 6 months."
    doc = nlp(text)
    results = parser.parse_temporal(doc)
    assert len(results) == 1
    assert results[0]['duration']['amount'] == 6.0
    assert results[0]['duration']['unit'] == 'months'
    
    # Test relative duration
    text = "I've been on Lexapro for several weeks."
    doc = nlp(text)
    results = parser.parse_temporal(doc)
    assert len(results) == 1
    assert results[0]['duration']['unit'] == 'weeks'

def test_transition_detection(parser, nlp):
    """Test detection of medication transitions."""
    # Test switching medications
    text = "I switched from Prozac to Lexapro."
    doc = nlp(text)
    results = parser.parse_temporal(doc)
    assert len(results) == 2
    assert any(r['transition']['type'] == 'switch' for r in results)
    
    # Test adding medication
    text = "I added Wellbutrin to my Prozac."
    doc = nlp(text)
    results = parser.parse_temporal(doc)
    assert len(results) == 2
    assert any(r['transition']['type'] == 'add' for r in results)
    
    # Test removing medication
    text = "I stopped taking Prozac."
    doc = nlp(text)
    results = parser.parse_temporal(doc)
    assert len(results) == 1
    assert results[0]['transition']['type'] == 'remove'

def test_negation_handling(parser, nlp):
    """Test handling of negated temporal mentions."""
    # Test negated current status
    text = "I am not currently taking Prozac."
    doc = nlp(text)
    results = parser.parse_temporal(doc)
    assert len(results) == 1
    assert results[0]['status'] == 'past'
    
    # Test negated past status
    text = "I never took Lexapro."
    doc = nlp(text)
    results = parser.parse_temporal(doc)
    assert len(results) == 1
    assert results[0]['status'] == 'unknown'
    
    # Test negated future status
    text = "I will not start taking Prozac."
    doc = nlp(text)
    results = parser.parse_temporal(doc)
    assert len(results) == 1
    assert results[0]['status'] == 'unknown'

def test_false_positive_prevention(parser, nlp):
    """Test prevention of false positive matches."""
    # Test non-medication "quit" context
    text = "I quit vaping last week."
    doc = nlp(text)
    results = parser.parse_temporal(doc)
    assert len(results) == 0
    
    # Test non-medication "taking" context
    text = "I'm taking a break from work."
    doc = nlp(text)
    results = parser.parse_temporal(doc)
    assert len(results) == 0
    
    # Test non-medication "start" context
    text = "I'm starting a new job next month."
    doc = nlp(text)
    results = parser.parse_temporal(doc)
    assert len(results) == 0

def test_cache_handling(parser, nlp):
    """Test cache handling and key generation."""
    # Test same text, different medications
    text1 = "I am taking Prozac."
    text2 = "I am taking Lexapro."
    
    doc1 = nlp(text1)
    doc2 = nlp(text2)
    
    results1 = parser.parse_temporal(doc1)
    results2 = parser.parse_temporal(doc2)
    
    assert results1 != results2
    
    # Test cache hit
    results1_again = parser.parse_temporal(doc1)
    assert results1 == results1_again
    
    # Test cache stats
    stats = parser.get_cache_stats()
    assert stats['temporal_cache']['size'] > 0
    assert stats['temporal_cache']['max_size'] == 1000

def test_secondary_status(parser, nlp):
    """Test detection of secondary temporal status."""
    # Test multiple statuses
    text = "I used to take Prozac, but now I'm on Lexapro."
    doc = nlp(text)
    results = parser.parse_temporal(doc)
    assert len(results) == 2
    assert any(r['secondary_status'] == 'past' for r in results)
    assert any(r['secondary_status'] == 'current' for r in results)
    
    # Test single status
    text = "I'm taking Prozac."
    doc = nlp(text)
    results = parser.parse_temporal(doc)
    assert len(results) == 1
    assert results[0]['secondary_status'] is None

def test_context_extraction(parser, nlp):
    """Test extraction of contextual information."""
    text = "I've been taking Prozac for my depression."
    doc = nlp(text)
    results = parser.parse_temporal(doc)
    
    assert len(results) == 1
    context = results[0]['context']
    assert 'sentence' in context
    assert 'medication_position' in context
    assert 'sentence_position' in context
    assert 'dependency_path' in context 
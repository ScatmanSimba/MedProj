"""Tests for the emoji processor module."""

import pytest
import spacy
from typing import List, Dict, Any
from src.features.emoji_processor import EmojiProcessor

@pytest.fixture
def nlp():
    """Load spaCy model for testing."""
    return spacy.load("en_core_web_sm")

@pytest.fixture
def processor(nlp):
    """Create EmojiProcessor instance for testing."""
    return EmojiProcessor(nlp)

def test_emoji_deduplication(processor):
    """Test that emoji mappings are properly deduplicated."""
    # Check that emojis are stored in frozensets
    for category, info in processor.emoji_mappings.items():
        assert isinstance(info['emojis'], frozenset)
    
    # Check that no emoji appears in multiple categories
    all_emojis = set()
    for category, info in processor.emoji_mappings.items():
        category_emojis = info['emojis']
        assert not (all_emojis & category_emojis), f"Duplicate emoji found in {category}"
        all_emojis.update(category_emojis)

def test_false_friend_detection(processor, nlp):
    """Test false friend detection and handling."""
    test_cases = [
        # Beer emoji in social context
        ("Having a 🍺 with friends at the bar", "emotional"),
        # Beer emoji in metabolic context
        ("The medication made me crave 🍺", "metabolic"),
        # Pizza emoji in food context
        ("Ordered a 🍕 for dinner", "emotional"),
        # Pizza emoji in metabolic context
        ("The medication increased my appetite for 🍕", "metabolic"),
        # Pill emoji in medical context
        ("Taking my 💊 as prescribed", "activation"),
        # Pill emoji in metabolic context
        ("The 💊 affected my metabolism", "metabolic")
    ]
    
    for text, expected_dimension in test_cases:
        doc = nlp(text)
        emoji_present, signals = processor.process_emoji(doc, ["medication"])
        
        if emoji_present:
            for med_signals in signals.values():
                for dimension, dimension_signals in med_signals.items():
                    if dimension_signals:
                        assert dimension == expected_dimension, \
                            f"Expected {expected_dimension} for '{text}', got {dimension}"

def test_caching_mechanism(processor, nlp):
    """Test that caching works correctly."""
    # Process same text twice
    text = "The medication made me feel 😊"
    doc1 = nlp(text)
    doc2 = nlp(text)
    
    # First call should miss cache
    emoji_present1, signals1 = processor.process_emoji(doc1, ["medication"])
    cache_stats1 = processor.get_cache_stats()
    assert cache_stats1['process_emoji']['misses'] > 0
    
    # Second call should hit cache
    emoji_present2, signals2 = processor.process_emoji(doc2, ["medication"])
    cache_stats2 = processor.get_cache_stats()
    assert cache_stats2['process_emoji']['hits'] > cache_stats1['process_emoji']['hits']
    
    # Results should be identical
    assert signals1 == signals2

def test_emoji_context_analysis(processor, nlp):
    """Test emoji context analysis with caching."""
    text = "The medication made me feel very 😊 today"
    doc = nlp(text)
    
    # Process emoji
    emoji_present, signals = processor.process_emoji(doc, ["medication"])
    assert emoji_present
    
    # Check context information
    for med_signals in signals.values():
        for dimension_signals in med_signals.values():
            for signal in dimension_signals:
                assert 'context' in signal
                context = signal['context']
                assert 'sentence' in context
                assert 'emoji_position' in context
                assert 'medication_position' in context
                assert 'dependency_path' in context

def test_cache_clearing(processor, nlp):
    """Test that cache clearing works correctly."""
    # Process some text to populate cache
    text = "The medication made me feel 😊"
    doc = nlp(text)
    processor.process_emoji(doc, ["medication"])
    
    # Get initial cache stats
    initial_stats = processor.get_cache_stats()
    assert initial_stats['process_emoji']['size'] > 0
    
    # Clear caches
    processor.clear_caches()
    
    # Check that caches are empty
    final_stats = processor.get_cache_stats()
    assert final_stats['process_emoji']['size'] == 0
    assert final_stats['emoji_context']['size'] == 0
    assert final_stats['dependency_path']['size'] == 0

def test_emoji_to_category_mapping(processor):
    """Test that emoji to category mapping is correct."""
    # Check that all emojis are in the mapping
    for category, info in processor.emoji_mappings.items():
        for emoji in info['emojis']:
            assert emoji in processor.emoji_to_category
            assert processor.emoji_to_category[emoji] == category

def test_confidence_calculation(processor, nlp):
    """Test confidence calculation for emoji signals."""
    test_cases = [
        ("The medication made me feel very 😊", 0.9),  # High confidence
        ("Maybe the medication made me feel 😊", 0.7),  # Lower confidence due to uncertainty
        ("The medication made me feel 😊 and 😄", 0.8)  # Medium confidence
    ]
    
    for text, min_confidence in test_cases:
        doc = nlp(text)
        emoji_present, signals = processor.process_emoji(doc, ["medication"])
        
        if emoji_present:
            for med_signals in signals.values():
                for dimension_signals in med_signals.values():
                    for signal in dimension_signals:
                        assert signal['confidence'] >= min_confidence, \
                            f"Confidence {signal['confidence']} below minimum {min_confidence} for '{text}'" 
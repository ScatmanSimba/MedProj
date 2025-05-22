"""
Test suite for ResponseAttributor.

This module tests the core functionality of the ResponseAttributor class,
including temporal status detection, proximity calculations, causality detection,
emoji signal processing, and overall response attribution.
"""

import pytest
import spacy
import pandas as pd
import numpy as np
from src.features.response_attribution import ResponseAttributor
from typing import List, Dict, Any, Optional, Tuple

@pytest.fixture
def attributor():
    """Create a ResponseAttributor instance for testing."""
    return ResponseAttributor(model_name="en_core_web_sm", debug=True)

@pytest.fixture
def sample_texts():
    """Sample texts for testing different scenarios."""
    return {
        'current_med': "I am currently taking Prozac and it makes me feel more energetic.",
        'past_med': "I used to take Lexapro but it made me feel numb.",
        'multiple_meds': "I'm on Prozac and Wellbutrin now, but I used to take Lexapro.",
        'emoji_text': "Prozac makes me feel 😊 and 💪, but Lexapro made me feel 😴.",
        'causal_text': "After taking Prozac for 2 weeks, I started feeling better.",
        'negated_text': "I'm not taking Prozac anymore, it didn't help with my energy.",
        'intensity_text': "Prozac makes me feel extremely energetic and very focused."
    }

@pytest.fixture
def sample_text():
    """Sample text for testing."""
    return """
    I've been taking Prozac for 2 months and it makes me feel more energetic.
    I tried Lexapro briefly for a week but it made me too sleepy.
    I'm planning to start Wellbutrin next week.
    """

@pytest.fixture
def sample_medications():
    """Sample medications for testing."""
    return ["Prozac", "Lexapro", "Wellbutrin"]

def test_initialization():
    """Test that the attributor initializes correctly."""
    attributor = ResponseAttributor()
    assert attributor.nlp is not None
    assert attributor.activation_matcher is not None
    assert attributor.emotional_matcher is not None
    assert attributor.metabolic_matcher is not None
    assert len(attributor.activation_keywords) > 0
    assert len(attributor.emotional_keywords) > 0
    assert len(attributor.metabolic_keywords) > 0
    assert attributor.min_signal_threshold == 0.05

def test_detect_basic_temporal_status(attributor, sample_texts):
    """Test temporal status detection for various scenarios."""
    # Test current medication detection
    status = attributor._detect_basic_temporal_status(
        sample_texts['current_med'],
        ["Prozac"]
    )
    assert status["Prozac"]["status"] == "current"
    assert status["Prozac"]["confidence"] > 0.8
    
    # Test past medication detection
    status = attributor._detect_basic_temporal_status(
        sample_texts['past_med'],
        ["Lexapro"]
    )
    assert status["Lexapro"]["status"] == "past"
    assert status["Lexapro"]["confidence"] > 0.8
    
    # Test multiple medications
    status = attributor._detect_basic_temporal_status(
        sample_texts['multiple_meds'],
        ["Prozac", "Wellbutrin", "Lexapro"]
    )
    assert status["Prozac"]["status"] == "current"
    assert status["Wellbutrin"]["status"] == "current"
    assert status["Lexapro"]["status"] == "past"

def test_calculate_proximity_weight(attributor):
    """Test proximity weight calculations."""
    # Create a test document
    doc = attributor.nlp("Prozac makes me feel energetic and focused.")
    
    # Test proximity for token near medication
    token = doc[3]  # "feel"
    weight = attributor._calculate_proximity_weight(token, ["Prozac"])
    assert weight > 0.5  # Should have high weight due to proximity
    
    # Test proximity for token far from medication
    token = doc[5]  # "focused"
    weight = attributor._calculate_proximity_weight(token, ["Prozac"])
    assert weight < 0.5  # Should have lower weight due to distance

def test_detect_causality(attributor):
    """Test causality detection patterns."""
    # Test direct causal pattern
    doc = attributor.nlp("Prozac makes me feel energetic.")
    causality = attributor._detect_causality(doc, 0, 1, 3, 4)  # Prozac -> feel
    assert causality > 0.8  # Should have high causality score
    
    # Test temporal causal pattern
    doc = attributor.nlp("After taking Prozac, I felt better.")
    causality = attributor._detect_causality(doc, 2, 3, 5, 6)  # Prozac -> felt
    assert causality > 0.7  # Should have good causality score
    
    # Test weak causal pattern
    doc = attributor.nlp("I took Prozac. I feel better now.")
    causality = attributor._detect_causality(doc, 1, 2, 4, 5)  # Prozac -> feel
    assert causality < 0.5  # Should have lower causality score

def test_detect_emoji_signals(attributor, sample_texts):
    """Test emoji signal detection and attribution."""
    doc = attributor.nlp(sample_texts['emoji_text'])
    emoji_present, emoji_signals = attributor._detect_emoji_signals(doc, ["Prozac", "Lexapro"])
    
    assert emoji_present
    assert "Prozac" in emoji_signals
    assert "Lexapro" in emoji_signals
    
    # Check Prozac emoji signals
    prozac_signals = emoji_signals["Prozac"]
    assert len(prozac_signals["emotional"]) > 0  # Should have positive emoji
    assert len(prozac_signals["activation"]) > 0  # Should have energetic emoji
    
    # Check Lexapro emoji signals
    lexapro_signals = emoji_signals["Lexapro"]
    assert len(lexapro_signals["activation"]) > 0  # Should have sedated emoji

def test_attribute_responses_integration(attributor, sample_texts):
    """Test complete response attribution pipeline."""
    # Test with current medication
    results = attributor.attribute_responses(
        sample_texts['current_med'],
        ["Prozac"]
    )
    
    assert "dimension_scores" in results
    assert "dimension_confidence" in results
    assert "temporal_status" in results
    assert "emoji_present" in results
    
    # Check dimension scores
    scores = results["dimension_scores"]["Prozac"]
    assert "activation" in scores
    assert scores["activation"] > 0.5  # Should be positive for "energetic"
    
    # Check temporal status
    status = results["temporal_status"]["Prozac"]
    assert status["status"] == "current"
    assert status["confidence"] > 0.8

def test_intensity_detection(attributor, sample_texts):
    """Test intensity modifier detection."""
    doc = attributor.nlp(sample_texts['intensity_text'])
    
    # Test intensity for "energetic"
    intensity = attributor._detect_intensity(doc, 4, 5)  # "energetic"
    assert intensity > 1.0  # Should be amplified by "extremely"
    
    # Test intensity for "focused"
    intensity = attributor._detect_intensity(doc, 6, 7)  # "focused"
    assert intensity > 1.0  # Should be amplified by "very"

def test_negation_handling(attributor, sample_texts):
    """Test negation detection and handling."""
    doc = attributor.nlp(sample_texts['negated_text'])
    
    # Test response attribution with negation
    results = attributor.attribute_responses(
        sample_texts['negated_text'],
        ["Prozac"]
    )
    
    # Check that the temporal status is correctly identified as past
    assert results["temporal_status"]["Prozac"]["status"] == "past"
    
    # Check that the response scores reflect the negation
    scores = results["dimension_scores"]["Prozac"]
    assert "activation" in scores
    assert scores["activation"] < 0.5  # Should be negative due to negation

def test_error_handling(attributor):
    """Test error handling and edge cases."""
    # Test with empty text
    results = attributor.attribute_responses("", ["Prozac"])
    assert results["dimension_scores"] == {}
    assert results["dimension_confidence"] == {}
    
    # Test with no medications
    results = attributor.attribute_responses("I feel better", [])
    assert results["dimension_scores"] == {}
    assert results["dimension_confidence"] == {}
    
    # Test with invalid medication names
    results = attributor.attribute_responses(
        "I'm taking medication",
        ["Invalid Med Name"]
    )
    assert "Invalid Med Name" in results["dimension_scores"]
    assert results["dimension_confidence"]["Invalid Med Name"]["activation"] < 0.5

def test_activation_attribution(attributor, sample_texts):
    """Test attribution of activation-related responses."""
    results = attributor.attribute_responses(
        sample_texts["activation"],
        medications=["Prozac"]
    )
    assert results["dimension_scores"]["Prozac"]["activation"] > 0.7
    assert results["dimension_confidence"]["Prozac"]["activation"] > 0.6

def test_emotional_attribution(attributor, sample_texts):
    """Test attribution of emotional responses."""
    results = attributor.attribute_responses(
        sample_texts["emotional"],
        medications=["Lexapro"]
    )
    assert results["dimension_scores"]["Lexapro"]["emotional"] < 0.3
    assert results["dimension_confidence"]["Lexapro"]["emotional"] > 0.6

def test_metabolic_attribution(attributor, sample_texts):
    """Test attribution of metabolic responses."""
    results = attributor.attribute_responses(
        sample_texts["metabolic"],
        medications=["Zoloft"]
    )
    assert results["dimension_scores"]["Zoloft"]["metabolic"] > 0.7
    assert results["dimension_confidence"]["Zoloft"]["metabolic"] > 0.6

def test_multiple_medications(attributor, sample_texts):
    """Test handling of multiple medications with different responses."""
    results = attributor.attribute_responses(
        sample_texts["multiple"],
        medications=["Prozac", "Wellbutrin"]
    )
    assert results["dimension_scores"]["Prozac"]["activation"] > 0.7
    assert results["dimension_scores"]["Wellbutrin"]["emotional"] < 0.3
    assert results["dimension_confidence"]["Prozac"]["activation"] > 0.6
    assert results["dimension_confidence"]["Wellbutrin"]["emotional"] > 0.6

def test_temporal_status_integration(attributor, sample_texts):
    """Test integration with temporal status information."""
    results = attributor.attribute_responses(
        sample_texts["complex"],
        medications=["Prozac", "Lexapro"],
        temporal_status={"Prozac": "current", "Lexapro": "past"}
    )
    # Current medication should have higher confidence
    assert results["dimension_confidence"]["Prozac"]["activation"] > results["dimension_confidence"]["Lexapro"]["emotional"]

def test_weak_signal_filtering(attributor, sample_texts):
    """Test filtering of weak signals."""
    results = attributor.attribute_responses(
        sample_texts["weak_signal"],
        medications=["Prozac"]
    )
    # Weak signals should be filtered out
    assert pd.isna(results["dimension_scores"]["Prozac"]["activation"])

def test_multi_word_medication(attributor, sample_texts):
    """Test handling of multi-word medication names."""
    results = attributor.attribute_responses(
        sample_texts["multi_word_med"],
        medications=["Abilify Maintena"]
    )
    assert results["dimension_scores"]["Abilify Maintena"]["activation"] > 0.7
    assert results["dimension_confidence"]["Abilify Maintena"]["activation"] > 0.6

def test_dataframe_processing(attributor):
    """Test processing of a pandas DataFrame."""
    df = pd.DataFrame({
        'text': [
            "Prozac makes me feel more energetic",
            "Lexapro made me feel numb",
            "Zoloft increased my appetite",
            "Maybe Prozac affects my mood"  # Weak signal
        ],
        'medications': [
            ["Prozac"],
            ["Lexapro"],
            ["Zoloft"],
            ["Prozac"]
        ],
        'temporal_status': [
            {"Prozac": "current"},
            {"Lexapro": "past"},
            {"Zoloft": "current"},
            {"Prozac": "current"}
        ]
    })
    
    result_df = attributor.process_dataframe(
        df, 'text', 'medications', 'temporal_status'
    )
    
    assert 'response_dimension_scores' in result_df.columns
    assert 'response_dimension_confidence' in result_df.columns
    assert 'activation_score' in result_df.columns
    assert 'emotional_score' in result_df.columns
    assert 'metabolic_score' in result_df.columns
    
    # Check first row (activation)
    assert result_df.iloc[0]['activation_score'] > 0.7
    
    # Check second row (emotional)
    assert result_df.iloc[1]['emotional_score'] < 0.3
    
    # Check third row (metabolic)
    assert result_df.iloc[2]['metabolic_score'] > 0.7
    
    # Check fourth row (weak signal)
    assert pd.isna(result_df.iloc[3]['activation_score'])

def test_confidence_calculation(attributor):
    """Test confidence score calculation for different scenarios."""
    # Test high confidence case
    high_conf_text = "Prozac makes me feel very energetic and alert"
    high_conf_results = attributor.attribute_responses(high_conf_text, ["Prozac"])
    assert high_conf_results["dimension_confidence"]["Prozac"]["activation"] > 0.8
    
    # Test medium confidence case
    med_conf_text = "I think Prozac makes me feel energetic"
    med_conf_results = attributor.attribute_responses(med_conf_text, ["Prozac"])
    assert 0.4 < med_conf_results["dimension_confidence"]["Prozac"]["activation"] < 0.8
    
    # Test low confidence case
    low_conf_text = "Maybe Prozac affects my energy"
    low_conf_results = attributor.attribute_responses(low_conf_text, ["Prozac"])
    assert pd.isna(low_conf_results["dimension_scores"]["Prozac"]["activation"])

def test_causal_detection(attributor):
    """Test enhanced causal detection with dependency parsing."""
    test_cases = [
        # Pattern 1: Direct causal verbs
        {
            "text": "Prozac made me feel more energetic",
            "med": "Prozac",
            "effect": "energetic",
            "expected_score": 1.0
        },
        # Pattern 2: Temporal markers
        {
            "text": "After taking Lexapro, I felt more emotional",
            "med": "Lexapro",
            "effect": "emotional",
            "expected_score": 0.8
        },
        # Pattern 3: Causal prepositions
        {
            "text": "I'm tired because of Seroquel",
            "med": "Seroquel",
            "effect": "tired",
            "expected_score": 0.9
        },
        # Pattern 4: Affect verbs
        {
            "text": "Wellbutrin affects my energy levels",
            "med": "Wellbutrin",
            "effect": "energy",
            "expected_score": 0.85
        },
        # Pattern 5: From preposition
        {
            "text": "I get side effects from Prozac",
            "med": "Prozac",
            "effect": "side effects",
            "expected_score": 0.75
        },
        # Complex case with multiple patterns
        {
            "text": "After starting Lexapro, it made me feel numb and gave me insomnia",
            "med": "Lexapro",
            "effect": "numb",
            "expected_score": 1.0
        }
    ]
    
    for case in test_cases:
        doc = attributor.nlp(case["text"])
        
        # Find medication and effect spans
        med_span = None
        effect_span = None
        
        for token in doc:
            if token.text.lower() == case["med"].lower():
                med_span = (token.i, token.i + 1)
            if case["effect"].lower() in token.text.lower():
                effect_span = (token.i, token.i + 1)
        
        if med_span and effect_span:
            score = attributor._detect_causality(doc, med_span[0], med_span[1], 
                                               effect_span[0], effect_span[1])
            assert score >= case["expected_score"], \
                f"Expected score >= {case['expected_score']} for '{case['text']}', got {score}"

def test_compile_regex_patterns(attributor):
    """Test that regex patterns are properly compiled during initialization."""
    # Verify duration patterns
    assert hasattr(attributor, 'duration_patterns')
    assert len(attributor.duration_patterns) > 0
    assert all(hasattr(pattern, 'search') for pattern in attributor.duration_patterns)
    
    # Verify start date patterns
    assert hasattr(attributor, 'start_date_patterns')
    assert len(attributor.start_date_patterns) > 0
    assert all(hasattr(pattern, 'search') for pattern in attributor.start_date_patterns)
    
    # Verify end date patterns
    assert hasattr(attributor, 'end_date_patterns')
    assert len(attributor.end_date_patterns) > 0
    assert all(hasattr(pattern, 'search') for pattern in attributor.end_date_patterns)

def test_extract_duration(attributor):
    """Test duration extraction with various formats."""
    test_cases = [
        ("for 2 months", "2 months"),
        ("for a couple of weeks", "2 weeks"),
        ("for several days", "4 days"),
        ("for a handful of months", "5 months"),
        ("briefly", None),
        ("for between 2 to 4 weeks", "2 weeks"),
        ("for around 3 months", "3 months"),
        ("for the past few days", None),
        ("trial period of 2 weeks", "2 weeks"),
        ("", None),
        ("random text", None)
    ]
    
    for input_text, expected in test_cases:
        result = attributor.extract_duration(input_text)
        assert result == expected, f"Failed for input: {input_text}"

def test_extract_dates(attributor):
    """Test date extraction with various formats."""
    test_cases = [
        ("since 01/15/2023", ("since 01/15/2023", None)),
        ("from January 15, 2023", ("from January 15, 2023", None)),
        ("until 02/28/2023", (None, "until 02/28/2023")),
        ("since last week until next month", ("since last week", "until next month")),
        ("from the beginning of this month", ("from the beginning of this month", None)),
        ("to the end of this year", (None, "to the end of this year")),
        ("since yesterday until tomorrow", ("since yesterday", "until tomorrow")),
        ("", (None, None)),
        ("random text", (None, None))
    ]
    
    for input_text, expected in test_cases:
        result = attributor.extract_dates(input_text)
        assert result == expected, f"Failed for input: {input_text}"

def test_find_medication_token(attributor, sample_text):
    """Test medication token finding with various cases."""
    doc = attributor.nlp(sample_text)
    
    # Test finding medication in sentence
    for sent in doc.sents:
        if "Prozac" in sent.text:
            token = attributor.find_medication_token(sent, "Prozac")
            assert token is not None
            assert token.text == "Prozac"
    
    # Test medication not in sentence
    for sent in doc.sents:
        if "Prozac" not in sent.text:
            token = attributor.find_medication_token(sent, "Prozac")
            assert token is None
    
    # Test case insensitivity
    for sent in doc.sents:
        if "prozac" in sent.text.lower():
            token = attributor.find_medication_token(sent, "PROZAC")
            assert token is not None
            assert token.text.lower() == "prozac"

def test_check_negation(attributor):
    """Test negation detection with various cases."""
    test_cases = [
        ("I am not taking Prozac", True),
        ("I don't take Prozac", True),
        ("I never took Prozac", True),
        ("I am taking Prozac", False),
        ("I take Prozac", False),
        ("I stopped taking Prozac", False),  # 'stopped' is not a negation
        ("I am definitely not taking Prozac", True),
        ("I am absolutely not taking Prozac", True)
    ]
    
    for text, expected in test_cases:
        doc = attributor.nlp(text)
        for token in doc:
            if token.text == "Prozac":
                result = attributor.check_negation(token)
                assert result == expected, f"Failed for text: {text}"

def test_detect_basic_temporal_status_initialization(attributor, sample_text, sample_medications):
    """Test proper initialization of status dictionary."""
    status = attributor._detect_basic_temporal_status(sample_text, sample_medications)
    
    # Check structure for each medication
    for med in sample_medications:
        assert med in status
        assert isinstance(status[med], dict)
        
        # Check all required fields are present with correct types
        assert status[med]['status'] == 'unknown'  # Initial status
        assert isinstance(status[med]['confidence'], float)
        assert status[med]['evidence'] is None
        assert status[med]['position'] is None
        assert status[med]['duration'] is None
        assert status[med]['start_date'] is None
        assert status[med]['end_date'] is None
        assert status[med]['transition'] is None
        assert status[med]['secondary_status'] is None
        assert isinstance(status[med]['mentions'], list)
        assert status[med]['most_recent'] is None

def test_detect_basic_temporal_status_updates(attributor, sample_text, sample_medications):
    """Test that status dictionary is properly updated."""
    status = attributor._detect_basic_temporal_status(sample_text, sample_medications)
    
    # Check Prozac status (current medication)
    assert status['Prozac']['status'] == 'current'
    assert status['Prozac']['confidence'] > 0.0
    assert status['Prozac']['evidence'] is not None
    assert status['Prozac']['duration'] is not None
    
    # Check Lexapro status (brief usage)
    assert status['Lexapro']['status'] == 'brief'
    assert status['Lexapro']['confidence'] > 0.0
    assert status['Lexapro']['evidence'] is not None
    assert status['Lexapro']['duration'] is not None
    
    # Check Wellbutrin status (prospective)
    assert status['Wellbutrin']['status'] == 'prospective'
    assert status['Wellbutrin']['confidence'] > 0.0
    assert status['Wellbutrin']['evidence'] is not None

def test_edge_cases(attributor):
    """Test various edge cases for temporal status detection."""
    # Empty text
    status = attributor._detect_basic_temporal_status("", ["Prozac"])
    assert status['Prozac']['status'] == 'unknown'
    assert status['Prozac']['confidence'] == 0.0
    
    # No medications
    status = attributor._detect_basic_temporal_status("Some text", [])
    assert len(status) == 0
    
    # Medication mentioned but no temporal context
    status = attributor._detect_basic_temporal_status("I take Prozac", ["Prozac"])
    assert status['Prozac']['status'] == 'current'
    assert status['Prozac']['confidence'] > 0.0
    
    # Multiple temporal mentions for same medication
    text = "I took Prozac for 2 weeks, then stopped. Now I'm taking it again for 3 months."
    status = attributor._detect_basic_temporal_status(text, ["Prozac"])
    assert status['Prozac']['status'] == 'current'
    assert status['Prozac']['confidence'] > 0.0
    assert len(status['Prozac']['mentions']) > 1
    
    # Conflicting temporal mentions
    text = "I took Prozac for 2 weeks, then stopped. I'm not taking it anymore."
    status = attributor._detect_basic_temporal_status(text, ["Prozac"])
    assert status['Prozac']['status'] == 'past'
    assert status['Prozac']['confidence'] > 0.0

def test_negation_handling(attributor):
    """Test proper handling of negated temporal mentions."""
    test_cases = [
        ("I am not taking Prozac", "Prozac", "unknown"),
        ("I don't take Prozac anymore", "Prozac", "past"),
        ("I never took Prozac", "Prozac", "unknown"),
        ("I won't be taking Prozac", "Prozac", "unknown"),
        ("I am not planning to take Prozac", "Prozac", "unknown")
    ]
    
    for text, med, expected_status in test_cases:
        status = attributor._detect_basic_temporal_status(text, [med])
        assert status[med]['status'] == expected_status
        assert status[med]['confidence'] >= 0.0

def test_duration_normalization(attributor):
    """Test proper normalization of duration expressions."""
    test_cases = [
        ("for a couple of weeks", "2 weeks"),
        ("for a few days", "3 days"),
        ("for several months", "4 months"),
        ("for a handful of weeks", "5 weeks"),
        ("for 2 weeks", "2 weeks"),
        ("for one month", "1 month"),
        ("for a short time", None),
        ("briefly", None)
    ]
    
    for input_text, expected in test_cases:
        result = attributor.extract_duration(input_text)
        assert result == expected, f"Failed for input: {input_text}"

def test_date_normalization(attributor):
    """Test proper normalization of date expressions."""
    test_cases = [
        ("since 01/15/2023", "since 01/15/2023"),
        ("from January 15, 2023", "from January 15, 2023"),
        ("since last week", "since last week"),
        ("from the beginning of this month", "from the beginning of this month"),
        ("since yesterday", "since yesterday"),
        ("from a few days ago", "from a few days ago")
    ]
    
    for input_text, expected in test_cases:
        start_date, _ = attributor.extract_dates(input_text)
        assert start_date == expected, f"Failed for input: {input_text}"

def test_causal_confidence_calculation(attributor):
    """Test causal confidence calculation against PRD expectations."""
    test_cases = [
        {
            "text": "Prozac makes me feel energetic.",
            "expected": {
                "base_confidence": 0.9,  # Direct dependency
                "pattern_confidence": 0.9,  # Direct causal pattern
                "temporal_confidence": 0.8,  # Immediate effect
                "event_confidence": 0.9,  # Direct cause
                "final_confidence": 0.9  # High overall confidence
            }
        },
        {
            "text": "After taking Prozac for a week, I started feeling better.",
            "expected": {
                "base_confidence": 0.7,  # Cross-sentence dependency
                "pattern_confidence": 0.7,  # Temporal pattern
                "temporal_confidence": 0.6,  # Delayed effect
                "event_confidence": 0.7,  # Indirect cause
                "final_confidence": 0.7  # Moderate overall confidence
            }
        },
        {
            "text": "I took Prozac. I feel better now.",
            "expected": {
                "base_confidence": 0.3,  # No direct dependency
                "pattern_confidence": 0.3,  # No explicit pattern
                "temporal_confidence": 0.5,  # Neutral temporal
                "event_confidence": 0.3,  # Weak causal link
                "final_confidence": 0.3  # Low overall confidence
            }
        },
        {
            "text": "Prozac might have helped with my energy.",
            "expected": {
                "base_confidence": 0.6,  # Moderate dependency
                "pattern_confidence": 0.6,  # Indirect pattern
                "temporal_confidence": 0.5,  # Neutral temporal
                "event_confidence": 0.6,  # Moderate cause
                "final_confidence": 0.4  # Reduced by uncertainty
            }
        }
    ]
    
    for case in test_cases:
        doc = attributor.nlp(case["text"])
        med_ent = next(ent for ent in doc.ents if ent.label_ == "MEDICATION")
        symptom_span = next(token for token in doc if token.text in ["energetic", "better"])
        
        # Calculate confidence
        confidence = attributor._calculate_causal_confidence(doc, med_ent, symptom_span)
        
        # Validate against expected values
        assert abs(confidence - case["expected"]["final_confidence"]) < 0.1, \
            f"Confidence {confidence} for '{case['text']}' differs from expected {case['expected']['final_confidence']}"

def test_causal_confidence_components(attributor):
    """Test individual components of causal confidence calculation."""
    text = "Prozac makes me feel energetic."
    doc = attributor.nlp(text)
    med_ent = next(ent for ent in doc.ents if ent.label_ == "MEDICATION")
    symptom_span = next(token for token in doc if token.text == "energetic")
    
    # Test dependency distance
    distance, path_quality = attributor._get_dependency_distance(med_ent.root, symptom_span)
    assert distance <= 3  # Should be close in dependency tree
    assert path_quality > 0.8  # Should have high quality path
    
    # Test temporal ordering
    temporal_conf = attributor._check_temporal_ordering(doc, med_ent, symptom_span)
    assert temporal_conf > 0.7  # Should indicate immediate effect
    
    # Test event ordering
    event_conf = attributor._check_event_ordering(doc, med_ent, symptom_span)
    assert event_conf > 0.8  # Should indicate direct cause
    
    # Test uncertainty markers
    has_uncertainty = attributor._has_uncertainty_markers(doc, med_ent, symptom_span)
    assert not has_uncertainty  # Should not have uncertainty markers

def test_coreference_resolution(attributor):
    """Test coreference resolution using coreferee."""
    test_cases = [
        {
            "text": "I started taking Prozac. It made me feel energetic.",
            "expected": "I started taking Prozac. Prozac made me feel energetic."
        },
        {
            "text": "I tried Lexapro for a week. It didn't help with my anxiety.",
            "expected": "I tried Lexapro for a week. Lexapro didn't help with my anxiety."
        },
        {
            "text": "I'm on Wellbutrin now. It's been helping with my focus.",
            "expected": "I'm on Wellbutrin now. Wellbutrin's been helping with my focus."
        },
        {
            "text": "I took Prozac for a month. It made me feel better, but then it started causing insomnia.",
            "expected": "I took Prozac for a month. Prozac made me feel better, but then Prozac started causing insomnia."
        }
    ]
    
    for case in test_cases:
        # Process text
        doc = attributor.nlp(case["text"])
        resolved_doc = attributor._resolve_coreferences(doc)
        
        # Get resolved text
        resolved_text = resolved_doc.text
        
        # Compare with expected
        assert resolved_text == case["expected"], \
            f"Coreference resolution failed for '{case['text']}'\nExpected: {case['expected']}\nGot: {resolved_text}"

def test_coreference_attribution(attributor):
    """Test response attribution with coreference resolution."""
    text = "I started taking Prozac last week. It made me feel more energetic and focused."
    
    # Process with coreference resolution
    results = attributor.attribute_responses(text, ["Prozac"])
    
    # Check that the response is correctly attributed to Prozac
    assert "Prozac" in results["dimension_scores"]
    scores = results["dimension_scores"]["Prozac"]
    
    # Should have positive activation scores
    assert scores["activation"] > 0.7  # High confidence in positive activation
    
    # Check temporal status
    assert results["temporal_status"]["Prozac"]["status"] == "current"
    assert results["temporal_status"]["Prozac"]["confidence"] > 0.8 

def test_confidence_calibration(attributor):
    """Test confidence calibration functionality."""
    # Create sample gold data
    gold_data = pd.DataFrame({
        'symptom_confidence': [0.8, 0.6, 0.4, 0.2],
        'symptom_gold': [0.9, 0.7, 0.3, 0.1],
        'temporal_confidence': [0.7, 0.5, 0.3, 0.1],
        'temporal_gold': [0.8, 0.6, 0.2, 0.1],
        'causal_confidence': [0.9, 0.7, 0.5, 0.3],
        'causal_gold': [0.95, 0.8, 0.4, 0.2],
        'emoji_confidence': [0.6, 0.4, 0.2, 0.1],
        'emoji_gold': [0.7, 0.5, 0.2, 0.1],
        'overall_gold': [0.8, 0.6, 0.3, 0.1]
    })
    
    # Calibrate confidence scores
    metrics = attributor.calibrate_confidence(gold_data)
    
    # Check calibration state
    assert attributor.is_calibrated
    assert all(component in metrics for component in ['symptom', 'temporal', 'causal', 'emoji', 'overall'])
    
    # Check calibration metrics
    for component in metrics:
        assert 'brier_score' in metrics[component]
        assert 'calibration_error' in metrics[component]
        assert 'confidence_correlation' in metrics[component]
        
        # Verify metrics are within expected ranges
        assert 0 <= metrics[component]['brier_score'] <= 1
        assert 0 <= metrics[component]['calibration_error'] <= 1
        assert -1 <= metrics[component]['confidence_correlation'] <= 1

def test_confidence_calibration_application(attributor):
    """Test application of calibrated confidence scores."""
    # First calibrate with sample data
    gold_data = pd.DataFrame({
        'symptom_confidence': [0.8, 0.6, 0.4, 0.2],
        'symptom_gold': [0.9, 0.7, 0.3, 0.1],
        'temporal_confidence': [0.7, 0.5, 0.3, 0.1],
        'temporal_gold': [0.8, 0.6, 0.2, 0.1],
        'causal_confidence': [0.9, 0.7, 0.5, 0.3],
        'causal_gold': [0.95, 0.8, 0.4, 0.2],
        'emoji_confidence': [0.6, 0.4, 0.2, 0.1],
        'emoji_gold': [0.7, 0.5, 0.2, 0.1],
        'overall_gold': [0.8, 0.6, 0.3, 0.1]
    })
    attributor.calibrate_confidence(gold_data)
    
    # Test raw confidence scores
    raw_scores = {
        'symptom': 0.8,
        'temporal': 0.7,
        'causal': 0.9,
        'emoji': 0.6
    }
    
    # Apply calibration
    calibrated_scores = attributor._apply_calibration(raw_scores)
    
    # Check calibrated scores
    assert all(component in calibrated_scores for component in raw_scores)
    assert 'overall' in calibrated_scores
    
    # Verify scores are within [0, 1] range
    for score in calibrated_scores.values():
        assert 0 <= score <= 1
    
    # Verify overall confidence is weighted average
    expected_overall = sum(
        calibrated_scores[component] * attributor.confidence_weights[component]
        for component in ['symptom', 'temporal', 'causal', 'emoji']
    )
    assert abs(calibrated_scores['overall'] - expected_overall) < 1e-6

def test_confidence_calibration_with_missing_data(attributor):
    """Test confidence calibration with missing gold data."""
    # Create sample gold data with missing columns
    gold_data = pd.DataFrame({
        'symptom_confidence': [0.8, 0.6, 0.4, 0.2],
        'symptom_gold': [0.9, 0.7, 0.3, 0.1],
        'temporal_confidence': [0.7, 0.5, 0.3, 0.1],
        'temporal_gold': [0.8, 0.6, 0.2, 0.1]
    })
    
    # Calibrate confidence scores
    metrics = attributor.calibrate_confidence(gold_data)
    
    # Check that only available components are calibrated
    assert 'symptom' in metrics
    assert 'temporal' in metrics
    assert 'causal' not in metrics
    assert 'emoji' not in metrics
    assert 'overall' not in metrics

def test_confidence_calibration_with_invalid_data(attributor):
    """Test confidence calibration with invalid data."""
    # Create sample gold data with invalid values
    gold_data = pd.DataFrame({
        'symptom_confidence': [0.8, 0.6, 0.4, 0.2],
        'symptom_gold': [1.5, -0.1, 0.3, 0.1],  # Invalid values
        'temporal_confidence': [0.7, 0.5, 0.3, 0.1],
        'temporal_gold': [0.8, 0.6, 0.2, 0.1]
    })
    
    # Calibrate confidence scores
    metrics = attributor.calibrate_confidence(gold_data)
    
    # Check that calibration still works with clipped values
    assert 'symptom' in metrics
    assert 'temporal' in metrics
    
    # Verify metrics are within expected ranges
    for component in metrics:
        assert 0 <= metrics[component]['brier_score'] <= 1
        assert 0 <= metrics[component]['calibration_error'] <= 1
        assert -1 <= metrics[component]['confidence_correlation'] <= 1 
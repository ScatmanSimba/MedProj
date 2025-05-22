"""
Tests for the Enhanced Features Module.

This module contains tests for the feature engineering functionality
that builds enhanced features from temporal and attribution information.
"""

import pytest
import pandas as pd
import numpy as np
from src.features.enhanced_features import (
    build_enhanced_features,
    build_temporal_features,
    build_response_attribution_features,
    build_confidence_features,
    build_medication_relationship_features
)

@pytest.fixture
def sample_df():
    """Create a sample DataFrame for testing."""
    return pd.DataFrame({
        'med_temporal_status': [
            {'Prozac': 'current', 'Lexapro': 'past'},
            {'Zoloft': 'current', 'Wellbutrin': 'current'},
            {'Abilify': 'past', 'Risperdal': 'past'},
            {'Prozac': 'current', 'Lexapro': 'stopped'}
        ],
        'response_dimension_confidence': [
            {'Prozac': {'activation': 0.8, 'emotional': 0.7, 'metabolic': 0.6}},
            {'Zoloft': {'activation': 0.9, 'emotional': 0.8, 'metabolic': 0.7}},
            {'Abilify': {'activation': 0.7, 'emotional': 0.6, 'metabolic': 0.5}},
            {'Prozac': {'activation': 0.8, 'emotional': 0.7, 'metabolic': 0.6}}
        ],
        'response_dimension_scores': [
            {'Prozac': {'activation': 0.8, 'emotional': 0.7, 'metabolic': 0.6}},
            {'Zoloft': {'activation': 0.9, 'emotional': 0.8, 'metabolic': 0.7}},
            {'Abilify': {'activation': 0.7, 'emotional': 0.6, 'metabolic': 0.5}},
            {'Prozac': {'activation': 0.8, 'emotional': 0.7, 'metabolic': 0.6}}
        ],
        'med_confidence': [
            {'Prozac': 0.8, 'Lexapro': 0.7},
            {'Zoloft': 0.9, 'Wellbutrin': 0.8},
            {'Abilify': 0.7, 'Risperdal': 0.6},
            {'Prozac': 0.8, 'Lexapro': 0.7}
        ],
        'medication_class': [
            {'Prozac': 'ssri', 'Lexapro': 'ssri'},
            {'Zoloft': 'ssri', 'Wellbutrin': 'ndri'},
            {'Abilify': 'atypical_antipsychotic', 'Risperdal': 'atypical_antipsychotic'},
            {'Prozac': 'ssri', 'Lexapro': 'ssri'}
        ]
    })

def test_build_temporal_features(sample_df):
    """Test building temporal features."""
    result_df = build_temporal_features(sample_df)
    
    # Test one-hot encoding
    assert result_df['med_status_current'].iloc[0] == 1.0
    assert result_df['med_status_past'].iloc[0] == 1.0
    assert result_df['med_status_stopped'].iloc[3] == 1.0
    
    # Test medication counts
    assert result_df['current_med_count'].iloc[0] == 1
    assert result_df['current_med_count'].iloc[1] == 2
    assert result_df['past_med_count'].iloc[2] == 2
    
    # Test current medication ratio
    assert result_df['current_med_ratio'].iloc[0] == 0.5
    assert result_df['current_med_ratio'].iloc[1] == 1.0
    
    # Test class-specific counts
    assert result_df['current_ssri_count'].iloc[0] == 1
    assert result_df['current_ndri_count'].iloc[1] == 1

def test_build_response_attribution_features(sample_df):
    """Test building response attribution features."""
    result_df = build_response_attribution_features(sample_df)
    
    # Test confidence metrics
    assert 'activation_confidence' in result_df.columns
    assert 'emotional_confidence' in result_df.columns
    assert 'metabolic_confidence' in result_df.columns
    assert 'overall_confidence' in result_df.columns
    
    # Test dimension scores
    assert 'current_activation_score' in result_df.columns
    assert 'past_activation_score' in result_df.columns
    assert 'activation_change' in result_df.columns
    
    # Test schema compatibility features
    assert 'activation_sedation_score' in result_df.columns
    assert 'activation_activation_score' in result_df.columns
    
    # Test score calculations
    assert result_df['activation_confidence'].iloc[0] > 0.0
    assert result_df['current_activation_score'].iloc[0] >= -1.0
    assert result_df['current_activation_score'].iloc[0] <= 1.0
    
    # Test score directionality
    assert result_df['activation_sedation_score'].iloc[0] >= 0.0
    assert result_df['activation_activation_score'].iloc[0] >= 0.0
    
    # Test overall confidence
    assert result_df['overall_confidence'].iloc[0] > 0.0
    assert result_df['overall_confidence'].iloc[0] <= 1.0

def test_build_confidence_features(sample_df):
    """Test building confidence features."""
    result_df = build_confidence_features(sample_df)
    
    # Test confidence metrics
    assert result_df['confidence_mean'].iloc[0] > 0.7
    assert result_df['confidence_max'].iloc[1] > 0.8
    assert result_df['confidence_min'].iloc[2] > 0.6
    
    # Test confidence level flags
    assert result_df['high_confidence'].iloc[1] == True
    assert result_df['medium_confidence'].iloc[0] == True
    
    # Test temporal-confidence interaction
    assert result_df['current_high_confidence'].iloc[1] == True

def test_build_medication_relationship_features(sample_df):
    """Test building medication relationship features."""
    result_df = build_medication_relationship_features(sample_df)
    
    # Test switching patterns
    assert result_df['medication_switching'].iloc[0] == 1.0
    assert result_df['medication_switching'].iloc[1] == 0.0
    
    # Test augmentation patterns
    assert result_df['medication_augmentation'].iloc[1] == 1.0
    assert result_df['medication_augmentation'].iloc[0] == 0.0
    
    # Test treatment failures
    assert result_df['treatment_failures'].iloc[2] == 1.0
    assert result_df['treatment_failures'].iloc[0] == 0.0
    
    # Test class switching
    assert result_df['class_switching'].iloc[1] == 1.0  # SSRI + NDRI
    assert result_df['class_switching'].iloc[0] == 0.0  # Both SSRI

def test_build_enhanced_features(sample_df):
    """Test the complete enhanced features pipeline."""
    result_df = build_enhanced_features(sample_df)
    
    # Test that all feature types are present
    assert 'med_status_current' in result_df.columns
    assert 'activation_confidence' in result_df.columns
    assert 'confidence_mean' in result_df.columns
    assert 'medication_switching' in result_df.columns
    
    # Test that features are properly calculated
    assert result_df['current_med_count'].iloc[1] == 2
    assert result_df['activation_confidence'].iloc[0] > 0.7
    assert result_df['confidence_mean'].iloc[1] > 0.8
    assert result_df['medication_switching'].iloc[0] == 1.0

def test_missing_data_handling():
    """Test handling of missing or invalid data."""
    # Create DataFrame with missing data
    df = pd.DataFrame({
        'response_dimension_confidence': [None, {}, {'Prozac': {'activation': 0.8}}],
        'response_dimension_scores': [None, {}, {'Prozac': {'activation': 0.5}}],
        'med_temporal_status': [None, {}, {'Prozac': {'status': 'current'}}]
    })
    
    result_df = build_response_attribution_features(df)
    
    # Test that missing data is handled gracefully
    assert result_df['activation_confidence'].iloc[0] == 0.0
    assert result_df['current_activation_score'].iloc[0] == 0.0
    assert result_df['activation_change'].iloc[0] == 0.0
    assert result_df['activation_sedation_score'].iloc[0] == 0.0
    assert result_df['activation_activation_score'].iloc[0] == 0.0
    assert result_df['overall_confidence'].iloc[0] == 0.0

def test_edge_cases():
    """Test edge cases and boundary conditions."""
    # Create DataFrame with edge cases
    df = pd.DataFrame({
        'med_temporal_status': [
            {'Prozac': 'current'},  # Single medication
            {'Prozac': 'current', 'Lexapro': 'current', 'Zoloft': 'current'},  # Multiple current
            {'Prozac': 'past', 'Lexapro': 'past', 'Zoloft': 'past'}  # All past
        ],
        'response_dimension_confidence': [
            {'Prozac': {'activation': 1.0}},  # Maximum confidence
            {'Prozac': {'activation': 0.0}},  # Minimum confidence
            {'Prozac': {'activation': 0.5}}   # Middle confidence
        ],
        'med_confidence': [
            {'Prozac': 1.0},
            {'Prozac': 0.0},
            {'Prozac': 0.5}
        ]
    })
    
    result_df = build_enhanced_features(df)
    
    # Test single medication case
    assert result_df['current_med_count'].iloc[0] == 1
    assert result_df['current_med_ratio'].iloc[0] == 1.0
    
    # Test multiple current medications
    assert result_df['current_med_count'].iloc[1] == 3
    assert result_df['medication_augmentation'].iloc[1] == 1.0
    
    # Test all past medications
    assert result_df['current_med_count'].iloc[2] == 0
    assert result_df['treatment_failures'].iloc[2] == 1.0
    
    # Test confidence boundaries
    assert result_df['high_confidence'].iloc[0] == True
    assert result_df['low_confidence'].iloc[1] == True
    assert result_df['medium_confidence'].iloc[2] == True

def test_build_response_attribution_features_basic():
    """Test basic feature engineering functionality."""
    # Create test data
    df = pd.DataFrame({
        'response_dimension_confidence': [
            {
                'med1': {'activation': 0.8, 'emotional': 0.7, 'metabolic': 0.6},
                'med2': {'activation': 0.9, 'emotional': 0.8, 'metabolic': 0.7}
            }
        ],
        'response_dimension_scores': [
            {
                'med1': {'activation': 0.5, 'emotional': 0.4, 'metabolic': 0.3},
                'med2': {'activation': 0.6, 'emotional': 0.5, 'metabolic': 0.4}
            }
        ],
        'med_temporal_status': [
            {
                'med1': {'status': 'current'},
                'med2': {'status': 'past'}
            }
        ]
    })
    
    # Build features
    result_df = build_response_attribution_features(df)
    
    # Check basic features
    assert 'activation_confidence' in result_df.columns
    assert 'emotional_confidence' in result_df.columns
    assert 'metabolic_confidence' in result_df.columns
    
    # Check values
    assert result_df['activation_confidence'].iloc[0] == pytest.approx(0.85)  # (0.8 + 0.9) / 2
    assert result_df['current_activation_score'].iloc[0] == pytest.approx(0.5)  # med1 score
    assert result_df['past_activation_score'].iloc[0] == pytest.approx(0.6)  # med2 score

def test_build_response_attribution_features_missing_columns():
    """Test handling of missing columns."""
    # Create test data with missing columns
    df = pd.DataFrame({
        'response_dimension_confidence': [{}],
        'response_dimension_scores': [{}]
    })
    
    # Check that it raises ValueError
    with pytest.raises(ValueError):
        build_response_attribution_features(df)

def test_build_response_attribution_features_invalid_types():
    """Test handling of invalid data types."""
    # Create test data with invalid types
    df = pd.DataFrame({
        'response_dimension_confidence': ['not a dict'],
        'response_dimension_scores': [{}],
        'med_temporal_status': [{}]
    })
    
    # Check that it raises TypeError
    with pytest.raises(TypeError):
        build_response_attribution_features(df)

def test_build_response_attribution_features_with_signals():
    """Test feature engineering with signal counts and strengths."""
    # Create test data with signals
    df = pd.DataFrame({
        'response_dimension_confidence': [
            {
                'med1': {'activation': 0.8, 'emotional': 0.7, 'metabolic': 0.6}
            }
        ],
        'response_dimension_scores': [
            {
                'med1': {'activation': 0.5, 'emotional': 0.4, 'metabolic': 0.3}
            }
        ],
        'med_temporal_status': [
            {
                'med1': {'status': 'current'}
            }
        ],
        'signal_counts': [
            {
                'med1': {'activation': 2, 'emotional': 1, 'metabolic': 3}
            }
        ],
        'signal_strengths': [
            {
                'med1': {'activation': 0.7, 'emotional': 0.6, 'metabolic': 0.8}
            }
        ]
    })
    
    # Build features
    result_df = build_response_attribution_features(df)
    
    # Check signal features
    assert 'activation_signal_count' in result_df.columns
    assert 'activation_signal_strength' in result_df.columns
    
    # Check values
    assert result_df['activation_signal_count'].iloc[0] == 2
    assert result_df['activation_signal_strength'].iloc[0] == pytest.approx(0.7)

def test_build_response_attribution_features_with_uncertainty():
    """Test feature engineering with uncertainty estimates."""
    # Create test data with uncertainty
    df = pd.DataFrame({
        'response_dimension_confidence': [
            {
                'med1': {'activation': 0.8, 'emotional': 0.7, 'metabolic': 0.6}
            }
        ],
        'response_dimension_scores': [
            {
                'med1': {'activation': 0.5, 'emotional': 0.4, 'metabolic': 0.3}
            }
        ],
        'med_temporal_status': [
            {
                'med1': {'status': 'current'}
            }
        ],
        'uncertainty': [
            {
                'med1': {
                    'activation': {'total': 0.2},
                    'emotional': {'total': 0.3},
                    'metabolic': {'total': 0.4}
                }
            }
        ]
    })
    
    # Build features
    result_df = build_response_attribution_features(df)
    
    # Check uncertainty features
    assert 'activation_uncertainty' in result_df.columns
    
    # Check values
    assert result_df['activation_uncertainty'].iloc[0] == pytest.approx(0.2)

def test_build_response_attribution_features_validation():
    """Test validation of output features."""
    # Create test data with out-of-range values
    df = pd.DataFrame({
        'response_dimension_confidence': [
            {
                'med1': {'activation': 1.5, 'emotional': -0.2, 'metabolic': 0.6}
            }
        ],
        'response_dimension_scores': [
            {
                'med1': {'activation': 0.5, 'emotional': 0.4, 'metabolic': 0.3}
            }
        ],
        'med_temporal_status': [
            {
                'med1': {'status': 'current'}
            }
        ]
    })
    
    # Build features
    result_df = build_response_attribution_features(df)
    
    # Check that values are clipped to [0,1]
    assert result_df['activation_confidence'].iloc[0] == 1.0
    assert result_df['emotional_confidence'].iloc[0] == 0.0 
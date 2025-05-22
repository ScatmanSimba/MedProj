"""
Enhanced Features Module.

This module implements feature engineering for temporal and attribution information
in medication response analysis. It builds upon the response attribution module
to create comprehensive features for model training.

The module includes:
1. Temporal Status Features
   - One-hot encoding of medication status
   - Medication counts by status
   - Class-specific temporal features

2. Response Attribution Features
   - Confidence-based features
   - Current vs past medication scores
   - Response change metrics

3. Confidence Features
   - Overall confidence metrics
   - Confidence level flags
   - Temporal-confidence interactions

4. Medication Relationship Features
   - Switching patterns
   - Augmentation patterns
   - Class transition features

Example:
    >>> df = pd.DataFrame({
    ...     'post_text': ['I am taking Prozac and it makes me feel better'],
    ...     'medications': [['Prozac']]
    ... })
    >>> attributor = ResponseAttributor()
    >>> enhanced_df = build_enhanced_features(df, attributor)
    >>> print(enhanced_df['current_med_count'].iloc[0])
    1
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Any, Optional, Tuple, Set
import json
import logging
import itertools
from pathlib import Path
from src.features.response_attribution import ResponseAttributor
from datetime import datetime
from dataclasses import dataclass

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

@dataclass
class FeatureSchema:
    """Schema for enhanced features."""
    name: str
    dtype: str
    description: str
    required: bool = True
    min_value: Optional[float] = None
    max_value: Optional[float] = None
    allowed_values: Optional[Set[Any]] = None

class FeatureSchemaValidator:
    """Validator for enhanced features schema."""
    
    def __init__(self, schema_path: Optional[str] = None):
        """Initialize the schema validator.
        
        Args:
            schema_path: Optional path to schema file
        """
        self.schema = self._load_schema(schema_path) if schema_path else self._get_default_schema()
    
    def _load_schema(self, schema_path: str) -> List[FeatureSchema]:
        """Load schema from file.
        
        Args:
            schema_path: Path to schema file
            
        Returns:
            List of feature schemas
        """
        with open(schema_path, 'r') as f:
            schema_data = json.load(f)
        
        return [FeatureSchema(**feature) for feature in schema_data]
    
    def _get_default_schema(self) -> List[FeatureSchema]:
        """Get default feature schema.
        
        Returns:
            List of feature schemas
        """
        return [
            FeatureSchema(
                name="activation_score",
                dtype="float64",
                description="Activation dimension score",
                min_value=-1.0,
                max_value=1.0
            ),
            FeatureSchema(
                name="emotional_score",
                dtype="float64",
                description="Emotional dimension score",
                min_value=-1.0,
                max_value=1.0
            ),
            FeatureSchema(
                name="metabolic_score",
                dtype="float64",
                description="Metabolic dimension score",
                min_value=-1.0,
                max_value=1.0
            ),
            FeatureSchema(
                name="activation_confidence",
                dtype="float64",
                description="Confidence in activation score",
                min_value=0.0,
                max_value=1.0
            ),
            FeatureSchema(
                name="emotional_confidence",
                dtype="float64",
                description="Confidence in emotional score",
                min_value=0.0,
                max_value=1.0
            ),
            FeatureSchema(
                name="metabolic_confidence",
                dtype="float64",
                description="Confidence in metabolic score",
                min_value=0.0,
                max_value=1.0
            ),
            FeatureSchema(
                name="temporal_status",
                dtype="category",
                description="Temporal status of medication",
                allowed_values={"current", "past", "prospective", "unknown"}
            ),
            FeatureSchema(
                name="signal_count",
                dtype="int64",
                description="Number of signals for medication",
                min_value=0
            ),
            FeatureSchema(
                name="signal_strength",
                dtype="float64",
                description="Average signal strength",
                min_value=0.0,
                max_value=1.0
            )
        ]
    
    def validate_features(self, features: pd.DataFrame) -> Tuple[bool, List[str]]:
        """Validate features against schema.
        
        Args:
            features: DataFrame of features
            
        Returns:
            Tuple of (is_valid, list of validation errors)
        """
        errors = []
        
        # Check required columns
        required_names = {schema.name for schema in self.schema if schema.required}
        missing_columns = required_names - set(features.columns)
        if missing_columns:
            errors.append(f"Missing required columns: {missing_columns}")
        
        # Check each column against its schema
        for schema in self.schema:
            if schema.name not in features.columns:
                continue
            
            # Check dtype
            if features[schema.name].dtype != schema.dtype:
                errors.append(
                    f"Column {schema.name} has dtype {features[schema.name].dtype}, "
                    f"expected {schema.dtype}"
                )
            
            # Check value ranges
            if schema.min_value is not None:
                min_val = features[schema.name].min()
                if min_val < schema.min_value:
                    errors.append(
                        f"Column {schema.name} has minimum value {min_val}, "
                        f"expected >= {schema.min_value}"
                    )
            
            if schema.max_value is not None:
                max_val = features[schema.name].max()
                if max_val > schema.max_value:
                    errors.append(
                        f"Column {schema.name} has maximum value {max_val}, "
                        f"expected <= {schema.max_value}"
                    )
            
            # Check allowed values
            if schema.allowed_values is not None:
                invalid_values = set(features[schema.name].unique()) - schema.allowed_values
                if invalid_values:
                    errors.append(
                        f"Column {schema.name} has invalid values: {invalid_values}, "
                        f"allowed values are: {schema.allowed_values}"
                    )
        
        return len(errors) == 0, errors

def build_enhanced_features(df: pd.DataFrame, attributor: ResponseAttributor, 
                          debug_dir: Optional[str] = None) -> Tuple[pd.DataFrame, Optional[Dict]]:
    """
    Build enhanced features for medication response analysis.
    
    Args:
        df: DataFrame containing post data with required columns:
            - post_text: Text content of the post
            - medications: List of medications mentioned
        attributor: ResponseAttributor instance for response analysis
        debug_dir: Optional directory to store detailed evidence
        
    Returns:
        Tuple of:
        - DataFrame with optimized features for Parquet storage
        - Optional dictionary containing detailed evidence for debug files
    """
    # Validate required columns
    required_columns = ['post_text', 'medications']
    missing_columns = [col for col in required_columns if col not in df.columns]
    if missing_columns:
        raise ValueError(f"Missing required columns: {missing_columns}")
    
    # Initialize feature containers
    temporal_features = []
    attribution_features = []
    confidence_features = []
    relationship_features = []
    
    # Initialize debug data if needed
    debug_data = [] if debug_dir else None
    
    # Process each post
    for idx, row in df.iterrows():
        # Get attribution results
        results = attributor.attribute_responses(
            row['post_text'],
            row['medications']
        )
        
        # Extract temporal features
        temporal = {
            'med_temporal_status': results['temporal_status'],
            'current_med_count': sum(1 for status in results['temporal_status'].values() 
                                   if status['status'] == 'current'),
            'past_med_count': sum(1 for status in results['temporal_status'].values() 
                                if status['status'] == 'past'),
            'brief_med_count': sum(1 for status in results['temporal_status'].values() 
                                 if status['status'] == 'brief'),
            'prospective_med_count': sum(1 for status in results['temporal_status'].values() 
                                       if status['status'] == 'prospective'),
            # Add optional temporal features
            'current_med_ratio': 0.0,  # Will be calculated after counts
            'med_status_current': 1.0 if any(status['status'] == 'current' for status in results['temporal_status'].values()) else 0.0,
            'med_status_past': 1.0 if any(status['status'] == 'past' for status in results['temporal_status'].values()) else 0.0,
            'med_status_future': 1.0 if any(status['status'] in ['brief', 'prospective'] for status in results['temporal_status'].values()) else 0.0
        }
        temporal_features.append(temporal)
        
        # Extract attribution features (optimized for Parquet)
        attribution = {
            # Required attribution features (flattened)
            'activation_sedation_score': np.mean([score['activation'] for score in results['dimension_scores'].values()]),
            'emotional_blunting_restoration_score': np.mean([score['emotional'] for score in results['dimension_scores'].values()]),
            'appetite_metabolic_score': np.mean([score['metabolic'] for score in results['dimension_scores'].values()]),
            # Optional attribution features (flattened)
            'activation_confidence': np.mean([conf['activation'] for conf in results['dimension_confidence'].values()]),
            'emotional_confidence': np.mean([conf['emotional'] for conf in results['dimension_confidence'].values()]),
            'metabolic_confidence': np.mean([conf['metabolic'] for conf in results['dimension_confidence'].values()]),
            # Top evidence scores (flattened)
            'top_activation_score': max([score['activation'] for score in results['dimension_scores'].values()], default=0.0),
            'top_emotional_score': max([score['emotional'] for score in results['dimension_scores'].values()], default=0.0),
            'top_metabolic_score': max([score['metabolic'] for score in results['dimension_scores'].values()], default=0.0),
            # Emoji features (flattened)
            'emoji_present': float(results['emoji_present']),
            'emoji_score': np.mean(list(results['emoji_scores'].values())) if results['emoji_scores'] else 0.0
        }
        attribution_features.append(attribution)
        
        # Extract confidence features (optimized for Parquet)
        confidence = {
            # Required confidence features (flattened)
            'overall_confidence': np.mean([conf for med_conf in results['dimension_confidence'].values() 
                                         for conf in med_conf.values()]),
            # Optional confidence features (flattened)
            'response_confidence': np.mean([conf for med_conf in results['dimension_confidence'].values() 
                                          for conf in med_conf.values()]),
            'temporal_confidence': np.mean([status['confidence'] for status in results['temporal_status'].values()]),
            # Additional confidence features (flattened)
            'has_emoji_signal': float(results['emoji_present']),
            'emoji_confidence': np.mean(list(results['emoji_scores'].values())) if results['emoji_scores'] else 0.0
        }
        confidence_features.append(confidence)
        
        # Extract relationship features (optimized for Parquet)
        relationship = {
            'medication_count': len(row['medications']),
            'has_multiple_meds': float(len(row['medications']) > 1),
            'medication_combinations_count': len(list(itertools.combinations(row['medications'], 2))) if len(row['medications']) > 1 else 0
        }
        relationship_features.append(relationship)
        
        # Store detailed evidence if debug directory is provided
        if debug_dir:
            debug_entry = {
                'post_id': idx,
                'text': row['post_text'],
                'medications': row['medications'],
                'evidence': {
                    'dimension_scores': results['dimension_scores'],
                    'dimension_confidence': results['dimension_confidence'],
                    'temporal_status': results['temporal_status'],
                    'emoji_signals': results['emoji_scores'],
                    'med_responses': results['med_responses']
                }
            }
            debug_data.append(debug_entry)
    
    # Convert feature lists to DataFrames
    temporal_df = pd.DataFrame(temporal_features)
    attribution_df = pd.DataFrame(attribution_features)
    confidence_df = pd.DataFrame(confidence_features)
    relationship_df = pd.DataFrame(relationship_features)
    
    # Calculate current_med_ratio after all counts are available
    temporal_df['current_med_ratio'] = temporal_df['current_med_count'] / (
        temporal_df['current_med_count'] + temporal_df['past_med_count'] + 
        temporal_df['brief_med_count'] + temporal_df['prospective_med_count']
    ).replace(0, 1)  # Avoid division by zero
    
    # Combine all features
    enhanced_df = pd.concat([
        df,
        temporal_df,
        attribution_df,
        confidence_df,
        relationship_df
    ], axis=1)
    
    # Write debug data if provided
    if debug_dir and debug_data:
        debug_path = Path(debug_dir)
        debug_path.mkdir(parents=True, exist_ok=True)
        debug_file = debug_path / f"evidence_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
        with open(debug_file, 'w') as f:
            json.dump(debug_data, f, indent=2)
        logger.info(f"Wrote detailed evidence to {debug_file}")
    
    return enhanced_df

def build_temporal_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Build features based on temporal information.
    
    Args:
        df: DataFrame with temporal status information
        
    Returns:
        DataFrame with temporal features added
        
    Features created:
        - med_status_{status}: One-hot encoded temporal status
        - {status}_med_count: Count of medications in each status
        - current_med_ratio: Ratio of current to total medications
        - current_{class}_count: Count of current medications by class
    """
    # Create a copy to avoid modifying the original
    result_df = df.copy()
    
    # Guard against missing column
    if 'med_temporal_status' not in df.columns:
        logger.warning("med_temporal_status column not found. Returning original DataFrame.")
        return df
    
    # One-hot encode temporal status
    for status in ['current', 'past', 'unknown', 'stopped', 'starting', 'tried']:
        result_df[f'med_status_{status}'] = result_df['med_temporal_status'].apply(
            lambda x: 1.0 if isinstance(x, dict) and any(s == status for s in x.values()) else 0.0
        )
    
    # Create features for medication count by status
    result_df['current_med_count'] = result_df['med_temporal_status'].apply(
        lambda x: sum(1 for s in x.values() if s == 'current') if isinstance(x, dict) else 0
    )
    
    result_df['past_med_count'] = result_df['med_temporal_status'].apply(
        lambda x: sum(1 for s in x.values() if s == 'past') if isinstance(x, dict) else 0
    )
    
    result_df['unknown_med_count'] = result_df['med_temporal_status'].apply(
        lambda x: sum(1 for s in x.values() if s == 'unknown') if isinstance(x, dict) else 0
    )
    
    # Calculate ratio of current to total medications with safe division
    result_df['current_med_ratio'] = result_df.apply(
        lambda row: row['current_med_count'] / max(len(row['med_temporal_status']), 1)
        if isinstance(row['med_temporal_status'], dict) else 0.0,
        axis=1
    )
    
    # For each drug class, calculate how many are current vs past
    if 'medications' in result_df.columns and 'medication_class' in result_df.columns:
        for med_class in ['ssri', 'snri', 'atypical_antipsychotic', 'ndri']:
            result_df[f'current_{med_class}_count'] = result_df.apply(
                lambda row: sum(1 for med, status in row['med_temporal_status'].items() 
                              if status == 'current' and row['medication_class'].get(med) == med_class)
                if isinstance(row['med_temporal_status'], dict) and isinstance(row['medication_class'], dict)
                else 0,
                axis=1
            )
    
    return result_df

def build_response_attribution_features(
    df: pd.DataFrame,
    attributor: Any,
    schema_validator: Optional[FeatureSchemaValidator] = None
) -> pd.DataFrame:
    """Build features from response attribution information.
    
    Args:
        df: Input DataFrame
        attributor: ResponseAttributor instance
        schema_validator: Optional schema validator
        
    Returns:
        DataFrame with enhanced features
    """
    # Initialize schema validator if not provided
    if schema_validator is None:
        schema_validator = FeatureSchemaValidator()
    
    # Process each row
    enhanced_features = []
    for _, row in df.iterrows():
        # Get attribution results
        results = attributor.attribute_responses(
            row['text'],
            row['medications']
        )
        
        # Extract features
        features = {}
        
        # Add dimension scores
        for med, scores in results['dimension_scores'].items():
            for dimension, score in scores.items():
                features[f"{dimension}_score"] = score
        
        # Add confidence scores
        for med, confidences in results['dimension_confidence'].items():
            for dimension, confidence in confidences.items():
                features[f"{dimension}_confidence"] = confidence
        
        # Add temporal status
        if results['temporal_status']:
            features['temporal_status'] = results['temporal_status'].get('status', 'unknown')
        
        # Add signal counts and strengths
        features['signal_count'] = sum(
            count for counts in results['signal_counts'].values()
            for count in counts.values()
        )
        
        features['signal_strength'] = np.mean([
            strength for strengths in results['signal_strengths'].values()
            for strength in strengths.values()
        ])
        
        enhanced_features.append(features)
    
    # Convert to DataFrame
    result_df = pd.DataFrame(enhanced_features)
    
    # Validate against schema
    is_valid, errors = schema_validator.validate_features(result_df)
    if not is_valid:
        error_msg = "\n".join(errors)
        raise ValueError(f"Feature validation failed:\n{error_msg}")
    
    return result_df

def build_confidence_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Build features based on confidence scores.
    
    Args:
        df: DataFrame with confidence information
        
    Returns:
        DataFrame with confidence features added
        
    Features created:
        - confidence_mean: Mean confidence across medications
        - confidence_max: Maximum confidence
        - confidence_min: Minimum confidence
        - confidence_std: Standard deviation of confidence
        - high/medium/low_confidence: Confidence level flags
        - current_high_confidence: Flag for high confidence current medications
    """
    # Create a copy to avoid modifying the original
    result_df = df.copy()
    
    # Guard against missing column
    if 'med_confidence' not in df.columns:
        logger.warning("med_confidence column not found. Skipping confidence feature creation.")
        return df
    
    # Calculate overall confidence metrics
    result_df['confidence_mean'] = result_df['med_confidence'].apply(
        lambda x: np.mean(list(x.values())) if isinstance(x, dict) and x else 0.0
    )
    
    result_df['confidence_max'] = result_df['med_confidence'].apply(
        lambda x: max(x.values(), default=0.0) if isinstance(x, dict) and x else 0.0
    )
    
    result_df['confidence_min'] = result_df['med_confidence'].apply(
        lambda x: min(x.values(), default=0.0) if isinstance(x, dict) and x else 0.0
    )
    
    result_df['confidence_std'] = result_df['med_confidence'].apply(
        lambda x: np.std(list(x.values())) if isinstance(x, dict) and len(x) > 1 else 0.0
    )
    
    # Create confidence level flags using np.select for cleaner binning
    conditions = [
        result_df['confidence_mean'] > 0.7,
        (result_df['confidence_mean'] >= 0.4) & (result_df['confidence_mean'] <= 0.7),
        result_df['confidence_mean'] < 0.4
    ]
    choices = ['high', 'medium', 'low']
    
    result_df['confidence_level'] = np.select(conditions, choices, default='unknown')
    
    # Create interaction features combining confidence with temporal status
    if 'med_temporal_status' in result_df.columns:
        result_df['current_high_confidence'] = result_df.apply(
            lambda row: any(
                status == 'current' and row['med_confidence'].get(med, 0) > 0.7
                for med, status in row['med_temporal_status'].items()
            ) if all(isinstance(x, dict) for x in [row['med_confidence'], row['med_temporal_status']]) else False,
            axis=1
        )
    
    return result_df

def build_medication_relationship_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Build features about relationships between medications.
    
    Args:
        df: DataFrame with medication information
        
    Returns:
        DataFrame with medication relationship features added
        
    Features created:
        - medication_switching: Flag for past->current transitions
        - medication_augmentation: Flag for multiple current medications
        - treatment_failures: Flag for multiple past medications
        - class_switching: Flag for medication class transitions
    """
    # Create a copy to avoid modifying the original
    result_df = df.copy()
    
    # Guard against missing column
    if 'med_temporal_status' not in df.columns:
        logger.warning("med_temporal_status column not found. Skipping relationship feature creation.")
        return df
    
    # Calculate medication sequence features
    # Flag switching patterns (past -> current transitions)
    result_df['medication_switching'] = result_df['med_temporal_status'].apply(
        lambda x: 1.0 if isinstance(x, dict) and any(s == 'past' for s in x.values()) and any(s == 'current' for s in x.values()) else 0.0
    )
    
    # Flag augmentation patterns (multiple current medications)
    result_df['medication_augmentation'] = result_df.apply(
        lambda row: 1.0 if row['current_med_count'] > 1 else 0.0,
        axis=1
    )
    
    # Flag treatment failure patterns (multiple past medications)
    result_df['treatment_failures'] = result_df.apply(
        lambda row: 1.0 if row['past_med_count'] > 1 else 0.0,
        axis=1
    )
    
    # Calculate medication class transition features if class info available
    if 'medications' in result_df.columns and 'medication_class' in result_df.columns:
        # Flag class switching (e.g., SSRI to SNRI)
        result_df['class_switching'] = result_df.apply(
            lambda row: 1.0 if all(isinstance(x, dict) for x in [row['medications'], row['medication_class']]) and len(set([
                row['medication_class'].get(med) 
                for med in row['medications']
                if med in row['medication_class']
            ])) > 1 else 0.0,
            axis=1
        )
    
    return result_df 

def save_feature_schema(schema: List[FeatureSchema], path: str) -> None:
    """Save feature schema to file.
    
    Args:
        schema: List of feature schemas
        path: Path to save schema
    """
    schema_data = [
        {
            'name': s.name,
            'dtype': s.dtype,
            'description': s.description,
            'required': s.required,
            'min_value': s.min_value,
            'max_value': s.max_value,
            'allowed_values': list(s.allowed_values) if s.allowed_values else None
        }
        for s in schema
    ]
    
    with open(path, 'w') as f:
        json.dump(schema_data, f, indent=2) 

def build_temporal_features(self, med_temporal_status: Dict[str, Any]) -> Dict[str, float]:
    """Build temporal features from medication temporal status.
    
    Args:
        med_temporal_status: Dictionary with temporal status information
        
    Returns:
        Dictionary with temporal features
    """
    if not isinstance(med_temporal_status, dict):
        logger.warning(f"Expected dict for med_temporal_status, got {type(med_temporal_status)}")
        return {
            'temporal_recency': 0.0,
            'temporal_duration': 0.0,
            'temporal_confidence': 0.0
        }
    
    # Extract temporal features with type safety
    recency = med_temporal_status.get('recency', 0.0)
    duration = med_temporal_status.get('duration', 0.0)
    confidence = med_temporal_status.get('confidence', 0.0)
    
    # Ensure numeric values
    try:
        recency = float(recency)
        duration = float(duration)
        confidence = float(confidence)
    except (ValueError, TypeError):
        logger.warning("Non-numeric temporal values encountered")
        return {
            'temporal_recency': 0.0,
            'temporal_duration': 0.0,
            'temporal_confidence': 0.0
        }
    
    return {
        'temporal_recency': recency,
        'temporal_duration': duration,
        'temporal_confidence': confidence
    } 
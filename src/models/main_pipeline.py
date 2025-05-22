"""
Main pipeline for medication response prediction.

This module implements the complete pipeline for training confidence-aware
models to predict medication response across three dimensions.
"""

import argparse
import pandas as pd
import numpy as np
import logging
from pathlib import Path
import json
from typing import Dict, List, Any, Set, Tuple, Optional
from datetime import datetime
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import MinMaxScaler
import matplotlib.pyplot as plt
import seaborn as sns
import xgboost as xgb
import warnings
import shutil
from sklearn.isotonic import IsotonicRegression

# Import custom modules
from src.features.response_attribution import ResponseAttributor
from src.features.enhanced_features import build_enhanced_features
from src.models.confidence_aware_training import ConfidenceAwareTrainer

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler("pipeline.log"),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger("main_pipeline")

# Define expected feature schemas
FEATURE_SCHEMAS = {
    'receptor': {
        'required': ['d2_affinity', '5ht2a_affinity', 'h1_affinity', 'alpha1_affinity', 'm1_affinity'],
        'optional': ['d1_affinity', 'd3_affinity', 'd4_affinity', 'd5_affinity', '5ht1a_affinity', '5ht2c_affinity']
    },
    'temporal': {
        'required': ['med_temporal_status', 'current_med_count', 'past_med_count'],
        'optional': ['current_med_ratio', 'med_status_current', 'med_status_past', 'med_status_future']
    },
    'confidence': {
        'required': ['overall_confidence'],
        'optional': ['med_confidence', 'response_confidence', 'temporal_confidence']
    },
    'class': {
        'required': [],
        'optional': ['class_antipsychotic', 'class_antidepressant', 'class_mood_stabilizer']
    },
    'attribution': {
        'required': ['activation_sedation_score', 'emotional_blunting_restoration_score', 'appetite_metabolic_score'],
        'optional': ['activation_confidence', 'emotional_confidence', 'metabolic_confidence']
    },
    'dosage': {
        'required': [],
        'optional': ['current_dosage', 'past_dosage', 'dosage_unit']
    }
}

def validate_feature_schema(df: pd.DataFrame) -> Dict[str, List[str]]:
    """
    Validate presence of required features and log missing columns.
    
    Args:
        df: Input DataFrame
        
    Returns:
        Dictionary of missing required features by category
        
    Raises:
        ValueError: If critical required features are missing
    """
    missing_features = {}
    critical_missing = []
    
    for category, schema in FEATURE_SCHEMAS.items():
        missing = []
        for feature in schema['required']:
            if feature not in df.columns:
                missing.append(feature)
                if category in ['receptor', 'attribution']:  # Critical features
                    critical_missing.append(feature)
        
        if missing:
            missing_features[category] = missing
            logger.warning(f"Missing {category} features: {missing}")
    
    if critical_missing:
        raise ValueError(f"Missing critical required features: {critical_missing}")
    
    return missing_features

def get_feature_columns(df: pd.DataFrame) -> Dict[str, List[str]]:
    """
    Get feature columns based on schema and available columns.
    
    Args:
        df: Input DataFrame
        
    Returns:
        Dictionary of feature columns by category
    """
    feature_cols = {}
    
    for category, schema in FEATURE_SCHEMAS.items():
        # Get required features that are present
        required = [col for col in schema['required'] if col in df.columns]
        
        # Get optional features that are present
        optional = [col for col in schema['optional'] if col in df.columns]
        
        # Add any additional columns matching category patterns
        if category == 'receptor':
            additional = [col for col in df.columns if any(x in col for x in ['_affinity'])]
        elif category == 'temporal':
            additional = [col for col in df.columns if col.startswith('med_status_')]
        elif category == 'confidence':
            additional = [col for col in df.columns if 'confidence' in col and col != 'overall_confidence']
        elif category == 'class':
            additional = [col for col in df.columns if col.startswith('class_')]
        elif category == 'attribution':
            additional = [col for col in df.columns if any(dim in col for dim in 
                         ['activation', 'emotional', 'metabolic']) and 
                        col not in schema['required']]
        elif category == 'dosage':
            additional = [col for col in df.columns if 'dosage' in col or 'dose' in col]
        
        # Combine all features for this category
        feature_cols[category] = list(set(required + optional + additional))
    
    return feature_cols

def handle_missing_values(df: pd.DataFrame, feature_cols: Dict[str, List[str]]) -> pd.DataFrame:
    """
    Handle missing values in features using appropriate strategies.
    
    Args:
        df: Input DataFrame
        feature_cols: Dictionary of feature columns by category
        
    Returns:
        DataFrame with missing values handled
    """
    # Create a copy to avoid modifying original
    df_clean = df.copy()
    
    # Handle missing values by category
    for category, cols in feature_cols.items():
        if not cols:
            continue
            
        if category == 'receptor':
            # Receptor affinities: fill with median
            imputer = SimpleImputer(strategy='median')
            df_clean[cols] = imputer.fit_transform(df_clean[cols])
            
        elif category == 'temporal':
            # Temporal features: fill with 0
            df_clean[cols] = df_clean[cols].fillna(0)
            
        elif category == 'confidence':
            # Confidence scores: fill with minimum confidence
            df_clean[cols] = df_clean[cols].fillna(0.1)
            
        elif category == 'class':
            # Class indicators: fill with 0 (no class)
            df_clean[cols] = df_clean[cols].fillna(0)
            
        elif category == 'attribution':
            # Response scores: fill with neutral value
            df_clean[cols] = df_clean[cols].fillna(0.5)
            
        elif category == 'dosage':
            # Dosage: fill with 0
            df_clean[cols] = df_clean[cols].fillna(0)
    
    # Verify no missing values remain
    missing = df_clean.isnull().sum().sum()
    if missing > 0:
        raise ValueError(f"Found {missing} missing values after cleaning")
    
    return df_clean

def split_temporal_data(df: pd.DataFrame, 
                       test_size: float = 0.2,
                       date_col: str = 'created_utc') -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Split data chronologically for temporal validation.
    
    Args:
        df: Input DataFrame
        test_size: Proportion of data to use for testing
        date_col: Column containing timestamps
        
    Returns:
        Tuple of (train_df, test_df)
    """
    if date_col not in df.columns:
        raise ValueError(f"Date column {date_col} not found in DataFrame")
    
    # Sort by date
    df_sorted = df.sort_values(date_col)
    
    # Calculate split point
    cutoff = int(len(df_sorted) * (1 - test_size))
    
    # Split data
    train_df = df_sorted.iloc[:cutoff]
    test_df = df_sorted.iloc[cutoff:]
    
    # Log split information
    train_start = train_df[date_col].min()
    train_end = train_df[date_col].max()
    test_start = test_df[date_col].min()
    test_end = test_df[date_col].max()
    
    logger.info(f"Temporal split:")
    logger.info(f"  Train: {len(train_df)} samples from {train_start} to {train_end}")
    logger.info(f"  Test: {len(test_df)} samples from {test_start} to {test_end}")
    
    return train_df, test_df

def calibrate_confidence_scores(self, df: pd.DataFrame) -> pd.DataFrame:
    """Calibrate confidence scores across all components.
    
    Args:
        df: Input DataFrame with raw confidence scores
        
    Returns:
        DataFrame with calibrated confidence scores
    """
    # Initialize calibration models
    self.calibration_models = {
        'symptom': IsotonicRegression(out_of_bounds='clip'),
        'temporal': IsotonicRegression(out_of_bounds='clip'),
        'causal': IsotonicRegression(out_of_bounds='clip'),
        'emoji': IsotonicRegression(out_of_bounds='clip'),
        'overall': IsotonicRegression(out_of_bounds='clip')
    }
    
    # Get calibration data
    calibration_data = self._get_calibration_data(df)
    
    # Calibrate each component
    for component, model in self.calibration_models.items():
        if component in calibration_data:
            X = calibration_data[component]['scores'].values.reshape(-1, 1)
            y = calibration_data[component]['true_scores'].values
            
            # Fit calibration model
            model.fit(X, y)
            
            # Apply calibration
            df[f'{component}_confidence'] = model.predict(
                df[f'{component}_confidence'].values.reshape(-1, 1)
            )
    
    # Calibrate overall confidence
    if 'overall' in calibration_data:
        X = calibration_data['overall']['scores'].values.reshape(-1, 1)
        y = calibration_data['overall']['true_scores'].values
        
        self.calibration_models['overall'].fit(X, y)
        
        df['overall_confidence'] = self.calibration_models['overall'].predict(
            df['overall_confidence'].values.reshape(-1, 1)
        )
    
    return df

def _get_calibration_data(self, df: pd.DataFrame) -> Dict[str, Dict[str, np.ndarray]]:
    """Get calibration data for each component.
    
    Args:
        df: Input DataFrame
        
    Returns:
        Dictionary mapping components to their calibration data
    """
    calibration_data = {}
    
    # Get symptom calibration data
    if 'symptom_confidence' in df.columns and 'symptom_gold' in df.columns:
        calibration_data['symptom'] = {
            'scores': df['symptom_confidence'].values,
            'true_scores': df['symptom_gold'].values
        }
    
    # Get temporal calibration data
    if 'temporal_confidence' in df.columns and 'temporal_gold' in df.columns:
        calibration_data['temporal'] = {
            'scores': df['temporal_confidence'].values,
            'true_scores': df['temporal_gold'].values
        }
    
    # Get causal calibration data
    if 'causal_confidence' in df.columns and 'causal_gold' in df.columns:
        calibration_data['causal'] = {
            'scores': df['causal_confidence'].values,
            'true_scores': df['causal_gold'].values
        }
    
    # Get emoji calibration data
    if 'emoji_confidence' in df.columns and 'emoji_gold' in df.columns:
        calibration_data['emoji'] = {
            'scores': df['emoji_confidence'].values,
            'true_scores': df['emoji_gold'].values
        }
    
    # Get overall calibration data
    if all(col in df.columns for col in ['symptom_confidence', 'temporal_confidence', 
                                       'causal_confidence', 'emoji_confidence', 'overall_gold']):
        # Calculate weighted average of component confidences
        weights = self.config.get('confidence_weights', {
            'symptom': 0.4,
            'temporal': 0.3,
            'causal': 0.2,
            'emoji': 0.1
        })
        
        overall_scores = (
            weights['symptom'] * df['symptom_confidence'] +
            weights['temporal'] * df['temporal_confidence'] +
            weights['causal'] * df['causal_confidence'] +
            weights['emoji'] * df['emoji_confidence']
        )
        
        calibration_data['overall'] = {
            'scores': overall_scores.values,
            'true_scores': df['overall_gold'].values
        }
    
    return calibration_data

def calibrate_dataframe_confidence(self, df: pd.DataFrame) -> pd.DataFrame:
    """Calibrate confidence scores in DataFrame.
    
    Args:
        df: Input DataFrame
        
    Returns:
        DataFrame with calibrated confidence scores
    """
    # Ensure all confidence scores are in [0, 1] range
    confidence_cols = [col for col in df.columns if col.endswith('_confidence')]
    for col in confidence_cols:
        df[col] = df[col].clip(0, 1)
    
    # Apply component-specific calibration
    df = self.calibrate_confidence_scores(df)
    
    # Apply minimum signal filtering
    min_signal_count = self.config.get('min_signal_count', 2)
    min_signal_strength = self.config.get('min_signal_strength', 0.3)
    
    # Filter based on minimum signal count
    signal_counts = df[confidence_cols].notna().sum(axis=1)
    df = df[signal_counts >= min_signal_count]
    
    # Filter based on minimum signal strength
    max_confidence = df[confidence_cols].max(axis=1)
    df = df[max_confidence >= min_signal_strength]
    
    return df

def plot_confidence_error_correlation(
    data_dict: Dict[str, Any],
    results: Dict[str, Any],
    output_dir: str = None
):
    """
    Create scatter plots showing correlation between confidence and prediction errors.
    
    Args:
        data_dict: Dictionary with prepared data
        results: Dictionary with model results
        output_dir: Directory to save plots
    """
    # Create output directory if provided
    if output_dir:
        Path(output_dir).mkdir(parents=True, exist_ok=True)
    
    # Create plot for each target
    for target_name in results['confidence_correlation']:
        target_corr = results['confidence_correlation'][target_name]['overall']['correlation']
        target_p = results['confidence_correlation'][target_name]['overall']['p_value']
        
        # Get test data
        X_test = data_dict['test']['X']
        y_test = data_dict['test']['y'][target_name]
        confidence = data_dict['test']['confidence']
        
        # Get predictions (handle different result structures)
        if 'weighted' in results[target_name]:
            model = results[target_name]['weighted']['model']
        else:
            model = results[target_name]['model']
            
        dtest = xgb.DMatrix(X_test)
        preds = model.predict(dtest)
        
        # Calculate absolute errors
        abs_errors = np.abs(y_test - preds)
        
        # Create scatter plot
        plt.figure(figsize=(10, 6))
        
        sns.regplot(
            x=confidence, 
            y=abs_errors,
            scatter_kws={'alpha': 0.5},
            line_kws={'color': 'red'}
        )
        
        # Add correlation line and annotation
        plt.title(f"{target_name}: Confidence vs. Prediction Error")
        plt.xlabel("Confidence Score")
        plt.ylabel("Absolute Error")
        
        # Add correlation information
        plt.annotate(
            f"Correlation: {target_corr:.3f}\nP-value: {target_p:.3e}",
            xy=(0.05, 0.95),
            xycoords='axes fraction',
            bbox=dict(boxstyle="round,pad=0.3", fc="white", ec="gray", alpha=0.8)
        )
        
        # Color points by confidence tier
        for tier_name, (lower, upper) in results['tier_metrics'][target_name].items():
            if tier_name not in ['high', 'medium', 'low']:
                continue
                
            tier_mask = (confidence >= lower) & (confidence < upper)
            if tier_mask.sum() > 0:
                plt.scatter(
                    confidence[tier_mask],
                    abs_errors[tier_mask],
                    label=f"{tier_name} tier ({tier_mask.sum()} samples)",
                    alpha=0.7
                )
        
        plt.legend()
        plt.tight_layout()
        
        # Save or show plot
        if output_dir:
            plt.savefig(Path(output_dir) / f"{target_name}_confidence_error.png")
            plt.close()
        else:
            plt.show()
            
    logger.info(f"Created confidence-error correlation plots for {len(results['confidence_correlation'])} targets")

def validate_config(df: pd.DataFrame, config: Dict[str, Any]) -> None:
    """
    Validate configuration parameters and raise warnings for potential issues.
    
    Args:
        df: Input DataFrame
        config: Configuration dictionary
    
    Raises:
        ValueError: If critical configuration issues are found
    """
    # Check for required columns
    if 'confidence_col' in config:
        confidence_col = config['confidence_col']
        if confidence_col not in df.columns:
            raise ValueError(f"Confidence column '{confidence_col}' not found in DataFrame")
    
    # Check confidence tier ranges
    if 'confidence_tiers' in config:
        tiers = config['confidence_tiers']
        
        # Check for overlapping ranges
        boundaries = []
        for tier_name, (lower, upper) in tiers.items():
            boundaries.append((lower, 'lower', tier_name))
            boundaries.append((upper, 'upper', tier_name))
        
        # Sort boundaries
        boundaries.sort()
        
        # Check for overlaps
        for i in range(1, len(boundaries) - 1):
            if boundaries[i][0] == boundaries[i+1][0] and boundaries[i][1] == 'upper' and boundaries[i+1][1] == 'lower':
                # This is a clean boundary between tiers
                continue
                
            if boundaries[i][1] == 'lower' and boundaries[i-1][1] == 'lower':
                logger.warning(f"Possible tier overlap: {boundaries[i-1][2]} and {boundaries[i][2]} both start at {boundaries[i][0]}")
            
            if boundaries[i][1] == 'upper' and boundaries[i+1][1] == 'upper':
                logger.warning(f"Possible tier overlap: {boundaries[i][2]} and {boundaries[i+1][2]} both end at {boundaries[i][0]}")
    
    # Check for sensible confidence threshold
    if 'confidence_threshold' in config:
        threshold = config['confidence_threshold']
        if threshold > 0.5:
            logger.warning(f"High confidence threshold ({threshold}) may exclude too many samples")
            
        # Calculate how many samples would be excluded
        if 'confidence_col' in config and config['confidence_col'] in df.columns:
            excluded = (df[config['confidence_col']] < threshold).sum()
            excluded_pct = excluded / len(df) * 100
            if excluded_pct > 25:
                logger.warning(f"Confidence threshold {threshold} would exclude {excluded} samples ({excluded_pct:.1f}%)")
    
    # Warn about very small training batches
    n_estimators = config.get('n_estimators', 0)
    if n_estimators < 100:
        logger.warning(f"Small number of estimators ({n_estimators}) may lead to poor model performance")
    
    logger.info("Configuration validation complete")

def export_results_summary(results: Dict[str, Any], output_dir: str = "results"):
    """Export comprehensive results summary for reproducibility.
    
    Args:
        results: Dictionary containing model results
        output_dir: Directory to save summaries
    """
    # Create output directory
    Path(output_dir).mkdir(parents=True, exist_ok=True)
    
    # Export metadata
    with open(Path(output_dir) / "metadata.json", "w") as f:
        # Convert any non-JSON serializable objects to strings
        metadata = {k: str(v) if not isinstance(v, (dict, list, str, int, float, bool, type(None))) 
                   else v for k, v in results['metadata'].items()}
        json.dump(metadata, f, indent=2)
    
    # Export overall metrics
    with open(Path(output_dir) / "metrics_summary.json", "w") as f:
        metrics = {
            'macro_r2': results.get('macro_r2', {}),
            'cv_macro_r2': results.get('cv_results', {}).get('macro_r2'),
            'confidence_correlation': {
                target: results['confidence_correlation'][target]['overall']['correlation']
                for target in results['confidence_correlation']
            }
        }
        json.dump(metrics, f, indent=2)
    
    # Export markdown summary for easy viewing
    with open(Path(output_dir) / "summary.md", "w") as f:
        f.write("# Medication Response Prediction Results\n\n")
        f.write(f"## Generated on {datetime.now().strftime('%Y-%m-%d %H:%M')}\n\n")
        
        # Add configuration
        f.write("## Configuration\n\n")
        f.write("```json\n")
        f.write(json.dumps(results['metadata']['config'], indent=2))
        f.write("\n```\n\n")
        
        # Add dataset info
        f.write("## Dataset Information\n\n")
        f.write(f"- Train set: {results['metadata']['train_size']} samples\n")
        f.write(f"- Test set: {results['metadata']['test_size']} samples\n")
        f.write(f"- Train date range: {results['metadata']['train_date_range'][0]} to {results['metadata']['train_date_range'][1]}\n")
        f.write(f"- Test date range: {results['metadata']['test_date_range'][0]} to {results['metadata']['test_date_range'][1]}\n\n")
        
        # Add overall metrics
        f.write("## Performance Metrics\n\n")
        f.write("### Overall\n\n")
        
        if isinstance(results.get('macro_r2'), dict) and 'weighted' in results['macro_r2']:
            f.write(f"- Macro R² (weighted): {results['macro_r2']['weighted']:.4f}\n")
            f.write(f"- Macro R² (unweighted): {results['macro_r2']['unweighted']:.4f}\n")
            f.write(f"- Improvement: {results['macro_r2']['improvement']:.4f}\n\n")
        else:
            f.write(f"- Macro R²: {results.get('macro_r2', 'N/A'):.4f}\n\n")
        
        # Add target-specific metrics
        f.write("### By Target\n\n")
        for target_name in results['tier_metrics']:
            if target_name in ['macro_r2', 'confidence_correlation', 'cv_results', 'tier_metrics', 'tier_importance']:
                continue
                
            f.write(f"#### {target_name}\n\n")
            
            # Handle both weighted and unweighted results
            if 'weighted' in results[target_name]:
                f.write(f"- Weighted R²: {results[target_name]['weighted']['metrics']['r2']:.4f}\n")
                f.write(f"- Unweighted R²: {results[target_name]['unweighted']['metrics']['r2']:.4f}\n")
                f.write(f"- Improvement: {results[target_name]['weight_improvement']['r2']:.4f}\n\n")
            else:
                f.write(f"- R²: {results[target_name]['metrics']['r2']:.4f}\n\n")
            
            # Add confidence tier metrics
            f.write("##### By Confidence Tier\n\n")
            for tier_name in ['high', 'medium', 'low']:
                if 'weighted' in results['tier_metrics'][target_name]:
                    if tier_name in results['tier_metrics'][target_name]['weighted']:
                        tier = results['tier_metrics'][target_name]['weighted'][tier_name]
                        f.write(f"- {tier_name.capitalize()} tier ({tier['count']} samples): R² = {tier['r2']:.4f}\n")
                else:
                    if tier_name in results['tier_metrics'][target_name]:
                        tier = results['tier_metrics'][target_name][tier_name]
                        f.write(f"- {tier_name.capitalize()} tier ({tier['count']} samples): R² = {tier['r2']:.4f}\n")
            
            f.write("\n")
        
        # Add confidence-error correlation
        f.write("## Confidence-Error Correlation\n\n")
        for target_name in results['confidence_correlation']:
            corr = results['confidence_correlation'][target_name]['overall']['correlation']
            p_val = results['confidence_correlation'][target_name]['overall']['p_value']
            f.write(f"- {target_name}: {corr:.4f} (p={p_val:.3e})\n")
        
        f.write("\n### Image Links\n\n")
        f.write("- [Confidence Performance](confidence_performance.png)\n")
        for target_name in results['confidence_correlation']:
            f.write(f"- [{target_name} Confidence-Error Correlation](plots/{target_name}_confidence_error.png)\n")
    
    logger.info(f"Exported comprehensive results summaries to {output_dir}/")

def save_shap_data(data_dict: Dict[str, Any], output_dir: str = "shap_data"):
    """Save feature matrix and targets for future SHAP analysis.
    
    Args:
        data_dict: Dictionary with prepared data
        output_dir: Directory to save data
    """
    Path(output_dir).mkdir(parents=True, exist_ok=True)
    
    # Save feature names
    with open(Path(output_dir) / "feature_names.json", "w") as f:
        json.dump(data_dict['feature_names'], f)
    
    # Save training data
    train_data = {
        'X': data_dict['train']['X'].to_dict(),
        'y': {
            target: data_dict['train']['y'][target].to_dict() 
            for target in data_dict['train']['y']
        }
    }
    
    with open(Path(output_dir) / "train_data.json", "w") as f:
        json.dump(train_data, f)
    
    # For larger datasets, use parquet format
    for dataset in ['train', 'val', 'test']:
        if dataset in data_dict:
            # Save X data
            pd.DataFrame(data_dict[dataset]['X']).to_parquet(
                Path(output_dir) / f"{dataset}_X.parquet"
            )
            
            # Save y data for each target
            for target in data_dict[dataset]['y']:
                pd.Series(data_dict[dataset]['y'][target]).to_parquet(
                    Path(output_dir) / f"{dataset}_y_{target}.parquet"
                )
    
    logger.info(f"Saved SHAP-ready data to {output_dir}/")

def get_output_directory(base_dir: str, config: Dict[str, Any]) -> Path:
    """
    Generate output directory path with optional timestamp and run name.
    
    Args:
        base_dir: Base directory name
        config: Configuration dictionary
        
    Returns:
        Path object for the output directory
    """
    if not config.get('timestamp_dirs', True):
        # Use base directory without timestamp
        return Path(base_dir)
    
    # Generate timestamp
    timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    
    # Add run name if provided
    if config.get('run_name'):
        dir_name = f"{timestamp}_{config['run_name']}"
    else:
        dir_name = timestamp
    
    # Create full path
    return Path(base_dir) / dir_name

def validate_pipeline(df: pd.DataFrame, config: Dict[str, Any]) -> Dict[str, Any]:
    """
    Validate pipeline configuration and data without training.
    
    Args:
        df: Input DataFrame
        config: Configuration dictionary
        
    Returns:
        Dictionary with validation results
    """
    # Initialize configuration with defaults
    trainer_config = {
        'n_estimators': 1000,
        'max_depth': 6,
        'learning_rate': 0.01,
        'subsample': 0.8,
        'colsample_bytree': 0.8,
        'random_state': 42,
        'early_stopping_rounds': 50,
        'confidence_threshold': 0.0,
        'use_sample_weights': True,
        'confidence_tiers': {
            'high': (0.7, 1.0),
            'medium': (0.4, 0.7),
            'low': (0.0, 0.4)
        },
        'calibrate_confidence': True,
        'calibration_method': 'minmax',
        'plot_normalize_counts': False,
        'compare_weights': True,
        'confidence_col': 'overall_confidence'
    }
    
    # Update with user config
    if config:
        trainer_config.update(config)
    
    # Validate configuration
    validate_config(df, trainer_config)
    
    # Validate feature schema
    missing_features = validate_feature_schema(df)
    
    # Get feature columns
    feature_cols = get_feature_columns(df)
    
    # Count feature categories
    feature_categories = set()
    for category in feature_cols:
        feature_categories.add(category)
    
    # Get confidence column
    confidence_col = trainer_config.get('confidence_col', 'overall_confidence')
    
    # Check if confidence column exists
    if confidence_col not in df.columns:
        logger.warning(f"Confidence column '{confidence_col}' not found in DataFrame")
    
    # Check for required target columns
    target_cols = [
        'activation_sedation_score',
        'emotional_blunting_restoration_score',
        'appetite_metabolic_score'
    ]
    missing_targets = [col for col in target_cols if col not in df.columns]
    if missing_targets:
        logger.warning(f"Missing target columns: {missing_targets}")
    
    # Validate temporal data
    if 'created_utc' in df.columns:
        # Check date range
        date_range = [df['created_utc'].min(), df['created_utc'].max()]
        date_span = date_range[1] - date_range[0]
        logger.info(f"Date range: {date_range[0]} to {date_range[1]} (span: {date_span} seconds)")
    else:
        logger.warning("No 'created_utc' column found for temporal validation")
        date_range = None
    
    # Return validation results
    return {
        'config_valid': True,  # We would have raised an exception if invalid
        'missing_features': missing_features,
        'feature_cols': {k: len(v) for k, v in feature_cols.items()},
        'feature_categories': list(feature_categories),
        'confidence_column_exists': confidence_col in df.columns,
        'missing_targets': missing_targets,
        'date_range': date_range if 'created_utc' in df.columns else None
    }

def print_results_summary(results: Dict[str, Any]) -> None:
    """
    Print a concise summary of model results.
    
    Args:
        results: Dictionary containing model results
    """
    print("\n=== MODEL PERFORMANCE SUMMARY ===")
    
    # Print overall metrics
    if isinstance(results.get('macro_r2'), dict) and 'weighted' in results['macro_r2']:
        print(f"\nMacro R² (weighted): {results['macro_r2']['weighted']:.4f}")
        print(f"Macro R² (unweighted): {results['macro_r2']['unweighted']:.4f}")
        print(f"Improvement: {results['macro_r2']['improvement']:.4f}")
    else:
        print(f"\nMacro R²: {results.get('macro_r2', 'N/A')}")
    
    # Print confidence-error correlations
    print("\nConfidence-error correlations:")
    for target, correlation in results['confidence_correlation'].items():
        corr_value = correlation['overall']['correlation']
        p_value = correlation['overall']['p_value']
        significant = "***" if p_value < 0.001 else "**" if p_value < 0.01 else "*" if p_value < 0.05 else ""
        print(f"  {target}: {corr_value:.3f}{significant}")
    
    if any("*" in line for line in [f"  {target}: {correlation['overall']['correlation']:.3f}" for target, correlation in results['confidence_correlation'].items()]):
        print("  Significance: * p<0.05, ** p<0.01, *** p<0.001")
    
    # Print per-target metrics
    print("\nTarget-specific performance:")
    for target_name in results['tier_metrics']:
        if target_name in ['macro_r2', 'confidence_correlation', 'cv_results', 'tier_metrics', 'tier_importance']:
            continue
            
        # Handle both weighted and unweighted results
        if 'weighted' in results[target_name]:
            r2_weighted = results[target_name]['weighted']['metrics']['r2']
            r2_unweighted = results[target_name]['unweighted']['metrics']['r2']
            improvement = results[target_name]['weight_improvement']['r2']
            print(f"  {target_name}: {r2_weighted:.3f} (weighted), {r2_unweighted:.3f} (unweighted), +{improvement:.3f}")
        else:
            r2 = results[target_name]['metrics']['r2']
            print(f"  {target_name}: {r2:.3f}")
    
    # Print high confidence tier performance
    print("\nHigh confidence tier performance:")
    for target_name in results['tier_metrics']:
        if target_name in ['macro_r2', 'confidence_correlation', 'cv_results', 'tier_metrics', 'tier_importance']:
            continue
            
        if 'weighted' in results['tier_metrics'][target_name]:
            if 'high' in results['tier_metrics'][target_name]['weighted']:
                tier = results['tier_metrics'][target_name]['weighted']['high']
                print(f"  {target_name}: {tier['r2']:.3f} ({tier['count']} samples, {tier['percentage']:.1f}%)")
        else:
            if 'high' in results['tier_metrics'][target_name]:
                tier = results['tier_metrics'][target_name]['high']
                print(f"  {target_name}: {tier['r2']:.3f} ({tier['count']} samples, {tier['percentage']:.1f}%)")
    
    # Print key feature importance
    print("\nTop features (weighted):")
    for target_name in results:
        if target_name in ['macro_r2', 'confidence_correlation', 'cv_results', 'tier_metrics', 'tier_importance']:
            continue
            
        if 'weighted' in results[target_name]:
            top_features = results[target_name]['weighted']['top_features'][:3]
            feature_names = [f"{name} ({importance:.2f})" for name, importance in top_features]
            print(f"  {target_name}: {', '.join(feature_names)}")
        else:
            top_features = results[target_name]['top_features'][:3]
            feature_names = [f"{name} ({importance:.2f})" for name, importance in top_features]
            print(f"  {target_name}: {', '.join(feature_names)}")
    
    print("\n=================================")

def compress_results(results_dir: Path, remove_original: bool = False) -> str:
    """
    Compress results directory into a zip archive.
    
    Args:
        results_dir: Path to results directory
        remove_original: Whether to remove the original directory after compression
        
    Returns:
        Path to the compressed archive
    """
    try:
        # Create archive
        archive_path = str(results_dir) + '.zip'
        logger.info(f"Compressing results to {archive_path}")
        
        shutil.make_archive(
            str(results_dir),  # Archive name (without extension)
            'zip',             # Archive format
            results_dir.parent,  # Root directory
            results_dir.name    # Directory to archive
        )
        
        # Remove original if requested
        if remove_original:
            logger.info(f"Removing original directory {results_dir}")
            shutil.rmtree(results_dir)
        
        return archive_path
    except Exception as e:
        logger.error(f"Error compressing results: {e}")
        return None

def parse_args():
    """Parse command line arguments for the training pipeline."""
    parser = argparse.ArgumentParser(description="Train medication response prediction models")
    parser.add_argument("--data", required=True, help="Path to input data")
    parser.add_argument("--config", help="Path to configuration JSON file")
    parser.add_argument("--output-dir", help="Base directory for results")
    parser.add_argument("--run-name", help="Name for this training run (will be combined with timestamp)")
    parser.add_argument("--no-timestamp", action="store_true", help="Disable timestamp in output directories")
    parser.add_argument("--dry-run", action="store_true", help="Validate configuration and data without training")
    parser.add_argument("--compress", action="store_true", help="Compress results directory after training")
    parser.add_argument("--remove-original", action="store_true", help="Remove original directories after compression")
    
    # Parse arguments
    args = parser.parse_args()
    
    # Load configuration
    config = {}
    if args.config:
        with open(args.config, 'r') as f:
            config = json.load(f)
    
    # Override with command line arguments
    if args.output_dir:
        config['output_dir'] = args.output_dir
    if args.run_name:
        config['run_name'] = args.run_name
    if args.no_timestamp:
        config['timestamp_dirs'] = False
    if args.compress:
        config['compress_results'] = True
    if args.remove_original:
        config['remove_after_compress'] = True
    
    return args, config

def main():
    """Main entry point for the training pipeline."""
    args, config = parse_args()
    
    # Load data
    logger.info(f"Loading data from {args.data}")
    df = pd.read_parquet(args.data)
    logger.info(f"Loaded {len(df)} samples with {len(df.columns)} columns")
    
    if args.dry_run:
        # Run validation only
        logger.info("Dry run mode - validating configuration and data only")
        results = validate_pipeline(df, config)
        
        # Print validation summary
        print("\nDry run completed successfully!")
        print(f"Data shape: {df.shape}")
        print(f"Missing features: {len(results['missing_features'])}")
        if results['missing_features']:
            print(f"  {', '.join(results['missing_features'][:5])}" + 
                  (f" and {len(results['missing_features'])-5} more..." if len(results['missing_features']) > 5 else ""))
        print(f"Validated {len(results['feature_cols'])} features across {len(results['feature_categories'])} categories")
        print("\nTo run full training, remove the --dry-run flag")
        return
    
    # Regular training run
    results = train_models(df, config)
    
    # Print results summary
    print_results_summary(results)
    
    print(f"\nTraining completed successfully!")
    print(f"Results saved to: {results['metadata']['output_paths']['results_dir']}")
    print(f"SHAP data saved to: {results['metadata']['output_paths']['shap_dir']}")
    
    # Print archive path if results were compressed
    if 'archive' in results['metadata']['output_paths']:
        print(f"Compressed archive: {results['metadata']['output_paths']['archive']}")

def train_models(df: pd.DataFrame, config: Dict[str, Any] = None) -> Dict[str, Any]:
    """Train models with confidence-aware training.
    
    Args:
        df: Input DataFrame with features and targets
        config: Optional configuration dictionary
        
    Returns:
        Dictionary containing model results, metrics, and evaluations
        
    Note:
        - Supports both weighted and unweighted training
        - Performs confidence tier analysis
        - Includes cross-validation
        - Generates performance visualizations
    """
    logger.info("Training models with confidence-aware approach")
    
    # Initialize trainer with configuration
    trainer_config = {
        'n_estimators': 1000,
        'max_depth': 6,
        'learning_rate': 0.01,
        'subsample': 0.8,
        'colsample_bytree': 0.8,
        'random_state': 42,
        'early_stopping_rounds': 50,
        'confidence_threshold': 0.0,
        'use_sample_weights': True,
        'confidence_tiers': {
            'high': (0.7, 1.0),
            'medium': (0.4, 0.7),
            'low': (0.0, 0.4)
        },
        'calibrate_confidence': True,
        'calibration_method': 'minmax',
        'plot_normalize_counts': False,
        'compare_weights': True,
        'confidence_col': 'overall_confidence',
        'output_dir': 'results',
        'shap_dir': 'shap_data',
        'timestamp_dirs': True,
        'run_name': None,
        'compress_results': False,
        'remove_after_compress': False
    }
    
    # Update with user config if provided
    if config:
        trainer_config.update(config)
    
    # Generate output directories
    results_dir = get_output_directory(
        trainer_config.get('output_dir', 'results'), 
        trainer_config
    )
    shap_dir = get_output_directory(
        trainer_config.get('shap_dir', 'shap_data'),
        trainer_config
    )
    
    # Create top-level directories if needed
    Path(trainer_config.get('output_dir', 'results')).mkdir(exist_ok=True)
    Path(trainer_config.get('shap_dir', 'shap_data')).mkdir(exist_ok=True)
    
    # Log output locations
    logger.info(f"Results will be saved to: {results_dir}")
    logger.info(f"SHAP data will be saved to: {shap_dir}")
    
    # Validate configuration
    validate_config(df, trainer_config)
    
    # Validate feature schema
    missing_features = validate_feature_schema(df)
    
    # Get feature columns
    feature_cols = get_feature_columns(df)
    
    # Get confidence column from config
    confidence_col = trainer_config.get('confidence_col', 'overall_confidence')
    
    # Calibrate confidence scores early
    if trainer_config['calibrate_confidence']:
        df = calibrate_dataframe_confidence(df)
    
    # Handle missing values
    df_clean = handle_missing_values(df, feature_cols)
    
    trainer = ConfidenceAwareTrainer(trainer_config)
    
    # Combine all features
    all_features = []
    for category, cols in feature_cols.items():
        all_features.extend(cols)
    
    # Remove duplicates
    feature_cols = list(set(all_features))
    
    # Log features
    logger.info(f"Using {len(feature_cols)} features:")
    for category, cols in feature_cols.items():
        logger.info(f"  {category} features: {len(cols)}")
    
    # Split data temporally
    train_df, test_df = split_temporal_data(df_clean)
    
    # Prepare data
    data_dict = trainer.prepare_data(
        train_df, 
        feature_cols,
        confidence_col=confidence_col
    )
    
    # Save SHAP-ready data
    save_shap_data(data_dict, output_dir=shap_dir)
    
    # Train models with weight comparison
    results = trainer.train_all_models(data_dict, compare_weights=trainer_config['compare_weights'])
    
    # Evaluate confidence tiers
    tier_metrics = trainer.evaluate_confidence_tiers(
        data_dict, 
        results, 
        compare_weights=trainer_config['compare_weights']
    )
    results['tier_metrics'] = tier_metrics
    
    # Plot confidence tier performance
    plot_path = results_dir / "confidence_performance.png"
    trainer.plot_confidence_performance(
        tier_metrics, 
        output_path=str(plot_path),
        compare_weights=trainer_config['compare_weights']
    )
    
    # Plot confidence-error correlation
    plots_dir = results_dir / "plots"
    plot_confidence_error_correlation(
        data_dict,
        results,
        output_dir=plots_dir
    )
    
    # Run cross-validation
    cv_results = trainer.run_cross_validation(train_df, feature_cols)
    results['cv_results'] = cv_results
    
    # Analyze confidence-error correlation
    correlation_results = trainer.analyze_confidence_error_correlation(data_dict, results)
    results['confidence_correlation'] = correlation_results
    
    # Get tier-specific feature importance
    tier_importance = trainer.get_tier_specific_feature_importance(data_dict, results)
    results['tier_importance'] = tier_importance
    
    # Add metadata
    results['metadata'] = {
        'feature_schema': {k: v for k, v in feature_cols.items()},
        'missing_features': missing_features,
        'train_size': len(train_df),
        'test_size': len(test_df),
        'train_date_range': [train_df['created_utc'].min(), train_df['created_utc'].max()],
        'test_date_range': [test_df['created_utc'].min(), test_df['created_utc'].max()],
        'config': trainer_config,
        'output_paths': {
            'results_dir': str(results_dir),
            'shap_dir': str(shap_dir),
            'plots_dir': str(plots_dir),
            'timestamp': datetime.now().isoformat()
        }
    }
    
    # Export comprehensive results summary
    export_results_summary(results, output_dir=results_dir)
    
    # Compress results if requested
    if trainer_config.get('compress_results', False):
        archive_path = compress_results(
            results_dir, 
            remove_original=trainer_config.get('remove_after_compress', False)
        )
        if archive_path:
            results['metadata']['output_paths']['archive'] = archive_path
    
    return results

def build_features(df: pd.DataFrame, config: Dict[str, Any], debug_dir: Optional[str] = None) -> Tuple[pd.DataFrame, Optional[Dict]]:
    """
    Build features for model training.
    
    Args:
        df: DataFrame with raw data
        config: Configuration dictionary
        debug_dir: Optional directory to store detailed evidence
        
    Returns:
        Tuple of:
        - DataFrame with optimized features for Parquet storage
        - Optional dictionary containing detailed evidence for debug files
    """
    # Initialize response attributor
    attributor = ResponseAttributor(
        model_name=config.get('model_name'),
        debug=config.get('debug', False)
    )
    
    # Build enhanced features
    enhanced_df, debug_data = build_enhanced_features(
        df,
        attributor,
        debug_dir=debug_dir
    )
    
    # Optimize data types for Parquet storage
    enhanced_df = optimize_dtypes(enhanced_df)
    
    return enhanced_df, debug_data

def optimize_dtypes(df: pd.DataFrame) -> pd.DataFrame:
    """
    Optimize data types for Parquet storage.
    
    Args:
        df: DataFrame to optimize
        
    Returns:
        DataFrame with optimized data types
    """
    # Create a copy to avoid modifying the original
    result_df = df.copy()
    
    # Optimize numeric columns
    for col in result_df.select_dtypes(include=['float64']).columns:
        # Check if column can be converted to float32
        if result_df[col].min() >= np.finfo(np.float32).min and result_df[col].max() <= np.finfo(np.float32).max:
            result_df[col] = result_df[col].astype('float32')
    
    # Optimize integer columns
    for col in result_df.select_dtypes(include=['int64']).columns:
        # Check if column can be converted to int32
        if result_df[col].min() >= np.iinfo(np.int32).min and result_df[col].max() <= np.iinfo(np.int32).max:
            result_df[col] = result_df[col].astype('int32')
    
    # Optimize boolean columns
    for col in result_df.select_dtypes(include=['bool']).columns:
        result_df[col] = result_df[col].astype('bool')
    
    # Optimize categorical columns
    for col in result_df.select_dtypes(include=['object']).columns:
        if result_df[col].nunique() / len(result_df) < 0.5:  # If less than 50% unique values
            result_df[col] = result_df[col].astype('category')
    
    return result_df

def save_features(df: pd.DataFrame, output_path: str) -> None:
    """
    Save features to Parquet file with optimized settings.
    
    Args:
        df: DataFrame to save
        output_path: Path to save the Parquet file
    """
    # Create output directory if it doesn't exist
    output_dir = Path(output_path).parent
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Save to Parquet with optimized settings
    df.to_parquet(
        output_path,
        engine='pyarrow',
        compression='snappy',  # Good balance of compression and speed
        index=False,
        coerce_timestamps='ms',
        allow_truncated_timestamps=True
    )
    
    logger.info(f"Saved features to {output_path}")

if __name__ == "__main__":
    main() 
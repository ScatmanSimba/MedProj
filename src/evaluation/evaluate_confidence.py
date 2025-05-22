"""
Evaluation module for analyzing confidence vs. accuracy relationship.

This module provides tools to evaluate how well confidence scores correlate
with prediction accuracy across different confidence tiers.
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import r2_score, mean_squared_error
from scipy.stats import pearsonr, spearmanr
import json
import xgboost as xgb
from pathlib import Path
import argparse
import logging
from typing import Dict, List, Tuple, Set, Any
from sklearn.isotonic import IsotonicRegression

# Configure logging
logging.basicConfig(
    level=logging.INFO, 
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger("evaluate_confidence")

def load_models_and_data(model_dir: str, test_data: str) -> Tuple[Dict, pd.DataFrame]:
    """
    Load trained models and test data.
    
    Args:
        model_dir: Directory containing trained models
        test_data: Path to test data parquet file
        
    Returns:
        Tuple of (models_dict, test_df)
    """
    logger.info(f"Loading models from {model_dir}")
    model_dir = Path(model_dir)
    
    # Find model files
    model_files = list(model_dir.glob("*_model.json"))
    
    if not model_files:
        raise ValueError(f"No model files found in {model_dir}")
    
    # Load models
    models = {}
    for model_file in model_files:
        target_name = model_file.stem.replace("_model", "")
        model = xgb.Booster()
        model.load_model(str(model_file))
        models[target_name] = model
        logger.info(f"Loaded model for {target_name}")
    
    # Load test data
    logger.info(f"Loading test data from {test_data}")
    test_df = pd.read_parquet(test_data)
    logger.info(f"Loaded {len(test_df)} test records")
    
    return models, test_df

def get_feature_columns(model_dir: Path, test_df: pd.DataFrame, target_cols: List[str], confidence_col: str) -> List[str]:
    """
    Get feature columns, prioritizing saved feature names from training.
    
    Args:
        model_dir: Directory containing model files
        test_df: Test dataframe
        target_cols: List of target column names
        confidence_col: Column with confidence scores
        
    Returns:
        List of feature column names
    """
    # Try to load feature names from saved file
    feature_names_file = model_dir / "feature_names.json"
    if feature_names_file.exists():
        with open(feature_names_file, 'r') as f:
            feature_cols = json.load(f)
        logger.info(f"Loaded {len(feature_cols)} feature names from {feature_names_file}")
        return feature_cols
    
    # Fallback to feature importance files
    feature_importance_files = list(model_dir.glob("*_importance.json"))
    if feature_importance_files:
        feature_cols = []
        for importance_file in feature_importance_files:
            with open(importance_file, 'r') as f:
                importance = json.load(f)
                feature_cols.extend(importance.keys())
        feature_cols = list(set(feature_cols))
        logger.info(f"Extracted {len(feature_cols)} feature names from importance files")
        return feature_cols
    
    # Last resort: heuristic approach
    exclude_cols = target_cols + [confidence_col, 'id', 'post_id', 'user_id', 'date', 'text', 'title', 'selftext']
    feature_cols = [col for col in test_df.columns if col not in exclude_cols and not col.startswith('med_responses')]
    logger.info(f"Using heuristic approach to identify {len(feature_cols)} features")
    return feature_cols

def validate_features(model: xgb.Booster, feature_cols: List[str], test_df: pd.DataFrame) -> List[str]:
    """
    Validate and align feature names between model and test data.
    
    Args:
        model: XGBoost model
        feature_cols: List of feature column names
        test_df: Test dataframe
        
    Returns:
        List of validated feature column names
    """
    model_features = model.feature_names
    if not model_features:
        logger.warning("Model has no feature names stored")
        return feature_cols
    
    if set(model_features) != set(feature_cols):
        logger.warning(f"Feature mismatch! Model expected {len(model_features)} features, got {len(feature_cols)}")
        # Align features if possible
        aligned_features = [f for f in model_features if f in test_df.columns]
        missing_features = set(model_features) - set(aligned_features)
        if missing_features:
            logger.warning(f"Missing features in test data: {missing_features}")
        return aligned_features
    
    return feature_cols

def get_adaptive_confidence_bins(confidence_values: np.ndarray, n_bins: int = 10) -> List[Tuple[float, float]]:
    """
    Create adaptive confidence bins based on data distribution.
    
    Args:
        confidence_values: Array of confidence scores
        n_bins: Number of desired bins
        
    Returns:
        List of (lower, upper) bin edges
    """
    # Check if confidence scores are tightly bunched
    confidence_range = confidence_values.max() - confidence_values.min()
    if confidence_range < 0.5:  # If range is less than 0.5, use quantiles
        logger.info("Confidence scores tightly bunched, using quantile-based bins")
        quantiles = np.linspace(0, 100, n_bins + 1)
        bin_edges = np.percentile(confidence_values, quantiles)
    else:
        # Use regular linear spacing
        bin_edges = np.linspace(confidence_values.min(), confidence_values.max(), n_bins + 1)
    
    return [(bin_edges[i], bin_edges[i+1]) for i in range(len(bin_edges)-1)]

def evaluate_confidence_vs_accuracy(
    models: Dict, 
    test_df: pd.DataFrame, 
    target_cols: List[str],
    confidence_col: str = 'overall_confidence',
    output_dir: str = None,
    min_bin_samples: int = 10
) -> Dict:
    """
    Evaluate relationship between confidence and prediction accuracy.
    
    Args:
        models: Dictionary of trained models
        test_df: Test dataframe
        target_cols: List of target column names
        confidence_col: Column with confidence scores
        output_dir: Directory to save output figures
        min_bin_samples: Minimum number of samples required in a bin for analysis
        
    Returns:
        Dictionary with evaluation results
    """
    logger.info("Evaluating confidence vs accuracy relationship")
    
    # Create output directory if provided
    if output_dir:
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
    
    # Get feature columns
    model_dir = Path(output_dir).parent if output_dir else Path(".")
    feature_cols = get_feature_columns(model_dir, test_df, target_cols, confidence_col)
    
    # Make predictions
    predictions = {}
    
    for target_name, model in models.items():
        if target_name not in target_cols:
            continue
        
        # Validate and align features
        validated_features = validate_features(model, feature_cols, test_df)
        
        # Create DMatrix with feature names
        dtest = xgb.DMatrix(test_df[validated_features], feature_names=validated_features)
        
        # Predict
        preds = model.predict(dtest)
        predictions[target_name] = preds
    
    # Calculate errors
    errors = {}
    squared_errors = {}
    normalized_errors = {}
    
    for target_name in target_cols:
        if target_name not in predictions:
            continue
        
        preds = predictions[target_name]
        true_values = test_df[target_name]
        
        # Calculate squared errors
        squared_errors[target_name] = (true_values - preds) ** 2
        
        # Calculate absolute errors
        errors[target_name] = np.abs(true_values - preds)
        
        # Calculate normalized errors
        target_std = test_df[target_name].std()
        normalized_errors[target_name] = errors[target_name] / target_std
    
    # Create adaptive confidence bins
    confidence_values = test_df[confidence_col].values
    bin_edges = get_adaptive_confidence_bins(confidence_values)
    
    # Calculate metrics by confidence bin
    bin_metrics = {}
    
    for i, (lower, upper) in enumerate(bin_edges):
        bin_mask = (test_df[confidence_col] >= lower) & (test_df[confidence_col] < upper)
        bin_count = bin_mask.sum()
        
        # Skip bins with too few samples
        if bin_count < min_bin_samples:
            logger.debug(f"Skipping bin {lower:.2f}-{upper:.2f} with only {bin_count} samples")
            continue
        
        bin_name = f"{lower:.2f}-{upper:.2f} (n={bin_count})"
        bin_metrics[bin_name] = {
            'count': bin_count,
            'percentage': bin_count / len(test_df) * 100,
            'mean_confidence': test_df.loc[bin_mask, confidence_col].mean(),
            'r2': {},
            'rmse': {},
            'mean_error': {},
            'median_error': {},
            'normalized_error': {},
            'median_normalized_error': {}
        }
        
        for target_name in target_cols:
            if target_name not in predictions:
                continue
            
            bin_preds = predictions[target_name][bin_mask]
            bin_true = test_df.loc[bin_mask, target_name]
            
            if len(bin_preds) > 1:
                # Calculate R²
                bin_r2 = r2_score(bin_true, bin_preds)
                bin_metrics[bin_name]['r2'][target_name] = bin_r2
                
                # Calculate RMSE
                bin_rmse = np.sqrt(mean_squared_error(bin_true, bin_preds))
                bin_metrics[bin_name]['rmse'][target_name] = bin_rmse
                
                # Calculate mean absolute error
                bin_mae = errors[target_name][bin_mask].mean()
                bin_metrics[bin_name]['mean_error'][target_name] = bin_mae
                
                # Calculate median absolute error
                bin_median_error = np.median(errors[target_name][bin_mask])
                bin_metrics[bin_name]['median_error'][target_name] = bin_median_error
                
                # Calculate normalized error
                bin_norm_error = normalized_errors[target_name][bin_mask].mean()
                bin_metrics[bin_name]['normalized_error'][target_name] = bin_norm_error
                
                # Calculate median normalized error
                bin_median_norm_error = np.median(normalized_errors[target_name][bin_mask])
                bin_metrics[bin_name]['median_normalized_error'][target_name] = bin_median_norm_error
    
    # Create correlation analysis
    correlation_metrics = {}
    
    for target_name in target_cols:
        if target_name not in predictions:
            continue
        
        # Calculate correlation between confidence and error
        confidence = test_df[confidence_col]
        error = errors[target_name]
        squared_error = squared_errors[target_name]
        norm_error = normalized_errors[target_name]
        
        # Calculate Pearson correlations with p-values
        confidence_error_corr, confidence_error_p = pearsonr(confidence, error)
        confidence_squared_error_corr, confidence_squared_error_p = pearsonr(confidence, squared_error)
        confidence_norm_error_corr, confidence_norm_error_p = pearsonr(confidence, norm_error)
        
        # Calculate Spearman correlations with p-values
        confidence_error_spearman, confidence_error_spearman_p = spearmanr(confidence, error)
        confidence_squared_error_spearman, confidence_squared_error_spearman_p = spearmanr(confidence, squared_error)
        confidence_norm_error_spearman, confidence_norm_error_spearman_p = spearmanr(confidence, norm_error)
        
        # Add significance indicators
        def get_significance(p_value):
            if p_value < 0.001:
                return " (p<0.001) ***"
            elif p_value < 0.01:
                return " (p<0.01) **"
            elif p_value < 0.05:
                return " (p<0.05) *"
            return " (ns)"
        
        correlation_metrics[target_name] = {
            'confidence_error_correlation': confidence_error_corr,
            'confidence_error_p_value': confidence_error_p,
            'confidence_squared_error_correlation': confidence_squared_error_corr,
            'confidence_squared_error_p_value': confidence_squared_error_p,
            'confidence_normalized_error_correlation': confidence_norm_error_corr,
            'confidence_normalized_error_p_value': confidence_norm_error_p,
            'confidence_error_spearman': confidence_error_spearman,
            'confidence_error_spearman_p_value': confidence_error_spearman_p,
            'confidence_squared_error_spearman': confidence_squared_error_spearman,
            'confidence_squared_error_spearman_p_value': confidence_squared_error_spearman_p,
            'confidence_normalized_error_spearman': confidence_norm_error_spearman,
            'confidence_normalized_error_spearman_p_value': confidence_norm_error_spearman_p
        }
        
        logger.info(f"{target_name} confidence-error Pearson correlation: {confidence_error_corr:.4f}{get_significance(confidence_error_p)}")
        logger.info(f"{target_name} confidence-error Spearman correlation: {confidence_error_spearman:.4f}{get_significance(confidence_error_spearman_p)}")
        logger.info(f"{target_name} confidence-squared error Pearson correlation: {confidence_squared_error_corr:.4f}{get_significance(confidence_squared_error_p)}")
        logger.info(f"{target_name} confidence-squared error Spearman correlation: {confidence_squared_error_spearman:.4f}{get_significance(confidence_squared_error_spearman_p)}")
        logger.info(f"{target_name} confidence-normalized error Pearson correlation: {confidence_norm_error_corr:.4f}{get_significance(confidence_norm_error_p)}")
        logger.info(f"{target_name} confidence-normalized error Spearman correlation: {confidence_norm_error_spearman:.4f}{get_significance(confidence_norm_error_spearman_p)}")
    
    # Create visualizations
    if output_dir:
        # Set consistent style
        plt.style.use('seaborn')
        
        # 1. Confidence distribution
        plt.figure(figsize=(10, 6))
        sns.histplot(test_df[confidence_col], bins=20, kde=True)
        plt.title("Confidence Score Distribution")
        plt.xlabel("Confidence Score")
        plt.ylabel("Count")
        plt.savefig(output_dir / "confidence_distribution_plot.png")
        plt.close()
        
        # 2. Error vs Confidence scatter plot
        for target_name in target_cols:
            if target_name not in predictions:
                continue
            
            plt.figure(figsize=(10, 6))
            sns.scatterplot(x=test_df[confidence_col], y=errors[target_name], alpha=0.5)
            plt.title(f"{target_name} - Error vs Confidence")
            plt.xlabel("Confidence Score")
            plt.ylabel("Absolute Error")
            
            # Add trend line
            sns.regplot(x=test_df[confidence_col], y=errors[target_name], scatter=False, color='red')
            
            plt.savefig(output_dir / f"confidence_{target_name}_error_plot.png")
            plt.close()
        
        # 3. Metrics by confidence bin
        for metric_name in ['r2', 'rmse', 'mean_error', 'median_error', 'normalized_error', 'median_normalized_error']:
            plt.figure(figsize=(12, 6))
            
            # Set consistent y-axis limits
            if metric_name == 'r2':
                plt.ylim(0, 1)  # R² typically ranges from 0 to 1
            elif metric_name in ['rmse', 'mean_error', 'median_error']:
                # Find max value across all targets and bins
                max_val = max(max(metrics[metric_name].values()) 
                            for metrics in bin_metrics.values() 
                            if metric_name in metrics)
                plt.ylim(0, max_val * 1.1)  # Add 10% padding
            
            for target_name in target_cols:
                if target_name not in predictions:
                    continue
                
                bin_names = []
                bin_values = []
                
                for bin_name, metrics in bin_metrics.items():
                    if target_name in metrics[metric_name]:
                        bin_names.append(bin_name)
                        bin_values.append(metrics[metric_name][target_name])
                
                if bin_values:
                    plt.plot(bin_names, bin_values, marker='o', label=target_name)
            
            plt.title(f"{metric_name.upper()} by Confidence Bin")
            plt.xlabel("Confidence Bin")
            plt.ylabel(metric_name.upper())
            plt.xticks(rotation=45)
            plt.legend()
            plt.tight_layout()
            plt.savefig(output_dir / f"confidence_{metric_name}_bin_plot.png")
            plt.close()
    
    # Prepare results
    results = {
        'bin_metrics': bin_metrics,
        'correlation_metrics': correlation_metrics
    }
    
    # Save results if output directory provided
    if output_dir:
        # Save JSON results
        with open(output_dir / "confidence_evaluation.json", 'w') as f:
            # Convert any numpy values to Python types for JSON serialization
            json_results = {}
            for key, value in results.items():
                if isinstance(value, dict):
                    json_results[key] = json.loads(pd.DataFrame([value]).to_json(orient='records'))[0]
                else:
                    json_results[key] = value
            
            json.dump(json_results, f, indent=2)
        
        # Save CSV results
        # 1. Flatten and save bin metrics
        bin_data = []
        for bin_name, metrics in bin_metrics.items():
            for target_name in target_cols:
                if target_name not in metrics['r2']:
                    continue
                
                bin_data.append({
                    'bin': bin_name,
                    'target': target_name,
                    'count': metrics['count'],
                    'percentage': metrics['percentage'],
                    'mean_confidence': metrics['mean_confidence'],
                    'r2': metrics['r2'][target_name],
                    'rmse': metrics['rmse'][target_name],
                    'mean_error': metrics['mean_error'][target_name],
                    'median_error': metrics['median_error'][target_name],
                    'normalized_error': metrics['normalized_error'][target_name],
                    'median_normalized_error': metrics['median_normalized_error'][target_name]
                })
        
        bin_df = pd.DataFrame(bin_data)
        bin_df.to_csv(output_dir / "confidence_bin_metrics.csv", index=False)
        
        # 2. Save correlation metrics
        corr_df = pd.DataFrame.from_dict(correlation_metrics, orient='index')
        corr_df.to_csv(output_dir / "confidence_correlations.csv")
        
        # 3. Save raw error data in long format
        error_data = []
        for target_name in target_cols:
            if target_name not in predictions:
                continue
            
            for i in range(len(test_df)):
                error_data.append({
                    'target': target_name,
                    'index': i,
                    'confidence': test_df[confidence_col].iloc[i],
                    'prediction': predictions[target_name][i],
                    'true_value': test_df[target_name].iloc[i],
                    'error': errors[target_name][i],
                    'squared_error': squared_errors[target_name][i],
                    'normalized_error': normalized_errors[target_name][i]
                })
        
        error_df = pd.DataFrame(error_data)
        error_df.to_csv(output_dir / "confidence_error_data.csv", index=False)
    
    return results

def main():
    """Main function."""
    parser = argparse.ArgumentParser(description="Evaluate confidence vs accuracy relationship")
    parser.add_argument("--model-dir", required=True, help="Directory containing trained models")
    parser.add_argument("--test-data", required=True, help="Path to test data parquet file")
    parser.add_argument("--output-dir", help="Directory to save output figures")
    parser.add_argument("--confidence-col", default="overall_confidence", help="Column with confidence scores")
    parser.add_argument("--min-bin-samples", type=int, default=10, help="Minimum samples required in a confidence bin")
    
    args = parser.parse_args()
    
    try:
        # Load models and data
        models, test_df = load_models_and_data(args.model_dir, args.test_data)
        
        # Define target columns
        target_cols = [
            'activation_sedation_score',
            'emotional_blunting_restoration_score',
            'appetite_metabolic_score'
        ]
        
        # Evaluate confidence vs accuracy
        results = evaluate_confidence_vs_accuracy(
            models, 
            test_df, 
            target_cols,
            confidence_col=args.confidence_col,
            output_dir=args.output_dir,
            min_bin_samples=args.min_bin_samples
        )
        
        logger.info("Evaluation completed successfully")
        
    except Exception as e:
        logger.error(f"Error in evaluation: {e}", exc_info=True)
        raise

if __name__ == "__main__":
    main()

class ConfidenceEvaluator:
    """Evaluate and tune confidence calibration."""
    
    def __init__(self, config: Dict[str, Any]):
        """Initialize the ConfidenceEvaluator.
        
        Args:
            config: Configuration dictionary
        """
        self.config = config
        self.calibration_models = {}
        self.error_metrics = {}
    
    def tune_attribution_confidence(self, attributor: 'ResponseAttributor', 
                                  gold_data: pd.DataFrame,
                                  max_iterations: int = 5) -> Dict[str, Any]:
        """Tune ResponseAttributor confidence calibration using error-driven approach.
        
        Args:
            attributor: ResponseAttributor instance
            gold_data: DataFrame with gold standard annotations
            max_iterations: Maximum number of tuning iterations
            
        Returns:
            Dictionary with tuning results
        """
        results = {
            'iterations': [],
            'error_metrics': [],
            'calibration_params': []
        }
        
        # Initialize calibration models
        self.calibration_models = {
            'symptom': IsotonicRegression(out_of_bounds='clip'),
            'temporal': IsotonicRegression(out_of_bounds='clip'),
            'causal': IsotonicRegression(out_of_bounds='clip'),
            'emoji': IsotonicRegression(out_of_bounds='clip')
        }
        
        # Initial evaluation
        current_metrics = self._evaluate_attribution_confidence(attributor, gold_data)
        results['iterations'].append(0)
        results['error_metrics'].append(current_metrics)
        results['calibration_params'].append(self._get_current_params(attributor))
        
        # Tuning loop
        for iteration in range(1, max_iterations + 1):
            # Get attribution errors
            errors = self._get_attribution_errors(attributor, gold_data)
            
            # Update calibration models
            self._update_calibration_models(errors)
            
            # Apply calibration to attributor
            self._apply_calibration_to_attributor(attributor)
            
            # Evaluate new performance
            new_metrics = self._evaluate_attribution_confidence(attributor, gold_data)
            
            # Check for improvement
            if not self._is_improvement(new_metrics, current_metrics):
                logger.info(f"No improvement in iteration {iteration}. Stopping.")
                break
            
            # Update results
            results['iterations'].append(iteration)
            results['error_metrics'].append(new_metrics)
            results['calibration_params'].append(self._get_current_params(attributor))
            
            current_metrics = new_metrics
        
        return results
    
    def _evaluate_attribution_confidence(self, attributor: 'ResponseAttributor',
                                       gold_data: pd.DataFrame) -> Dict[str, float]:
        """Evaluate attribution confidence against gold standard.
        
        Args:
            attributor: ResponseAttributor instance
            gold_data: DataFrame with gold standard annotations
            
        Returns:
            Dictionary with error metrics
        """
        metrics = {}
        
        # Get predictions
        predictions = attributor.attribute_responses(gold_data['text'].tolist())
        
        # Calculate metrics for each component
        for component in ['symptom', 'temporal', 'causal', 'emoji']:
            if f'{component}_confidence' in predictions.columns:
                metrics[component] = {
                    'brier_score': self._calculate_brier_score(
                        predictions[f'{component}_confidence'],
                        gold_data[f'{component}_gold']
                    ),
                    'calibration_error': self._calculate_calibration_error(
                        predictions[f'{component}_confidence'],
                        gold_data[f'{component}_gold']
                    ),
                    'confidence_correlation': self._calculate_confidence_correlation(
                        predictions[f'{component}_confidence'],
                        gold_data[f'{component}_gold']
                    )
                }
        
        return metrics
    
    def _get_attribution_errors(self, attributor: 'ResponseAttributor',
                              gold_data: pd.DataFrame) -> Dict[str, pd.DataFrame]:
        """Get attribution errors for calibration.
        
        Args:
            attributor: ResponseAttributor instance
            gold_data: DataFrame with gold standard annotations
            
        Returns:
            Dictionary mapping components to their error DataFrames
        """
        errors = {}
        
        # Get predictions
        predictions = attributor.attribute_responses(gold_data['text'].tolist())
        
        # Calculate errors for each component
        for component in ['symptom', 'temporal', 'causal', 'emoji']:
            if f'{component}_confidence' in predictions.columns:
                errors[component] = pd.DataFrame({
                    'predicted': predictions[f'{component}_confidence'],
                    'actual': gold_data[f'{component}_gold'],
                    'error': predictions[f'{component}_confidence'] - gold_data[f'{component}_gold']
                })
        
        return errors
    
    def _update_calibration_models(self, errors: Dict[str, pd.DataFrame]) -> None:
        """Update calibration models based on errors.
        
        Args:
            errors: Dictionary mapping components to their error DataFrames
        """
        for component, error_df in errors.items():
            if component in self.calibration_models:
                # Fit calibration model
                X = error_df['predicted'].values.reshape(-1, 1)
                y = error_df['actual'].values
                
                self.calibration_models[component].fit(X, y)
    
    def _apply_calibration_to_attributor(self, attributor: 'ResponseAttributor') -> None:
        """Apply calibration models to ResponseAttributor.
        
        Args:
            attributor: ResponseAttributor instance
        """
        for component, model in self.calibration_models.items():
            if hasattr(attributor, f'_{component}_confidence'):
                # Update confidence calculation method
                setattr(attributor, f'_{component}_confidence',
                       lambda x, m=model: m.predict(x.reshape(-1, 1)).ravel())
    
    def _get_current_params(self, attributor: 'ResponseAttributor') -> Dict[str, Any]:
        """Get current parameters from ResponseAttributor.
        
        Args:
            attributor: ResponseAttributor instance
            
        Returns:
            Dictionary with current parameters
        """
        return {
            'min_signal_count': attributor.min_signal_count,
            'min_signal_strength': attributor.min_signal_strength,
            'confidence_weights': attributor.confidence_weights
        }
    
    def _is_improvement(self, new_metrics: Dict[str, Dict[str, float]],
                       current_metrics: Dict[str, Dict[str, float]]) -> bool:
        """Check if new metrics show improvement.
        
        Args:
            new_metrics: New error metrics
            current_metrics: Current error metrics
            
        Returns:
            True if metrics show improvement
        """
        # Calculate weighted average of improvements
        improvements = []
        weights = []
        
        for component in new_metrics:
            if component in current_metrics:
                # Brier score (lower is better)
                brier_improvement = (
                    current_metrics[component]['brier_score'] -
                    new_metrics[component]['brier_score']
                )
                improvements.append(brier_improvement)
                weights.append(0.4)
                
                # Calibration error (lower is better)
                cal_improvement = (
                    current_metrics[component]['calibration_error'] -
                    new_metrics[component]['calibration_error']
                )
                improvements.append(cal_improvement)
                weights.append(0.3)
                
                # Confidence correlation (higher is better)
                corr_improvement = (
                    new_metrics[component]['confidence_correlation'] -
                    current_metrics[component]['confidence_correlation']
                )
                improvements.append(corr_improvement)
                weights.append(0.3)
        
        # Calculate weighted average improvement
        weighted_improvement = np.average(improvements, weights=weights)
        
        return weighted_improvement > 0.001  # Small threshold to avoid noise
    
    def _calculate_brier_score(self, predictions: np.ndarray,
                             actuals: np.ndarray) -> float:
        """Calculate Brier score.
        
        Args:
            predictions: Predicted probabilities
            actuals: Actual values
            
        Returns:
            Brier score
        """
        return np.mean((predictions - actuals) ** 2)
    
    def _calculate_calibration_error(self, predictions: np.ndarray,
                                   actuals: np.ndarray,
                                   n_bins: int = 10) -> float:
        """Calculate calibration error.
        
        Args:
            predictions: Predicted probabilities
            actuals: Actual values
            n_bins: Number of bins for calibration curve
            
        Returns:
            Calibration error
        """
        bin_edges = np.linspace(0, 1, n_bins + 1)
        bin_indices = np.digitize(predictions, bin_edges) - 1
        
        calibration_error = 0
        for i in range(n_bins):
            mask = bin_indices == i
            if np.any(mask):
                pred_mean = np.mean(predictions[mask])
                actual_mean = np.mean(actuals[mask])
                calibration_error += abs(pred_mean - actual_mean)
        
        return calibration_error / n_bins
    
    def _calculate_confidence_correlation(self, predictions: np.ndarray,
                                        actuals: np.ndarray) -> float:
        """Calculate correlation between confidence and accuracy.
        
        Args:
            predictions: Predicted probabilities
            actuals: Actual values
            
        Returns:
            Correlation coefficient
        """
        return np.corrcoef(predictions, actuals)[0, 1] 
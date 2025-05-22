"""
Confidence-Aware Model Training Module.

This module implements confidence-aware training for medication response prediction
using XGBoost. It incorporates confidence scores from the response attribution
system to weight training samples and evaluate model performance across different
confidence tiers.

Key features:
1. Sample weighting based on confidence scores
2. Confidence threshold filtering
3. Cross-validation with confidence weighting
4. Performance evaluation across confidence tiers
5. Feature importance analysis
6. Visualization of confidence-based performance
7. Confidence score calibration
8. Tier-specific feature importance
9. Confidence-error correlation analysis
10. Weighted vs. unweighted performance comparison

Example:
    >>> trainer = ConfidenceAwareTrainer()
    >>> data_dict = trainer.prepare_data(df, feature_cols, confidence_col='overall_confidence')
    >>> results = trainer.train_all_models(data_dict)
    >>> tier_metrics = trainer.evaluate_confidence_tiers(data_dict, results)
    >>> trainer.plot_confidence_performance(tier_metrics, 'confidence_performance.png')
"""

import pandas as pd
import numpy as np
import xgboost as xgb
from sklearn.model_selection import train_test_split, KFold
from sklearn.metrics import r2_score, mean_squared_error
from sklearn.preprocessing import MinMaxScaler
from scipy import stats
from typing import Dict, List, Tuple, Any, Optional
import matplotlib.pyplot as plt
import seaborn as sns
import logging

logger = logging.getLogger(__name__)

class ConfidenceAwareTrainer:
    """Confidence-aware model trainer for medication response prediction."""
    
    def __init__(self, config: Dict[str, Any] = None):
        """Initialize the trainer with configuration."""
        self.config = config or {
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
            'calibration_method': 'minmax',  # or 'quantile'
            'plot_normalize_counts': False
        }
        
        # Target columns for the three response dimensions
        self.target_cols = [
            'activation_sedation_score',
            'emotional_blunting_restoration_score',
            'appetite_metabolic_score'
        ]
        
        # Store metrics during training
        self.metrics_history = {}
        
        # Initialize confidence calibrator
        self.confidence_calibrator = MinMaxScaler()
    
    def calibrate_confidence_scores(self, 
                                  confidence_scores: np.ndarray,
                                  method: str = None) -> np.ndarray:
        """
        Calibrate confidence scores using specified method.
        
        Args:
            confidence_scores: Raw confidence scores
            method: Calibration method ('minmax' or 'quantile')
            
        Returns:
            Calibrated confidence scores
            
        Note:
            - MinMax scaling: Scales to [0, 1] range
            - Quantile scaling: Uses quantile-based scaling to handle outliers
        """
        method = method or self.config['calibration_method']
        
        if method == 'minmax':
            # Reshape for sklearn
            scores_2d = confidence_scores.reshape(-1, 1)
            calibrated = self.confidence_calibrator.fit_transform(scores_2d).ravel()
            
        elif method == 'quantile':
            # Use quantile-based scaling
            q1, q99 = np.percentile(confidence_scores, [1, 99])
            calibrated = np.clip(confidence_scores, q1, q99)
            calibrated = (calibrated - q1) / (q99 - q1)
            
        else:
            raise ValueError(f"Unknown calibration method: {method}")
        
        logger.info(f"Confidence calibration ({method}):")
        logger.info(f"  Original: mean={confidence_scores.mean():.3f}, min={confidence_scores.min():.3f}, max={confidence_scores.max():.3f}")
        logger.info(f"  Calibrated: mean={calibrated.mean():.3f}, min={calibrated.min():.3f}, max={calibrated.max():.3f}")
        
        return calibrated
    
    def analyze_confidence_error_correlation(self,
                                          data_dict: Dict[str, Any],
                                          results: Dict[str, Any]) -> Dict[str, Any]:
        """
        Analyze correlation between confidence scores and prediction errors.
        
        Args:
            data_dict: Dictionary with prepared data
            results: Dictionary with trained models and metrics
            
        Returns:
            Dictionary with correlation analysis results
            
        Note:
            - Calculates correlation between confidence and absolute error
            - Performs correlation test for statistical significance
            - Analyzes correlation by confidence tier
        """
        correlation_results = {}
        
        for target_name in self.target_cols:
            model = results[target_name]['model']
            
            X_test = data_dict['test']['X']
            y_test = data_dict['test']['y'][target_name]
            confidence = data_dict['test']['confidence']
            
            # Get predictions
            dtest = xgb.DMatrix(X_test)
            preds = model.predict(dtest)
            
            # Calculate absolute errors
            abs_errors = np.abs(y_test - preds)
            
            # Calculate overall correlation
            correlation, p_value = stats.pearsonr(confidence, abs_errors)
            
            correlation_results[target_name] = {
                'overall': {
                    'correlation': correlation,
                    'p_value': p_value
                },
                'by_tier': {}
            }
            
            # Calculate correlation by confidence tier
            for tier_name, (lower, upper) in self.config['confidence_tiers'].items():
                tier_mask = (confidence >= lower) & (confidence < upper)
                
                if tier_mask.sum() > 0:
                    tier_corr, tier_p = stats.pearsonr(
                        confidence[tier_mask],
                        abs_errors[tier_mask]
                    )
                    
                    correlation_results[target_name]['by_tier'][tier_name] = {
                        'correlation': tier_corr,
                        'p_value': tier_p,
                        'count': tier_mask.sum()
                    }
            
            logger.info(f"{target_name} confidence-error correlation:")
            logger.info(f"  Overall: {correlation:.3f} (p={p_value:.3e})")
            for tier_name, tier_results in correlation_results[target_name]['by_tier'].items():
                logger.info(f"  {tier_name}: {tier_results['correlation']:.3f} (p={tier_results['p_value']:.3e}, n={tier_results['count']})")
        
        return correlation_results
    
    def get_tier_specific_feature_importance(self,
                                          data_dict: Dict[str, Any],
                                          results: Dict[str, Any]) -> Dict[str, Any]:
        """
        Calculate feature importance specific to each confidence tier.
        
        Args:
            data_dict: Dictionary with prepared data
            results: Dictionary with trained models and metrics
            
        Returns:
            Dictionary with tier-specific feature importance
            
        Note:
            - Trains separate models for each confidence tier
            - Compares feature importance across tiers
            - Identifies tier-specific important features
        """
        tier_importance = {}
        
        for target_name in self.target_cols:
            tier_importance[target_name] = {}
            
            X_test = data_dict['test']['X']
            y_test = data_dict['test']['y'][target_name]
            confidence = data_dict['test']['confidence']
            
            # Calculate importance for each tier
            for tier_name, (lower, upper) in self.config['confidence_tiers'].items():
                tier_mask = (confidence >= lower) & (confidence < upper)
                
                if tier_mask.sum() > 0:
                    # Train model on tier-specific data
                    X_tier = X_test[tier_mask]
                    y_tier = y_test[tier_mask]
                    
                    dtier = xgb.DMatrix(X_tier, label=y_tier)
                    
                    # Use same parameters as main model
                    params = {
                        'objective': 'reg:squarederror',
                        'eval_metric': 'rmse',
                        'eta': self.config['learning_rate'],
                        'max_depth': self.config['max_depth'],
                        'subsample': self.config['subsample'],
                        'colsample_bytree': self.config['colsample_bytree'],
                        'seed': self.config['random_state']
                    }
                    
                    tier_model = xgb.train(
                        params,
                        dtier,
                        num_boost_round=self.config['n_estimators']
                    )
                    
                    # Get feature importance
                    importance = tier_model.get_score(importance_type='gain')
                    feature_importance = {}
                    for feature in data_dict['feature_names']:
                        if feature in importance:
                            feature_importance[feature] = importance[feature]
                        else:
                            feature_importance[feature] = 0
                    
                    # Sort features by importance
                    sorted_importance = sorted(feature_importance.items(), key=lambda x: x[1], reverse=True)
                    top_features = sorted_importance[:10]
                    
                    tier_importance[target_name][tier_name] = {
                        'importance': feature_importance,
                        'top_features': top_features
                    }
                    
                    logger.info(f"{target_name} - {tier_name} tier top features:")
                    logger.info(f"  {[f[0] for f in top_features]}")
        
        return tier_importance
    
    def prepare_data(self, 
                    df: pd.DataFrame, 
                    feature_cols: List[str],
                    confidence_col: str = 'overall_confidence',
                    random_state: int = None) -> Dict[str, Any]:
        """
        Prepare data for training.
        
        Args:
            df: Input DataFrame
            feature_cols: Feature column names
            confidence_col: Column with confidence scores
            random_state: Random state for reproducibility
            
        Returns:
            Dictionary with prepared data
            
        Note:
            - Applies confidence threshold if specified
            - Creates sample weights from confidence scores
            - Splits data into train/validation/test sets
            - Optionally calibrates confidence scores
        """
        # Use config random state if not specified
        random_state = random_state or self.config['random_state']
        
        # Apply confidence threshold
        if self.config['confidence_threshold'] > 0:
            df = df[df[confidence_col] >= self.config['confidence_threshold']]
            logger.info(f"Applied confidence threshold {self.config['confidence_threshold']}: {len(df)} samples remaining")
        
        # Split into train, validation, and test sets
        train_df, test_df = train_test_split(
            df, test_size=0.2, random_state=random_state
        )
        
        train_df, val_df = train_test_split(
            train_df, test_size=0.25, random_state=random_state
        )
        
        logger.info(f"Data split: {len(train_df)} train, {len(val_df)} validation, {len(test_df)} test")
        
        # Calibrate confidence scores if enabled
        if self.config['calibrate_confidence']:
            train_df[confidence_col] = self.calibrate_confidence_scores(train_df[confidence_col].values)
            val_df[confidence_col] = self.calibrate_confidence_scores(val_df[confidence_col].values)
            test_df[confidence_col] = self.calibrate_confidence_scores(test_df[confidence_col].values)
        
        # Create sample weights from confidence scores
        if self.config['use_sample_weights']:
            train_weights = train_df[confidence_col].values
            val_weights = val_df[confidence_col].values
            test_weights = test_df[confidence_col].values
            
            # Log weights distribution
            logger.info(f"Train weights: mean={train_weights.mean():.3f}, min={train_weights.min():.3f}, max={train_weights.max():.3f}")
        else:
            train_weights = np.ones(len(train_df))
            val_weights = np.ones(len(val_df))
            test_weights = np.ones(len(test_df))
        
        # Prepare features and targets
        data_dict = {
            'train': {
                'X': train_df[feature_cols],
                'y': train_df[self.target_cols],
                'weights': train_weights,
                'confidence': train_df[confidence_col]
            },
            'val': {
                'X': val_df[feature_cols],
                'y': val_df[self.target_cols],
                'weights': val_weights,
                'confidence': val_df[confidence_col]
            },
            'test': {
                'X': test_df[feature_cols],
                'y': test_df[self.target_cols],
                'weights': test_weights,
                'confidence': test_df[confidence_col]
            },
            'feature_names': feature_cols,
            'target_names': self.target_cols
        }
        
        return data_dict
    
    def train_model(self, 
                   data_dict: Dict[str, Any],
                   target_idx: int = 0,
                   use_weights: bool = None) -> Dict[str, Any]:
        """
        Train a model for a specific target.
        
        Args:
            data_dict: Dictionary with prepared data
            target_idx: Index of target column to predict
            use_weights: Whether to use sample weights (overrides config)
            
        Returns:
            Dictionary with trained model and metrics
            
        Note:
            - Uses XGBoost with early stopping
            - Can optionally disable sample weights for comparison
            - Tracks feature importance
        """
        # Get target name
        target_name = self.target_cols[target_idx]
        
        # Determine whether to use weights
        use_weights = use_weights if use_weights is not None else self.config['use_sample_weights']
        weight_desc = "weighted" if use_weights else "unweighted"
        logger.info(f"Training {weight_desc} model for {target_name}")
        
        # Extract data
        X_train = data_dict['train']['X']
        y_train = data_dict['train']['y'][target_name]
        train_weights = data_dict['train']['weights'] if use_weights else np.ones(len(X_train))
        
        X_val = data_dict['val']['X']
        y_val = data_dict['val']['y'][target_name]
        val_weights = data_dict['val']['weights'] if use_weights else np.ones(len(X_val))
        
        # Create DMatrix objects
        dtrain = xgb.DMatrix(
            X_train, 
            label=y_train,
            weight=train_weights
        )
        
        dval = xgb.DMatrix(
            X_val,
            label=y_val,
            weight=val_weights
        )
        
        # Set XGBoost parameters
        params = {
            'objective': 'reg:squarederror',
            'eval_metric': 'rmse',
            'eta': self.config['learning_rate'],
            'max_depth': self.config['max_depth'],
            'subsample': self.config['subsample'],
            'colsample_bytree': self.config['colsample_bytree'],
            'seed': self.config['random_state']
        }
        
        # Train model with early stopping
        evals = [(dtrain, 'train'), (dval, 'val')]
        evals_result = {}
        
        model = xgb.train(
            params,
            dtrain,
            num_boost_round=self.config['n_estimators'],
            evals=evals,
            early_stopping_rounds=self.config['early_stopping_rounds'],
            evals_result=evals_result,
            verbose_eval=100
        )
        
        # Calculate metrics on validation set
        val_preds = model.predict(dval)
        val_r2 = r2_score(y_val, val_preds, sample_weight=val_weights)
        val_rmse = np.sqrt(mean_squared_error(y_val, val_preds, sample_weight=val_weights))
        
        # Get feature importance
        importance = model.get_score(importance_type='gain')
        feature_importance = {}
        for feature in data_dict['feature_names']:
            if feature in importance:
                feature_importance[feature] = importance[feature]
            else:
                feature_importance[feature] = 0
        
        # Sort features by importance
        sorted_importance = sorted(feature_importance.items(), key=lambda x: x[1], reverse=True)
        top_features = sorted_importance[:10]
        
        logger.info(f"{target_name} {weight_desc} model validation metrics:")
        logger.info(f"  R²: {val_r2:.4f}")
        logger.info(f"  RMSE: {val_rmse:.4f}")
        logger.info(f"  Top features: {[f[0] for f in top_features]}")
        
        return {
            'model': model,
            'metrics': {
                'r2': val_r2,
                'rmse': val_rmse
            },
            'feature_importance': feature_importance,
            'evals_result': evals_result,
            'top_features': top_features,
            'use_weights': use_weights
        }
    
    def train_all_models(self, data_dict: Dict[str, Any], compare_weights: bool = False) -> Dict[str, Any]:
        """
        Train models for all targets.
        
        Args:
            data_dict: Dictionary with prepared data
            compare_weights: Whether to train both weighted and unweighted models
            
        Returns:
            Dictionary with all trained models and metrics
            
        Note:
            - Trains separate models for each response dimension
            - Optionally trains both weighted and unweighted models
            - Calculates macro-R² across all dimensions
        """
        results = {}
        
        for i, target_name in enumerate(self.target_cols):
            logger.info(f"Training model {i+1}/{len(self.target_cols)}: {target_name}")
            
            if compare_weights:
                # Train both weighted and unweighted models
                weighted_results = self.train_model(data_dict, i, use_weights=True)
                unweighted_results = self.train_model(data_dict, i, use_weights=False)
                
                results[target_name] = {
                    'weighted': weighted_results,
                    'unweighted': unweighted_results,
                    'weight_improvement': {
                        'r2': weighted_results['metrics']['r2'] - unweighted_results['metrics']['r2'],
                        'rmse': unweighted_results['metrics']['rmse'] - weighted_results['metrics']['rmse']
                    }
                }
                
                logger.info(f"{target_name} weight improvement:")
                logger.info(f"  R²: +{results[target_name]['weight_improvement']['r2']:.4f}")
                logger.info(f"  RMSE: -{results[target_name]['weight_improvement']['rmse']:.4f}")
            else:
                # Train single model using config weights setting
                results[target_name] = self.train_model(data_dict, i)
        
        # Calculate macro-R²
        if compare_weights:
            weighted_macro_r2 = np.mean([results[target]['weighted']['metrics']['r2'] for target in self.target_cols])
            unweighted_macro_r2 = np.mean([results[target]['unweighted']['metrics']['r2'] for target in self.target_cols])
            logger.info(f"Macro-R² across all dimensions:")
            logger.info(f"  Weighted: {weighted_macro_r2:.4f}")
            logger.info(f"  Unweighted: {unweighted_macro_r2:.4f}")
            logger.info(f"  Improvement: +{weighted_macro_r2 - unweighted_macro_r2:.4f}")
            
            results['macro_r2'] = {
                'weighted': weighted_macro_r2,
                'unweighted': unweighted_macro_r2,
                'improvement': weighted_macro_r2 - unweighted_macro_r2
            }
        else:
            macro_r2 = np.mean([results[target]['metrics']['r2'] for target in self.target_cols])
            logger.info(f"Macro-R² across all dimensions: {macro_r2:.4f}")
            results['macro_r2'] = macro_r2
        
        return results
    
    def evaluate_confidence_tiers(self, 
                               data_dict: Dict[str, Any],
                               results: Dict[str, Any],
                               compare_weights: bool = False) -> Dict[str, Any]:
        """
        Evaluate model performance across confidence tiers.
        
        Args:
            data_dict: Dictionary with prepared data
            results: Dictionary with trained models and metrics
            compare_weights: Whether to evaluate both weighted and unweighted models
            
        Returns:
            Dictionary with confidence tier metrics
            
        Note:
            - Evaluates performance in high/medium/low confidence tiers
            - Optionally compares weighted vs. unweighted performance
            - Calculates R² and RMSE for each tier
            - Tracks sample counts and percentages
        """
        tier_metrics = {}
        
        # Evaluate on test set for each target
        for target_name in self.target_cols:
            tier_metrics[target_name] = {}
            
            if compare_weights:
                # Evaluate both weighted and unweighted models
                for weight_type in ['weighted', 'unweighted']:
                    model = results[target_name][weight_type]['model']
                    tier_metrics[target_name][weight_type] = self._evaluate_tiers_for_model(
                        data_dict, model, target_name
                    )
                
                # Calculate improvements
                tier_metrics[target_name]['improvement'] = {}
                for tier_name in self.config['confidence_tiers'].keys():
                    if tier_name in tier_metrics[target_name]['weighted']:
                        tier_metrics[target_name]['improvement'][tier_name] = {
                            'r2': tier_metrics[target_name]['weighted'][tier_name]['r2'] - 
                                 tier_metrics[target_name]['unweighted'][tier_name]['r2'],
                            'rmse': tier_metrics[target_name]['unweighted'][tier_name]['rmse'] - 
                                   tier_metrics[target_name]['weighted'][tier_name]['rmse']
                        }
            else:
                # Evaluate single model
                model = results[target_name]['model']
                tier_metrics[target_name] = self._evaluate_tiers_for_model(
                    data_dict, model, target_name
                )
        
        return tier_metrics
    
    def _evaluate_tiers_for_model(self,
                                data_dict: Dict[str, Any],
                                model: xgb.Booster,
                                target_name: str) -> Dict[str, Any]:
        """
        Helper method to evaluate model performance across confidence tiers.
        
        Args:
            data_dict: Dictionary with prepared data
            model: Trained XGBoost model
            target_name: Name of target column
            
        Returns:
            Dictionary with tier-specific metrics
        """
        X_test = data_dict['test']['X']
        y_test = data_dict['test']['y'][target_name]
        confidence = data_dict['test']['confidence']
        
        dtest = xgb.DMatrix(X_test)
        preds = model.predict(dtest)
        
        tier_results = {}
        
        for tier_name, (lower, upper) in self.config['confidence_tiers'].items():
            # Get samples in this confidence tier
            tier_mask = (confidence >= lower) & (confidence < upper)
            
            if tier_mask.sum() > 0:
                tier_preds = preds[tier_mask]
                tier_true = y_test.iloc[tier_mask]
                
                # Calculate metrics
                tier_r2 = r2_score(tier_true, tier_preds)
                tier_rmse = np.sqrt(mean_squared_error(tier_true, tier_preds))
                
                tier_results[tier_name] = {
                    'r2': tier_r2,
                    'rmse': tier_rmse,
                    'count': tier_mask.sum(),
                    'percentage': tier_mask.sum() / len(confidence) * 100
                }
                
                logger.info(f"{target_name} - {tier_name} confidence tier ({tier_results[tier_name]['count']} samples, {tier_results[tier_name]['percentage']:.1f}%):")
                logger.info(f"  R²: {tier_r2:.4f}")
                logger.info(f"  RMSE: {tier_rmse:.4f}")
        
        return tier_results
    
    def run_cross_validation(self, 
                           df: pd.DataFrame,
                           feature_cols: List[str],
                           confidence_col: str = 'overall_confidence',
                           n_folds: int = 5) -> Dict[str, Any]:
        """
        Run cross-validation with confidence weighting.
        
        Args:
            df: Input DataFrame
            feature_cols: Feature column names
            confidence_col: Column with confidence scores
            n_folds: Number of CV folds
            
        Returns:
            Dictionary with CV metrics
            
        Note:
            - Uses K-fold cross-validation
            - Applies confidence threshold if specified
            - Incorporates sample weights from confidence scores
        """
        # Apply confidence threshold
        if self.config['confidence_threshold'] > 0:
            df = df[df[confidence_col] >= self.config['confidence_threshold']]
        
        # Initialize KFold
        kf = KFold(n_splits=n_folds, shuffle=True, random_state=self.config['random_state'])
        
        # Initialize metrics storage
        cv_results = {target: {'r2': [], 'rmse': []} for target in self.target_cols}
        
        # Run CV
        for fold, (train_idx, test_idx) in enumerate(kf.split(df)):
            logger.info(f"Running CV fold {fold+1}/{n_folds}")
            
            # Split data
            train_df = df.iloc[train_idx]
            test_df = df.iloc[test_idx]
            
            # Create sample weights
            if self.config['use_sample_weights']:
                train_weights = train_df[confidence_col].values
            else:
                train_weights = np.ones(len(train_df))
            
            # Train models for each target
            for target_idx, target_name in enumerate(self.target_cols):
                # Extract data
                X_train = train_df[feature_cols]
                y_train = train_df[target_name]
                
                X_test = test_df[feature_cols]
                y_test = test_df[target_name]
                
                # Create DMatrix objects
                dtrain = xgb.DMatrix(X_train, label=y_train, weight=train_weights)
                dtest = xgb.DMatrix(X_test)
                
                # Set XGBoost parameters
                params = {
                    'objective': 'reg:squarederror',
                    'eval_metric': 'rmse',
                    'eta': self.config['learning_rate'],
                    'max_depth': self.config['max_depth'],
                    'subsample': self.config['subsample'],
                    'colsample_bytree': self.config['colsample_bytree'],
                    'seed': self.config['random_state']
                }
                
                # Train model
                model = xgb.train(
                    params,
                    dtrain,
                    num_boost_round=self.config['n_estimators']
                )
                
                # Predict on test set
                preds = model.predict(dtest)
                
                # Calculate metrics
                r2 = r2_score(y_test, preds)
                rmse = np.sqrt(mean_squared_error(y_test, preds))
                
                # Store metrics
                cv_results[target_name]['r2'].append(r2)
                cv_results[target_name]['rmse'].append(rmse)
        
        # Calculate average metrics
        for target_name in self.target_cols:
            cv_results[target_name]['mean_r2'] = np.mean(cv_results[target_name]['r2'])
            cv_results[target_name]['std_r2'] = np.std(cv_results[target_name]['r2'])
            cv_results[target_name]['mean_rmse'] = np.mean(cv_results[target_name]['rmse'])
            cv_results[target_name]['std_rmse'] = np.std(cv_results[target_name]['rmse'])
            
            logger.info(f"{target_name} CV metrics:")
            logger.info(f"  R²: {cv_results[target_name]['mean_r2']:.4f} ± {cv_results[target_name]['std_r2']:.4f}")
            logger.info(f"  RMSE: {cv_results[target_name]['mean_rmse']:.4f} ± {cv_results[target_name]['std_rmse']:.4f}")
        
        # Calculate macro-R²
        macro_r2 = np.mean([cv_results[target_name]['mean_r2'] for target_name in self.target_cols])
        logger.info(f"Macro-R² across all dimensions: {macro_r2:.4f}")
        
        cv_results['macro_r2'] = macro_r2
        return cv_results
    
    def plot_confidence_performance(self, 
                                  tier_metrics: Dict[str, Any], 
                                  output_path: str = None,
                                  normalize_counts: bool = None,
                                  compare_weights: bool = False):
        """
        Plot performance across confidence tiers.
        
        Args:
            tier_metrics: Dictionary with confidence tier metrics
            output_path: Path to save plot
            normalize_counts: Whether to normalize sample counts to percentages
            compare_weights: Whether to plot weighted vs. unweighted comparison
            
        Note:
            - Creates bar plots for each response dimension
            - Shows R² scores and sample counts
            - Uses consistent color scheme
            - Optionally normalizes sample counts
            - Optionally compares weighted vs. unweighted performance
        """
        normalize_counts = normalize_counts if normalize_counts is not None else self.config['plot_normalize_counts']
        
        # Create figure
        n_cols = len(self.target_cols)
        n_rows = 2 if compare_weights else 1
        fig, axes = plt.subplots(n_rows, n_cols, figsize=(15, 5 * n_rows))
        
        # Set color palette
        colors = sns.color_palette("viridis", 3)
        
        # Plot for each target
        for i, target_name in enumerate(self.target_cols):
            if compare_weights:
                # Plot weighted vs. unweighted comparison
                for j, weight_type in enumerate(['weighted', 'unweighted']):
                    ax = axes[j, i] if n_cols > 1 else axes[j]
                    
                    # Extract data
                    tiers = []
                    r2_scores = []
                    counts = []
                    
                    for tier_name in ['high', 'medium', 'low']:
                        if tier_name in tier_metrics[target_name][weight_type]:
                            tiers.append(tier_name)
                            r2_scores.append(tier_metrics[target_name][weight_type][tier_name]['r2'])
                            counts.append(tier_metrics[target_name][weight_type][tier_name]['count'])
                    
                    # Normalize counts if requested
                    if normalize_counts:
                        total = sum(counts)
                        counts = [count/total*100 for count in counts]
                        count_label = "Percentage"
                    else:
                        count_label = "Count"
                    
                    # Plot bars
                    bars = ax.bar(tiers, r2_scores, color=colors)
                    
                    # Add value labels
                    for k, bar in enumerate(bars):
                        height = bar.get_height()
                        ax.text(bar.get_x() + bar.get_width()/2., height + 0.02,
                                f'{r2_scores[k]:.3f}\n({counts[k]:.1f}{"%" if normalize_counts else ""})',
                                ha='center', va='bottom', rotation=0)
                    
                    # Set labels and title
                    ax.set_ylim(0, 1.0)
                    ax.set_title(f"{target_name} by Confidence ({weight_type})")
                    ax.set_ylabel("R² Score")
                    ax.set_xlabel("Confidence Tier")
                    
                    # Add count axis
                    ax2 = ax.twinx()
                    ax2.set_ylabel(count_label)
                    ax2.set_ylim(0, max(counts) * 1.1)
            else:
                # Plot single model results
                ax = axes[i] if n_cols > 1 else axes
                
                # Extract data
                tiers = []
                r2_scores = []
                counts = []
                
                for tier_name in ['high', 'medium', 'low']:
                    if tier_name in tier_metrics[target_name]:
                        tiers.append(tier_name)
                        r2_scores.append(tier_metrics[target_name][tier_name]['r2'])
                        counts.append(tier_metrics[target_name][tier_name]['count'])
                
                # Normalize counts if requested
                if normalize_counts:
                    total = sum(counts)
                    counts = [count/total*100 for count in counts]
                    count_label = "Percentage"
                else:
                    count_label = "Count"
                
                # Plot bars
                bars = ax.bar(tiers, r2_scores, color=colors)
                
                # Add value labels
                for j, bar in enumerate(bars):
                    height = bar.get_height()
                    ax.text(bar.get_x() + bar.get_width()/2., height + 0.02,
                            f'{r2_scores[j]:.3f}\n({counts[j]:.1f}{"%" if normalize_counts else ""})',
                            ha='center', va='bottom', rotation=0)
                
                # Set labels and title
                ax.set_ylim(0, 1.0)
                ax.set_title(f"{target_name} by Confidence")
                ax.set_ylabel("R² Score")
                ax.set_xlabel("Confidence Tier")
                
                # Add count axis
                ax2 = ax.twinx()
                ax2.set_ylabel(count_label)
                ax2.set_ylim(0, max(counts) * 1.1)
        
        plt.tight_layout()
        
        if output_path:
            plt.savefig(output_path)
        else:
            plt.show() 
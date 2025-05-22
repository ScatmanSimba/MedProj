# -*- coding: utf-8 -*-
"""train_model.py - Multi-task GBT trainer for medication response prediction

This module trains an XGBoost model to predict three response dimensions:
1. Activation ↔ Sedation
2. Emotional Blunting ↔ Restoration
3. Appetite / Metabolic impact

It implements multi-task regression and includes early stopping with validation data.

Usage:
    python -m src.models.train_model \
        --data data/processed/features_dataset.parquet \
        --output models/receptor_predictor.pkl \
        --config configs/model_config.yaml
"""

import os
import logging
import pandas as pd
import numpy as np
import argparse
import yaml
import pickle
from pathlib import Path
from typing import Dict, List, Set, Optional, Any, Tuple, Union
import json
from tqdm import tqdm
from sklearn.model_selection import train_test_split, KFold
import xgboost as xgb
from sklearn.metrics import r2_score, mean_squared_error
import matplotlib.pyplot as plt
import shap

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler("train_model.log"),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger("train_model")

class ModelTrainer:
    def __init__(self, config_path: str = "configs/model_config.yaml"):
        """
        Initialize the model trainer.
        
        Args:
            config_path: Path to model configuration file
        """
        self.config_path = Path(config_path)
        self.load_config()
        
        # Response dimensions
        self.target_cols = [
            'activation_sedation',
            'emotional_blunting_restoration',
            'appetite_metabolic'
        ]
        
    def load_config(self):
        """Load model configuration from YAML file."""
        if not self.config_path.exists():
            raise FileNotFoundError(f"Config file not found at {self.config_path}")
            
        with open(self.config_path) as f:
            self.config = yaml.safe_load(f)
            
    def prepare_data(self, 
                    df: pd.DataFrame,
                    feature_cols: List[str]) -> Tuple[np.ndarray, np.ndarray]:
        """
        Prepare data for model training.
        
        Args:
            df: Input DataFrame
            feature_cols: List of feature column names
            
        Returns:
            Tuple of (X, y) arrays
        """
        X = df[feature_cols].values
        y = df[self.target_cols].values
        
        return X, y
        
    def train_model(self,
                   X: np.ndarray,
                   y: np.ndarray,
                   feature_names: List[str]) -> Tuple[xgb.XGBRegressor, Dict]:
        """
        Train multi-task XGBoost model.
        
        Args:
            X: Feature matrix
            y: Target matrix
            feature_names: List of feature names
            
        Returns:
            Tuple of (trained model, training metrics)
        """
        # Initialize model
        model = xgb.XGBRegressor(
            objective='reg:squarederror',
            n_estimators=self.config['n_estimators'],
            max_depth=self.config['max_depth'],
            learning_rate=self.config['learning_rate'],
            subsample=self.config['subsample'],
            colsample_bytree=self.config['colsample_bytree'],
            random_state=self.config['random_state']
        )
        
        # Train model
        model.fit(
            X, y,
            eval_set=[(X, y)],
            early_stopping_rounds=self.config['early_stopping_rounds'],
            verbose=False
        )
        
        # Calculate training metrics
        y_pred = model.predict(X)
        metrics = {
            'r2_scores': {
                col: r2_score(y[:, i], y_pred[:, i])
                for i, col in enumerate(self.target_cols)
            }
        }
        
        return model, metrics
        
    def cross_validate(self,
                      X: np.ndarray,
                      y: np.ndarray,
                      feature_names: List[str],
                      n_splits: int = 5) -> Dict:
        """
        Perform k-fold cross-validation.
        
        Args:
            X: Feature matrix
            y: Target matrix
            feature_names: List of feature names
            n_splits: Number of CV folds
            
        Returns:
            Dictionary of CV metrics
        """
        kf = KFold(n_splits=n_splits, shuffle=True, random_state=42)
        cv_metrics = {
            'fold_scores': [],
            'feature_importance': []
        }
        
        for fold, (train_idx, val_idx) in enumerate(kf.split(X)):
            # Split data
            X_train, X_val = X[train_idx], X[val_idx]
            y_train, y_val = y[train_idx], y[val_idx]
            
            # Train model
            model, metrics = self.train_model(X_train, y_train, feature_names)
            
            # Evaluate on validation set
            y_pred = model.predict(X_val)
            val_metrics = {
                col: r2_score(y_val[:, i], y_pred[:, i])
                for i, col in enumerate(self.target_cols)
            }
            
            # Calculate feature importance
            importance = model.feature_importances_
            feature_importance = dict(zip(feature_names, importance))
            
            cv_metrics['fold_scores'].append(val_metrics)
            cv_metrics['feature_importance'].append(feature_importance)
            
        # Calculate average metrics
        cv_metrics['mean_scores'] = {
            col: np.mean([fold[col] for fold in cv_metrics['fold_scores']])
            for col in self.target_cols
        }
        
        cv_metrics['std_scores'] = {
            col: np.std([fold[col] for fold in cv_metrics['fold_scores']])
            for col in self.target_cols
        }
        
        return cv_metrics
        
    def analyze_feature_importance(self,
                                 model: xgb.XGBRegressor,
                                 X: np.ndarray,
                                 feature_names: List[str]) -> Dict:
        """
        Analyze feature importance using SHAP values.
        
        Args:
            model: Trained XGBoost model
            X: Feature matrix
            feature_names: List of feature names
            
        Returns:
            Dictionary of SHAP analysis results
        """
        # Calculate SHAP values
        explainer = shap.TreeExplainer(model)
        shap_values = explainer.shap_values(X)
        
        # Calculate mean absolute SHAP values for each feature
        mean_shap = np.abs(shap_values).mean(axis=0)
        feature_importance = dict(zip(feature_names, mean_shap))
        
        return {
            'shap_values': shap_values,
            'feature_importance': feature_importance
        }
        
    def save_model(self,
                  model: xgb.XGBRegressor,
                  metrics: Dict,
                  output_dir: str = "models"):
        """
        Save trained model and metrics.
        
        Args:
            model: Trained XGBoost model
            metrics: Training metrics
            output_dir: Output directory
        """
        output_dir = Path(output_dir)
        output_dir.mkdir(exist_ok=True)
        
        # Save model
        model.save_model(output_dir / "model.json")
        
        # Save metrics
        with open(output_dir / "metrics.yaml", 'w') as f:
            yaml.dump(metrics, f)
            
        logger.info(f"Saved model and metrics to {output_dir}")
        
    def load_model(self, model_path: str) -> xgb.XGBRegressor:
        """
        Load trained model.
        
        Args:
            model_path: Path to saved model
            
        Returns:
            Loaded XGBoost model
        """
        model = xgb.XGBRegressor()
        model.load_model(model_path)
        return model

def load_config(config_path: str) -> Dict[str, Any]:
    """
    Load configuration from YAML file.

    Args:
        config_path: Path to configuration file

    Returns:
        Configuration dictionary
    """
    logger.info(f"Loading configuration from {config_path}")
    
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    
    return config

def prepare_data(features_df: pd.DataFrame, 
                config: Dict[str, Any],
                random_state: int = 42) -> Tuple[Dict[str, Any], Dict[str, Any]]:
    """
    Prepare data for training.

    Args:
        features_df: DataFrame with features
        config: Configuration dictionary
        random_state: Random state for reproducibility

    Returns:
        Tuple of (data_dict, feature_info)
    """
    logger.info("Preparing data for training...")
    
    # Extract target variables
    target_columns = [
        'activation_sedation_score',
        'emotional_blunting_restoration_score',
        'appetite_metabolic_score'
    ]
    
    # Check if target columns exist
    missing_targets = [col for col in target_columns if col not in features_df.columns]
    if missing_targets:
        raise ValueError(f"Missing target columns: {missing_targets}")
    
    # Rename targets for clarity in metrics
    targets = {
        'activation': features_df['activation_sedation_score'],
        'emotional': features_df['emotional_blunting_restoration_score'],
        'metabolic': features_df['appetite_metabolic_score']
    }
    
    # Define feature selection based on config
    receptor_features = [
        'd2_affinity_log', '5ht2a_affinity_log', 'h1_affinity_log',
        'alpha1_affinity_log', 'm1_affinity_log',
        'd2_affinity_missing', '5ht2a_affinity_missing', 'h1_affinity_missing',
        'alpha1_affinity_missing', 'm1_affinity_missing',
        'd2_affinity_confidence', '5ht2a_affinity_confidence', 'h1_affinity_confidence',
        'alpha1_affinity_confidence', 'm1_affinity_confidence'
    ]
    
    # Start with receptor features as base
    feature_cols = receptor_features.copy()
    
    # Add interaction features and their confidence scores
    interaction_features = [
        'd2_5ht2a_ratio', 'h1_alpha1_sum',
        'd2_5ht2a_ratio_confidence', 'h1_alpha1_sum_confidence'
    ]
    feature_cols.extend(interaction_features)
    
    # Add dosage features if specified in config
    if config.get('use_dosage_features', True):
        if 'normalized_dosage' in features_df.columns:
            feature_cols.append('normalized_dosage')
        if 'has_dosage' in features_df.columns:
            feature_cols.append('has_dosage')
    
    # Add drug class features if specified in config
    if config.get('use_drug_class_features', True):
        class_columns = [col for col in features_df.columns if col.startswith('class_')]
        feature_cols.extend(class_columns)
    
    # Add embedding if specified in config and available
    embeddings = None
    if config.get('use_embeddings', False) and 'embedding' in features_df.columns:
        # Get embeddings as numpy array
        embeddings = np.array(features_df['embedding'].tolist())
        logger.info(f"Loaded embeddings with shape {embeddings.shape}")
    
    # Record feature information
    feature_info = {
        'receptor_features': receptor_features,
        'interaction_features': interaction_features,
        'feature_columns': feature_cols,
        'has_embeddings': embeddings is not None,
        'embedding_dim': embeddings.shape[1] if embeddings is not None else 0
    }
    
    # Split into train/val/test sets
    train_val_df, test_df = train_test_split(
        features_df, 
        test_size=config.get('test_size', 0.2),
        random_state=random_state
    )
    
    train_df, val_df = train_test_split(
        train_val_df,
        test_size=config.get('validation_size', 0.25),  # 25% of 80% = 20% validation
        random_state=random_state
    )
    
    logger.info(f"Data split: {len(train_df)} train, {len(val_df)} validation, {len(test_df)} test")
    
    # Prepare data dictionary for each set
    data_dict = {
        'train': {
            'X': train_df[feature_cols].copy(),
            'y': {target: train_df[col] for target, col in zip(
                ['activation', 'emotional', 'metabolic'], target_columns)}
        },
        'val': {
            'X': val_df[feature_cols].copy(),
            'y': {target: val_df[col] for target, col in zip(
                ['activation', 'emotional', 'metabolic'], target_columns)}
        },
        'test': {
            'X': test_df[feature_cols].copy(),
            'y': {target: test_df[col] for target, col in zip(
                ['activation', 'emotional', 'metabolic'], target_columns)}
        }
    }
    
    # Add embeddings if available
    if embeddings is not None:
        train_indices = train_df.index
        val_indices = val_df.index
        test_indices = test_df.index
        
        data_dict['train']['embeddings'] = embeddings[train_indices]
        data_dict['val']['embeddings'] = embeddings[val_indices]
        data_dict['test']['embeddings'] = embeddings[test_indices]
    
    return data_dict, feature_info

def train_multi_task_model(data_dict: Dict[str, Any], 
                           config: Dict[str, Any],
                           feature_info: Dict[str, Any],
                           random_state: int = 42) -> Dict[str, Any]:
    """
    Train multi-task XGBoost model for the three response dimensions.

    Args:
        data_dict: Dictionary with prepared data
        config: Configuration dictionary
        feature_info: Feature information dictionary
        random_state: Random state for reproducibility

    Returns:
        Dictionary with trained models and performance metrics
    """
    logger.info("Training multi-task XGBoost model...")
    
    # Extract XGBoost parameters from config
    xgb_params = config.get('xgb_params', {})
    if not xgb_params:
        # Default parameters if not specified
        xgb_params = {
            'objective': 'reg:squarederror',
            'eval_metric': 'rmse',
            'eta': 0.1,
            'max_depth': 6,
            'subsample': 0.8,
            'colsample_bytree': 0.8,
            'min_child_weight': 1,
            'gamma': 0,
            'seed': random_state
        }
    
    # Set fixed random seed for reproducibility
    xgb_params['seed'] = random_state
    
    # Extract training parameters
    num_boost_round = config.get('num_boost_round', 1000)
    early_stopping_rounds = config.get('early_stopping_rounds', 50)
    
    # Train a separate model for each target dimension
    models = {}
    metrics = {}
    feature_importance = {}
    shap_values = {}
    
    for target in ['activation', 'emotional', 'metabolic']:
        logger.info(f"Training model for {target} dimension...")
        
        # Prepare datasets
        dtrain = xgb.DMatrix(
            data_dict['train']['X'],
            label=data_dict['train']['y'][target]
        )
        
        dval = xgb.DMatrix(
            data_dict['val']['X'],
            label=data_dict['val']['y'][target]
        )
        
        dtest = xgb.DMatrix(
            data_dict['test']['X'],
            label=data_dict['test']['y'][target]
        )
        
        # Define evaluation sets
        evals = [(dtrain, 'train'), (dval, 'validation')]
        
        # Train model with early stopping
        model = xgb.train(
            xgb_params,
            dtrain,
            num_boost_round=num_boost_round,
            evals=evals,
            early_stopping_rounds=early_stopping_rounds,
            verbose_eval=100
        )
        
        # Save best model
        models[target] = model
        
        # Calculate metrics on test set
        test_preds = model.predict(dtest)
        test_rmse = np.sqrt(mean_squared_error(data_dict['test']['y'][target], test_preds))
        test_r2 = r2_score(data_dict['test']['y'][target], test_preds)
        
        metrics[target] = {
            'best_iteration': model.best_iteration,
            'validation_rmse': model.best_score,
            'test_rmse': test_rmse,
            'test_r2': test_r2
        }
        
        logger.info(f"{target} model - Best iteration: {model.best_iteration}, "
                   f"Validation RMSE: {model.best_score:.4f}, "
                   f"Test RMSE: {test_rmse:.4f}, "
                   f"Test R²: {test_r2:.4f}")
        
        # Extract feature importance
        importance = model.get_score(importance_type='gain')
        feature_importance[target] = {feature: importance.get(feature, 0) for feature in feature_info['feature_columns']}
        
        # Calculate SHAP values for feature attribution
        explainer = shap.TreeExplainer(model)
        shap_values[target] = explainer.shap_values(data_dict['test']['X'])
    
    # Aggregate metrics
    macro_r2 = np.mean([metrics[target]['test_r2'] for target in metrics])
    logger.info(f"Macro-R² across three axes: {macro_r2:.4f}")
    
    # Check if macro-R² meets target in PRD (0.25)
    if macro_r2 >= 0.25:
        logger.info("SUCCESS: Model meets target Macro-R² of 0.25 or higher")
    else:
        logger.warning(f"Model does not meet target Macro-R² (got {macro_r2:.4f}, target is 0.25)")
    
    # Prepare results dictionary
    results = {
        'models': models,
        'metrics': metrics,
        'feature_importance': feature_importance,
        'shap_values': shap_values,
        'macro_r2': macro_r2,
        'feature_info': feature_info
    }
    
    return results

def save_model_results(results: Dict[str, Any], output_path: str) -> None:
    """
    Save model results to file.

    Args:
        results: Results dictionary
        output_path: Path to output file
    """
    logger.info(f"Saving model results to {output_path}")
    
    # Create output directory if it doesn't exist
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    
    # Save model with pickle
    with open(output_path, 'wb') as f:
        pickle.dump(results, f)
    
    # Save metrics separately as JSON
    metrics_path = output_path.replace('.pkl', '_metrics.json')
    metrics_data = {
        'metrics': results['metrics'],
        'macro_r2': results['macro_r2']
    }
    
    with open(metrics_path, 'w') as f:
        json.dump(metrics_data, f, indent=2)
    
    logger.info(f"Saved metrics to {metrics_path}")
    
    # Save feature importance plot
    for target in results['feature_importance']:
        importance = results['feature_importance'][target]
        
        # Sort features by importance
        sorted_features = sorted(importance.items(), key=lambda x: x[1], reverse=True)
        features = [f[0] for f in sorted_features]
        values = [f[1] for f in sorted_features]
        
        plt.figure(figsize=(12, 8))
        plt.barh(range(len(features)), values, align='center')
        plt.yticks(range(len(features)), features)
        plt.xlabel('Importance')
        plt.title(f'Feature Importance ({target} dimension)')
        
        plot_path = output_path.replace('.pkl', f'_{target}_importance.png')
        plt.tight_layout()
        plt.savefig(plot_path)
        plt.close()
        
        logger.info(f"Saved feature importance plot for {target} to {plot_path}")
    
    # Save SHAP summary plot for each target
    for target in results['shap_values']:
        plt.figure()
        shap.summary_plot(
            results['shap_values'][target],
            features=results['feature_info']['feature_columns'],
            feature_names=results['feature_info']['feature_columns'],
            show=False
        )
        
        shap_path = output_path.replace('.pkl', f'_{target}_shap.png')
        plt.tight_layout()
        plt.savefig(shap_path)
        plt.close()
        
        logger.info(f"Saved SHAP plot for {target} to {shap_path}")

def main():
    """Main entry point"""
    parser = argparse.ArgumentParser(description="Train multi-task GBT model for medication response prediction")
    parser.add_argument("--data", required=True, help="Input parquet file with features")
    parser.add_argument("--output", required=True, help="Output path for trained model")
    parser.add_argument("--config", required=True, help="Configuration YAML file")
    parser.add_argument("--seed", type=int, default=42, help="Random seed for reproducibility")
    
    args = parser.parse_args()
    
    try:
        # Load configuration
        config = load_config(args.config)
        
        # Load feature data
        features_df = pd.read_parquet(args.data)
        logger.info(f"Loaded {len(features_df)} records from {args.data}")
        
        # Prepare data for training
        data_dict, feature_info = prepare_data(features_df, config, args.seed)
        
        # Train multi-task model
        results = train_multi_task_model(data_dict, config, feature_info, args.seed)
        
        # Save model results
        save_model_results(results, args.output)
        
        logger.info("Model training complete!")
        
    except Exception as e:
        logger.error(f"Error training model: {e}", exc_info=True)
        exit(1)

if __name__ == "__main__":
    main() 
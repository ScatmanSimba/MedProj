# -*- coding: utf-8 -*-
"""shap_analysis.py - SHAP-based feature importance analysis

This module generates SHAP explanations for the trained XGBoost models
to provide feature importance analysis and interpretation.

Usage:
    python -m src.evaluation.shap_analysis \
        --model models/receptor_predictor.pkl \
        --data data/processed/features_dataset.parquet \
        --output evaluation/shap_analysis
"""

import os
import logging
import pandas as pd
import numpy as np
import argparse
import pickle
from pathlib import Path
from typing import Dict, List, Set, Optional, Any, Tuple, Union
import json
import matplotlib.pyplot as plt
import shap
import xgboost as xgb

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler("shap_analysis.log"),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger("shap_analysis")

def load_model_and_data(model_path: str, data_path: str) -> Tuple[Dict[str, Any], pd.DataFrame]:
    """
    Load model and feature data.

    Args:
        model_path: Path to model pickle file
        data_path: Path to feature data parquet file

    Returns:
        Tuple of (model_results, feature_df)
    """
    logger.info(f"Loading model from {model_path}")
    with open(model_path, 'rb') as f:
        model_results = pickle.load(f)
    
    logger.info(f"Loading feature data from {data_path}")
    feature_df = pd.read_parquet(data_path)
    
    return model_results, feature_df

def generate_shap_explanations(model_results: Dict[str, Any], 
                              feature_df: pd.DataFrame,
                              output_dir: str,
                              sample_size: int = 100) -> Dict[str, Any]:
    """
    Generate SHAP explanations and visualizations.

    Args:
        model_results: Model results dictionary
        feature_df: Feature DataFrame
        output_dir: Output directory for visualizations
        sample_size: Sample size for SHAP analysis (for efficiency)

    Returns:
        Dictionary with SHAP analysis results
    """
    logger.info(f"Generating SHAP explanations (sample size: {sample_size})")
    
    # Create output directory
    os.makedirs(output_dir, exist_ok=True)
    
    # Get models and feature information
    models = model_results['models']
    feature_info = model_results['feature_info']
    feature_columns = feature_info['feature_columns']
    
    # Take a sample of the data for analysis
    if len(feature_df) > sample_size:
        feature_sample = feature_df.sample(sample_size, random_state=42)
    else:
        feature_sample = feature_df
    
    # Prepare feature data
    X = feature_sample[feature_columns]
    
    # Create results dictionary
    results = {
        'feature_importance': {},
        'shap_values': {},
        'receptor_importance': {},
        'summary': {}
    }
    
    # For each target dimension
    for target, model in models.items():
        logger.info(f"Analyzing SHAP values for {target} dimension")
        
        # Initialize TreeExplainer
        explainer = shap.TreeExplainer(model)
        
        # Calculate SHAP values
        shap_values = explainer.shap_values(X)
        
        # Store SHAP values
        results['shap_values'][target] = shap_values
        
        # Create summary plot
        plt.figure(figsize=(12, 8))
        shap.summary_plot(
            shap_values,
            X,
            feature_names=feature_columns,
            show=False
        )
        plt.title(f"SHAP Summary Plot - {target} Dimension")
        plt.tight_layout()
        summary_path = os.path.join(output_dir, f"{target}_summary_plot.png")
        plt.savefig(summary_path)
        plt.close()
        
        # Create bar plot of feature importance
        plt.figure(figsize=(12, 8))
        shap.summary_plot(
            shap_values,
            X,
            feature_names=feature_columns,
            plot_type="bar",
            show=False
        )
        plt.title(f"SHAP Feature Importance - {target} Dimension")
        plt.tight_layout()
        bar_path = os.path.join(output_dir, f"{target}_importance_bar.png")
        plt.savefig(bar_path)
        plt.close()
        
        # Create dependence plots for top features
        # Calculate mean absolute SHAP value for each feature
        mean_abs_shap = np.abs(shap_values).mean(axis=0)
        feature_importance = dict(zip(feature_columns, mean_abs_shap))
        
        # Sort features by importance
        sorted_features = sorted(feature_importance.items(), key=lambda x: x[1], reverse=True)
        top_features = [f[0] for f in sorted_features[:5]]  # Top 5 features
        
        for feature in top_features:
            plt.figure(figsize=(10, 6))
            feature_idx = feature_columns.index(feature)
            shap.dependence_plot(
                feature_idx,
                shap_values,
                X,
                feature_names=feature_columns,
                show=False
            )
            plt.title(f"SHAP Dependence Plot - {feature} ({target} dimension)")
            plt.tight_layout()
            dep_path = os.path.join(output_dir, f"{target}_{feature}_dependence.png")
            plt.savefig(dep_path)
            plt.close()
        
        # Store feature importance
        results['feature_importance'][target] = feature_importance
        
        # Calculate receptor importance (sum of receptor features)
        receptor_features = feature_info['receptor_features']
        receptor_importance = {f: feature_importance.get(f, 0) for f in receptor_features}
        results['receptor_importance'][target] = receptor_importance
        
        # Check if receptor features are in top features (PRD success criteria)
        top_feature_names = [f[0] for f in sorted_features[:5]]
        receptor_in_top = sum(1 for f in top_feature_names if f in receptor_features)
        
        results['summary'][target] = {
            'receptor_in_top5': receptor_in_top,
            'top5_features': top_feature_names,
            'mean_abs_shap': {f: float(mean_abs_shap[feature_columns.index(f)]) for f in top_feature_names}
        }
    
    # Evaluate success criteria from PRD
    success_criteria_met = sum(1 for target in results['summary'] if results['summary'][target]['receptor_in_top5'] >= 3)
    
    # Generate summary report
    summary = {
        'success_criteria_met': success_criteria_met >= 2,  # Success if at least 2 of 3 dimensions meet criteria
        'dimensions_meeting_criteria': success_criteria_met,
        'dimension_summaries': results['summary']
    }
    
    results['overall_summary'] = summary
    
    # Save summary to JSON
    summary_path = os.path.join(output_dir, "shap_summary.json")
    with open(summary_path, 'w') as f:
        json.dump(summary, f, indent=2)
    
    # Check success criteria
    if summary['success_criteria_met']:
        logger.info("SUCCESS: At least 3 receptor dimensions are in the top-5 features")
    else:
        logger.warning(f"Model does not meet success criteria - only {summary['dimensions_meeting_criteria']} dimensions have 3+ receptor features in top-5")
    
    return results

def generate_html_report(results: Dict[str, Any], output_dir: str) -> None:
    """
    Generate a comprehensive HTML report with SHAP analysis results.

    Args:
        results: Results dictionary from generate_shap_explanations
        output_dir: Output directory
    """
    logger.info("Generating HTML report")
    
    html_path = os.path.join(output_dir, "shap_analysis_report.html")
    
    # Prepare data for HTML
    overall_summary = results['overall_summary']
    dimension_data = []
    
    for dimension, summary in results['summary'].items():
        dimension_data.append({
            'dimension': dimension,
            'receptor_in_top5': summary['receptor_in_top5'],
            'top_features': summary['top5_features'],
            'mean_abs_shap': summary['mean_abs_shap']
        })
    
    # Generate HTML content
    html_content = f"""
    <!DOCTYPE html>
    <html>
    <head>
        <title>SHAP Analysis Report</title>
        <style>
            body {{ font-family: Arial, sans-serif; margin: 20px; }}
            h1, h2, h3 {{ color: #333; }}
            .success {{ color: green; }}
            .warning {{ color: orange; }}
            .feature-table {{ border-collapse: collapse; width: 100%; margin-top: 20px; }}
            .feature-table th, .feature-table td {{ border: 1px solid #ddd; padding: 8px; text-align: left; }}
            .feature-table th {{ background-color: #f2f2f2; }}
            .feature-table tr:nth-child(even) {{ background-color: #f9f9f9; }}
            .visualization {{ margin-top: 30px; }}
            .viz-container {{ display: flex; flex-wrap: wrap; gap: 20px; }}
            .viz-item {{ margin-bottom: 30px; }}
        </style>
    </head>
    <body>
        <h1>SHAP Analysis Report</h1>
        
        <h2>Overall Summary</h2>
        <p class="{('success' if overall_summary['success_criteria_met'] else 'warning')}">
            Success Criteria Met: {overall_summary['success_criteria_met']}
        </p>
        <p>{overall_summary['dimensions_meeting_criteria']} of 3 dimensions have at least 3 receptor features in the top-5</p>
        
        <h2>Dimension Summaries</h2>
    """
    
    # Add dimension summaries
    for dim_data in dimension_data:
        html_content += f"""
        <h3>{dim_data['dimension'].title()} Dimension</h3>
        <p>Receptor features in top-5: {dim_data['receptor_in_top5']}/5</p>
        
        <table class="feature-table">
            <tr>
                <th>Rank</th>
                <th>Feature</th>
                <th>Mean |SHAP|</th>
                <th>Is Receptor</th>
            </tr>
        """
        
        receptor_features = results['receptor_importance'][dim_data['dimension']].keys()
        
        for i, feature in enumerate(dim_data['top_features']):
            is_receptor = feature in receptor_features
            html_content += f"""
            <tr>
                <td>{i+1}</td>
                <td>{feature}</td>
                <td>{dim_data['mean_abs_shap'][feature]:.6f}</td>
                <td>{is_receptor}</td>
            </tr>
            """
        
        html_content += """
        </table>
        """
    
    # Add visualizations
    html_content += """
        <h2>Visualizations</h2>
        <div class="viz-container">
    """
    
    # Get all visualization files
    viz_files = [f for f in os.listdir(output_dir) if f.endswith('.png')]
    
    for viz_file in viz_files:
        viz_name = viz_file.replace('.png', '').replace('_', ' ').title()
        html_content += f"""
            <div class="viz-item">
                <h3>{viz_name}</h3>
                <img src="{viz_file}" alt="{viz_name}" width="100%"/>
            </div>
        """
    
    html_content += """
        </div>
    </body>
    </html>
    """
    
    # Write HTML file
    with open(html_path, 'w') as f:
        f.write(html_content)
    
    logger.info(f"HTML report saved to {html_path}")

def main():
    """Main entry point"""
    parser = argparse.ArgumentParser(description="Generate SHAP-based feature importance analysis")
    parser.add_argument("--model", required=True, help="Path to model pickle file")
    parser.add_argument("--data", required=True, help="Path to feature data parquet file")
    parser.add_argument("--output", required=True, help="Output directory for visualizations")
    parser.add_argument("--sample", type=int, default=100, help="Sample size for SHAP analysis")
    parser.add_argument("--html", action="store_true", help="Generate HTML report")
    
    args = parser.parse_args()
    
    try:
        # Load model and data
        model_results, feature_df = load_model_and_data(args.model, args.data)
        
        # Generate SHAP explanations
        results = generate_shap_explanations(
            model_results,
            feature_df,
            args.output,
            args.sample
        )
        
        # Generate HTML report if requested
        if args.html:
            generate_html_report(results, args.output)
        
        logger.info("SHAP analysis complete!")
        
        # Output success status to stdout
        if results['overall_summary']['success_criteria_met']:
            print("SUCCESS: Model meets the receptor feature importance criteria (≥ 3 in top-5 features)")
        else:
            print(f"WARNING: Model does not meet criteria - only {results['overall_summary']['dimensions_meeting_criteria']} dimensions have sufficient receptor features in top-5")
        
    except Exception as e:
        logger.error(f"Error generating SHAP analysis: {e}", exc_info=True)
        exit(1)

if __name__ == "__main__":
    main() 
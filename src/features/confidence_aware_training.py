import matplotlib.pyplot as plt
import numpy as np
from typing import Dict, List, Any, Optional
import logging

logger = logging.getLogger(__name__)

def plot_calibration_curve(confidence_scores: np.ndarray, 
                          true_scores: np.ndarray,
                          n_bins: int = 10) -> Dict[str, Any]:
    """Plot calibration curve using matplotlib.
    
    Args:
        confidence_scores: Array of confidence scores
        true_scores: Array of true scores
        n_bins: Number of bins for calibration curve
        
    Returns:
        Dictionary with plot data
    """
    # Calculate calibration curve
    bin_edges = np.linspace(0, 1, n_bins + 1)
    bin_indices = np.digitize(confidence_scores, bin_edges) - 1
    
    calibration_curve = []
    for i in range(n_bins):
        mask = bin_indices == i
        if np.any(mask):
            pred_mean = np.mean(confidence_scores[mask])
            actual_mean = np.mean(true_scores[mask])
            calibration_curve.append({
                'confidence': pred_mean,
                'accuracy': actual_mean,
                'count': np.sum(mask)
            })
    
    # Create plot
    fig, ax = plt.subplots(figsize=(8, 6))
    
    # Plot calibration curve
    confidences = [point['confidence'] for point in calibration_curve]
    accuracies = [point['accuracy'] for point in calibration_curve]
    counts = [point['count'] for point in calibration_curve]
    
    # Plot points with size proportional to count
    sizes = np.array(counts) * 100 / max(counts)
    ax.scatter(confidences, accuracies, s=sizes, alpha=0.6)
    
    # Plot perfect calibration line
    ax.plot([0, 1], [0, 1], 'k--', alpha=0.5, label='Perfect Calibration')
    
    # Add labels and title
    ax.set_xlabel('Confidence Score')
    ax.set_ylabel('Actual Accuracy')
    ax.set_title('Calibration Curve')
    
    # Add grid
    ax.grid(True, alpha=0.3)
    
    # Add legend
    ax.legend()
    
    return {
        'figure': fig,
        'calibration_curve': calibration_curve
    }

def plot_confidence_distribution(confidence_scores: np.ndarray,
                               true_scores: np.ndarray,
                               n_bins: int = 20) -> Dict[str, Any]:
    """Plot confidence distribution using matplotlib.
    
    Args:
        confidence_scores: Array of confidence scores
        true_scores: Array of true scores
        n_bins: Number of bins for histogram
        
    Returns:
        Dictionary with plot data
    """
    # Create plot
    fig, ax = plt.subplots(figsize=(8, 6))
    
    # Plot histogram
    ax.hist(confidence_scores, bins=n_bins, alpha=0.6, 
            label='All Predictions')
    
    # Plot correct predictions
    correct_mask = np.abs(confidence_scores - true_scores) < 0.1
    ax.hist(confidence_scores[correct_mask], bins=n_bins, alpha=0.6,
            label='Correct Predictions')
    
    # Add labels and title
    ax.set_xlabel('Confidence Score')
    ax.set_ylabel('Count')
    ax.set_title('Confidence Distribution')
    
    # Add grid
    ax.grid(True, alpha=0.3)
    
    # Add legend
    ax.legend()
    
    return {
        'figure': fig,
        'correct_ratio': np.mean(correct_mask)
    }

def plot_confidence_vs_error(confidence_scores: np.ndarray,
                           true_scores: np.ndarray) -> Dict[str, Any]:
    """Plot confidence vs error using matplotlib.
    
    Args:
        confidence_scores: Array of confidence scores
        true_scores: Array of true scores
        
    Returns:
        Dictionary with plot data
    """
    # Calculate errors
    errors = np.abs(confidence_scores - true_scores)
    
    # Create plot
    fig, ax = plt.subplots(figsize=(8, 6))
    
    # Plot scatter
    ax.scatter(confidence_scores, errors, alpha=0.6)
    
    # Add trend line
    z = np.polyfit(confidence_scores, errors, 1)
    p = np.poly1d(z)
    ax.plot(confidence_scores, p(confidence_scores), "r--", alpha=0.8)
    
    # Add labels and title
    ax.set_xlabel('Confidence Score')
    ax.set_ylabel('Absolute Error')
    ax.set_title('Confidence vs Error')
    
    # Add grid
    ax.grid(True, alpha=0.3)
    
    return {
        'figure': fig,
        'correlation': np.corrcoef(confidence_scores, errors)[0, 1]
    } 
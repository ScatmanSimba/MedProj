"""Export functions for converting annotations to training format.

This module provides functions to convert raw annotations into the format
expected by the training pipeline.
"""

from typing import Dict, List, Optional
import pandas as pd
import json
from pathlib import Path
import logging
from datetime import datetime

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

def load_annotations(annotation_dir: str) -> pd.DataFrame:
    """Load all annotation files from directory.
    
    Args:
        annotation_dir: Directory containing annotation JSONL files
        
    Returns:
        DataFrame containing all annotations
    """
    annotation_dir = Path(annotation_dir)
    all_annotations = []
    
    # Load each annotation file
    for file in annotation_dir.glob("annotations_*.jsonl"):
        with open(file, 'r') as f:
            for line in f:
                try:
                    annotation = json.loads(line.strip())
                    all_annotations.append(annotation)
                except json.JSONDecodeError as e:
                    logger.warning(f"Error parsing annotation in {file}: {str(e)}")
                    continue
    
    return pd.DataFrame(all_annotations)

def aggregate_annotations(annotations_df: pd.DataFrame) -> pd.DataFrame:
    """Aggregate multiple annotations per post.
    
    Args:
        annotations_df: DataFrame containing raw annotations
        
    Returns:
        DataFrame with aggregated scores
    """
    # Group by post_id
    grouped = annotations_df.groupby('post_id')
    
    # Aggregate scores
    aggregated = pd.DataFrame()
    
    # Mean scores for each dimension
    for dimension in ['activation_sedation', 'emotional_blunting', 'appetite_metabolic']:
        aggregated[f"{dimension}_score"] = grouped[dimension].mean()
        aggregated[f"{dimension}_std"] = grouped[dimension].std()
    
    # Mean confidence
    aggregated['confidence'] = grouped['confidence'].mean()
    
    # Number of annotations
    aggregated['annotation_count'] = grouped.size()
    
    # Add metadata
    aggregated['last_updated'] = grouped['timestamp'].max()
    
    return aggregated

def export_to_training_format(
    gold_set_path: str,
    annotation_dir: str,
    output_path: str,
    min_annotations: int = 2,
    min_confidence: float = 0.7
) -> pd.DataFrame:
    """Export annotations to training format.
    
    Args:
        gold_set_path: Path to gold set parquet file
        annotation_dir: Directory containing annotation files
        output_path: Path to save training data
        min_annotations: Minimum number of annotations per post
        min_confidence: Minimum confidence threshold
        
    Returns:
        DataFrame in training format
    """
    logger.info("Loading and processing annotations...")
    
    # Load gold set
    gold_set = pd.read_parquet(gold_set_path)
    
    # Load annotations
    annotations = load_annotations(annotation_dir)
    
    # Aggregate annotations
    aggregated = aggregate_annotations(annotations)
    
    # Filter by minimum annotations and confidence
    valid_posts = aggregated[
        (aggregated['annotation_count'] >= min_annotations) &
        (aggregated['confidence'] >= min_confidence)
    ]
    
    # Merge with gold set
    training_data = gold_set.join(valid_posts)
    
    # Select and rename columns for training
    training_data = training_data[[
        'post_id',
        'post_text',
        'medication',
        'activation_sedation_score',
        'emotional_blunting_score',
        'appetite_metabolic_score',
        'activation_sedation_std',
        'emotional_blunting_std',
        'appetite_metabolic_std',
        'confidence',
        'annotation_count'
    ]]
    
    # Save to parquet
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    training_data.to_parquet(output_path)
    
    logger.info(f"Exported {len(training_data)} posts to {output_path}")
    return training_data

if __name__ == "__main__":
    # Example usage
    training_data = export_to_training_format(
        gold_set_path="data/annotation/gold_set.parquet",
        annotation_dir="data/annotation",
        output_path="data/processed/training_data.parquet",
        min_annotations=2,
        min_confidence=0.7
    ) 
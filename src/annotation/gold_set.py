"""Gold set creation for medication response annotation.

This module handles the creation of a balanced gold set for annotation,
ensuring good coverage across response dimensions and medication types.
"""

from typing import List, Dict, Optional, Tuple
import pandas as pd
import numpy as np
from pathlib import Path
import logging
from ..features.response_classifier import ResponseDimensionClassifier

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

def get_preliminary_scores(
    df: pd.DataFrame,
    dimension: str,
    batch_size: int = 32
) -> Tuple[pd.Series, pd.Series]:
    """Get preliminary scores for a dimension using the response classifier.
    
    Args:
        df: DataFrame containing Reddit posts
        dimension: Response dimension to score
        batch_size: Number of posts to process at once
        
    Returns:
        Tuple of (scores, uncertainties) Series indexed by post ID
    """
    classifier = ResponseDimensionClassifier()
    all_scores = []
    all_uncertainties = []
    
    # Process in batches
    for i in range(0, len(df), batch_size):
        batch = df.iloc[i:i+batch_size]
        try:
            # Get predictions with uncertainty
            scores = classifier.predict(
                batch['post_text'].tolist(),
                return_uncertainty=True
            )
            
            # Debug prints
            logger.info(f"Batch {i//batch_size + 1}:")
            logger.info(f"Scores returned for batch: {scores.keys()}")
            if 'uncertainties' in scores:
                logger.info(f"Uncertainty keys: {scores['uncertainties'].keys()}")
                logger.info(f"Sample uncertainty values for {dimension}: {scores['uncertainties'][dimension][:2]}")
            
            # Extract scores and uncertainties with error handling
            try:
                all_scores.extend(scores[dimension])
                all_uncertainties.extend(scores['uncertainties'][dimension])
            except Exception as e:
                logger.error(f"Error processing scores: {e}")
                logger.error(f"Full scores structure: {scores}")
                # Fall back to default values
                all_scores.extend([0.5] * len(batch))
                all_uncertainties.extend([1.0] * len(batch))
            
        except Exception as e:
            logger.warning(f"Error processing batch {i//batch_size}: {str(e)}")
            # Fill with neutral scores and high uncertainty for failed batch
            all_scores.extend([0.5] * len(batch))
            all_uncertainties.extend([1.0] * len(batch))
    
    return (
        pd.Series(all_scores, index=df.index),
        pd.Series(all_uncertainties, index=df.index)
    )

def create_gold_set(
    reddit_posts_df: pd.DataFrame,
    n_per_axis: int = 100,
    min_confidence: float = 0.7,
    batch_size: int = 32,
    output_dir: Optional[str] = None
) -> pd.DataFrame:
    """Create balanced gold set for annotation.
    
    Args:
        reddit_posts_df: DataFrame containing Reddit posts
        n_per_axis: Number of posts to sample per dimension
        min_confidence: Minimum confidence threshold for preliminary scores
        batch_size: Number of posts to process at once
        output_dir: Directory to save gold set (optional)
        
    Returns:
        DataFrame containing the gold set
    """
    logger.info("Creating gold set for annotation...")
    
    # Get preliminary scores for all dimensions
    prelim_scores = {}
    for dimension in ['activation_sedation', 'emotional_blunting', 'appetite_metabolic']:
        logger.info(f"Getting preliminary scores for {dimension}...")
        scores, uncertainties = get_preliminary_scores(
            reddit_posts_df,
            dimension,
            batch_size=batch_size
        )
        prelim_scores[dimension] = scores
        prelim_scores[f"{dimension}_uncertainty"] = uncertainties
    
    # Create scoring DataFrame
    scores_df = pd.DataFrame(prelim_scores)
    
    # Filter by confidence
    high_confidence = (
        (scores_df['activation_sedation_uncertainty'] < (1 - min_confidence)) &
        (scores_df['emotional_blunting_uncertainty'] < (1 - min_confidence)) &
        (scores_df['appetite_metabolic_uncertainty'] < (1 - min_confidence))
    )
    scores_df = scores_df[high_confidence]
    
    # Sample posts for each dimension
    gold_posts = []
    for dimension in ['activation_sedation', 'emotional_blunting', 'appetite_metabolic']:
        # Sample from extremes
        low_extreme = scores_df.nsmallest(n_per_axis//2, dimension).sample(n_per_axis//2)
        high_extreme = scores_df.nlargest(n_per_axis//2, dimension).sample(n_per_axis//2)
        
        gold_posts.extend(low_extreme.index.tolist())
        gold_posts.extend(high_extreme.index.tolist())
    
    # Remove duplicates while maintaining diversity
    gold_set = reddit_posts_df.loc[list(set(gold_posts))]
    
    # Add random samples for calibration
    random_samples = reddit_posts_df.sample(n=50)
    gold_set = pd.concat([gold_set, random_samples]).drop_duplicates()
    
    # Add preliminary scores and metadata
    gold_set = gold_set.join(scores_df)
    
    # Add annotation status
    gold_set['annotation_status'] = 'pending'
    gold_set['annotation_count'] = 0
    
    # Save if output directory provided
    if output_dir:
        output_path = Path(output_dir) / 'gold_set.parquet'
        output_path.parent.mkdir(parents=True, exist_ok=True)
        gold_set.to_parquet(output_path)
        logger.info(f"Saved gold set to {output_path}")
    
    logger.info(f"Created gold set with {len(gold_set)} posts")
    return gold_set

if __name__ == "__main__":
    # Example usage
    reddit_posts = pd.read_parquet("data/processed/reddit_posts.parquet")
    gold_set = create_gold_set(
        reddit_posts,
        n_per_axis=100,
        min_confidence=0.7,
        batch_size=32,
        output_dir="data/annotation"
    ) 
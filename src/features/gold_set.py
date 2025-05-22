from typing import List, Optional, Dict, Any

def calculate_post_uncertainty(row):
    """
    Calculate uncertainty metrics for a post using MC-Dropout.
    
    Args:
        row: DataFrame row containing post data
        
    Returns:
        pd.Series with attribution results and uncertainty metrics
    """
    if not row['medications'] or not isinstance(row['medications'], list):
        return pd.Series({
            'response_dimension_scores': {}, 
            'response_dimension_confidence': {},
            'response_dimension_variances': {},
            'med_to_sentences': {},
            'uncertainty': 1.0,  # Maximum uncertainty when no medications
            'uncertainty_components': {
                'mean_variance': 1.0,
                'confidence_weighted': 1.0,
                'max_variance': 1.0
            }
        })
    
    # Run attribution with uncertainty using MC-Dropout
    results = self.attributor.attribute_with_uncertainty(
        row['post_text'], 
        row['medications'],
        n_samples=10,  # Use default number of samples
        use_cache=True
    )
    
    # Extract uncertainty components
    uncertainties = []
    variances = []
    confidences = []
    
    # Process each medication and dimension
    for med, dims in results['dimension_variances'].items():
        for dim, variance in dims.items():
            variances.append(variance)
            # Get corresponding confidence
            confidence = results['dimension_confidence'][med][dim]
            confidences.append(confidence)
            # Calculate uncertainty as inverse of confidence weighted by variance
            uncertainty = (1 - confidence) * (1 + variance)
            uncertainties.append(uncertainty)
    
    # Calculate different uncertainty metrics
    mean_variance = np.mean(variances) if variances else 1.0
    max_variance = np.max(variances) if variances else 1.0
    
    # Calculate confidence-weighted uncertainty
    if confidences and variances:
        confidence_weighted = np.average(variances, weights=confidences)
    else:
        confidence_weighted = 1.0
    
    # Calculate composite uncertainty score
    # Weight different components based on their importance
    composite_uncertainty = (
        0.4 * mean_variance +  # Average variance across all dimensions
        0.4 * confidence_weighted +  # Confidence-weighted variance
        0.2 * max_variance  # Maximum variance as a safety factor
    )
    
    # Normalize to [0,1] range
    composite_uncertainty = min(1.0, composite_uncertainty)
    
    return pd.Series({
        'response_dimension_scores': results['dimension_scores'],
        'response_dimension_confidence': results['dimension_confidence'],
        'response_dimension_variances': results['dimension_variances'],
        'med_to_sentences': results.get('med_to_sentences', {}),
        'uncertainty': composite_uncertainty,
        'uncertainty_components': {
            'mean_variance': mean_variance,
            'confidence_weighted': confidence_weighted,
            'max_variance': max_variance
        }
    })

def create_gold_set(self, 
                   posts_df: pd.DataFrame,
                   sample_size: int = 100,
                   uncertainty_threshold: float = 0.3,
                   random_state: int = 42) -> pd.DataFrame:
    """
    Create gold set by sampling posts with model uncertainty guidance.
    
    Args:
        posts_df: DataFrame containing posts
        sample_size: Number of posts to sample
        uncertainty_threshold: Maximum uncertainty for confident samples
        random_state: Random seed for reproducibility
        
    Returns:
        DataFrame containing gold set
    """
    logger.info(f"Creating gold set with {sample_size} posts")
    
    # Apply attribution to all posts to get uncertainty scores
    logger.info("Calculating attribution uncertainty for all posts...")
    posts_df[['response_dimension_scores', 'response_dimension_confidence', 
             'response_dimension_variances', 'med_to_sentences', 
             'uncertainty', 'uncertainty_components']] = posts_df.apply(
        calculate_post_uncertainty, axis=1)
    
    # Split into confident and uncertain groups
    confident_posts = posts_df[posts_df['uncertainty'] < uncertainty_threshold]
    uncertain_posts = posts_df[posts_df['uncertainty'] >= uncertainty_threshold]
    
    logger.info(f"Found {len(confident_posts)} posts with low uncertainty (<{uncertainty_threshold})")
    logger.info(f"Found {len(uncertain_posts)} posts with high uncertainty (≥{uncertainty_threshold})")
    
    # Sample from both groups with 70/30 split
    confident_sample_size = int(sample_size * 0.7)
    uncertain_sample_size = sample_size - confident_sample_size
    
    # Adjust if we don't have enough posts in either group
    if len(confident_posts) < confident_sample_size:
        logger.warning(f"Not enough confident posts. Adjusting sampling strategy.")
        confident_sample_size = min(len(confident_posts), confident_sample_size)
        uncertain_sample_size = sample_size - confident_sample_size
    
    if len(uncertain_posts) < uncertain_sample_size:
        logger.warning(f"Not enough uncertain posts. Adjusting sampling strategy.")
        uncertain_sample_size = min(len(uncertain_posts), uncertain_sample_size)
        confident_sample_size = sample_size - confident_sample_size
    
    # Sample from each group
    confident_sample = confident_posts.sample(n=confident_sample_size, random_state=random_state)
    uncertain_sample = uncertain_posts.sample(n=uncertain_sample_size, random_state=random_state)
    
    # Combine samples
    gold_set = pd.concat([confident_sample, uncertain_sample], ignore_index=True)
    
    # Save gold set
    gold_set.to_parquet(self.gold_set_path)
    logger.info(f"Saved gold set to {self.gold_set_path}")
    logger.info(f"Gold set composition: {confident_sample_size} confident posts, {uncertain_sample_size} uncertain posts")
    
    # Log uncertainty statistics
    logger.info("Uncertainty statistics for gold set:")
    logger.info(f"Mean uncertainty: {gold_set['uncertainty'].mean():.3f}")
    logger.info(f"Mean variance: {gold_set['uncertainty_components'].apply(lambda x: x['mean_variance']).mean():.3f}")
    logger.info(f"Mean confidence-weighted uncertainty: {gold_set['uncertainty_components'].apply(lambda x: x['confidence_weighted']).mean():.3f}")
    
    return gold_set

def process_text(self, text: str, medications: Optional[List[str]] = None) -> Dict[str, Any]:
    """Process text with optional pre-extracted medications.
    
    Args:
        text: Input text
        medications: Optional list of pre-extracted medications
        
    Returns:
        Dictionary with processed results
    """
    # Parse text with spaCy
    doc = self.nlp(text)
    
    # Extract medications if not provided
    if medications is None:
        medications = [ent.text for ent in doc.ents if ent.label_ == "MEDICATION"]
        logger.debug(f"Extracted medications: {medications}")
    else:
        logger.debug(f"Using provided medications: {medications}")
    
    # Get temporal information
    temporal_info = self.temporal_parser.parse_temporal_info(doc)
    
    # Get emoji signals
    emoji_signals = self.emoji_processor.process_emoji_signals(doc)
    
    return {
        'text': text,
        'medications': medications,
        'temporal_info': temporal_info,
        'emoji_signals': emoji_signals
    } 
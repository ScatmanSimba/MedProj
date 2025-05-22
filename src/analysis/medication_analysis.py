"""
Analyze medication distribution and quality in the cleaned Reddit dataset.

This script:
1. Analyzes the distribution of medications across posts
2. Checks the quality of medication mentions
3. Generates visualizations and statistics
"""

import pandas as pd
import numpy as np
from collections import Counter
import matplotlib.pyplot as plt
import seaborn as sns
from typing import Dict, List, Tuple
import json
from pathlib import Path
import re

def load_cleaned_data(file_path: str) -> pd.DataFrame:
    """Load the cleaned Reddit posts dataset."""
    return pd.read_csv(file_path)

def analyze_medication_distribution(df: pd.DataFrame) -> Dict:
    """
    Analyze the distribution of medications in the dataset.
    
    Returns:
        Dictionary containing distribution statistics
    """
    # Convert medication strings to lists
    df['medications'] = df['medications'].apply(
        lambda x: x.split(',') if isinstance(x, str) else []
    )
    
    # Count medication mentions
    all_meds = []
    for meds in df['medications']:
        all_meds.extend(meds)
    
    med_counts = Counter(all_meds)
    
    # Calculate statistics
    total_posts = len(df)
    posts_with_meds = df['medications'].apply(len).gt(0).sum()
    
    stats = {
        'total_posts': total_posts,
        'posts_with_medications': posts_with_meds,
        'percentage_with_medications': (posts_with_meds / total_posts) * 100,
        'total_medication_mentions': len(all_meds),
        'unique_medications': len(med_counts),
        'medication_counts': dict(med_counts.most_common()),
        'posts_with_multiple_meds': df['medications'].apply(len).gt(1).sum(),
        'avg_medications_per_post': len(all_meds) / total_posts
    }
    
    return stats

def analyze_medication_quality(df: pd.DataFrame) -> Dict:
    """
    Analyze the quality of medication mentions.
    
    Returns:
        Dictionary containing quality metrics
    """
    # Convert medication strings to lists if not already
    df['medications'] = df['medications'].apply(
        lambda x: x.split(',') if isinstance(x, str) else []
    )
    
    # Calculate quality metrics
    quality_stats = {
        'posts_with_exact_med_names': 0,
        'posts_with_abbreviations': 0,
        'posts_with_dosage_info': 0,
        'posts_with_duration_info': 0,
        'medication_context_quality': []
    }
    
    # Common medication abbreviations
    common_abbrevs = {
        'ssri': ['zoloft', 'prozac', 'lexapro', 'paxil', 'celexa'],
        'snri': ['effexor', 'cymbalta', 'pristiq'],
        'ndri': ['wellbutrin', 'bupropion'],
        'tca': ['amitriptyline', 'nortriptyline', 'imipramine'],
        'maoi': ['nardil', 'parnate', 'selegiline'],
        'antipsychotic': ['abilify', 'risperdal', 'zyprexa', 'seroquel'],
        'mood_stabilizer': ['lithium', 'lamictal', 'depakote']
    }
    
    # Check each post
    for _, row in df.iterrows():
        post_quality = {
            'has_exact_name': False,
            'has_abbreviation': False,
            'has_dosage': False,
            'has_duration': False,
            'context_quality': 'low'
        }
        
        text = f"{row['title']} {row['selftext']}".lower()
        meds = [med.lower() for med in row['medications']]
        
        # Check for exact medication names
        if any(med in text for med in meds):
            post_quality['has_exact_name'] = True
            quality_stats['posts_with_exact_med_names'] += 1
        
        # Check for abbreviations
        for abbrev, meds_list in common_abbrevs.items():
            if abbrev in text and any(med in meds for med in meds_list):
                post_quality['has_abbreviation'] = True
                quality_stats['posts_with_abbreviations'] += 1
                break
        
        # Check for dosage information
        dosage_patterns = [
            r'\d+\s*(?:mg|g|ml|mcg)',
            r'\d+\s*(?:milligram|gram|milliliter|microgram)',
            r'\d+\s*(?:mg|g|ml|mcg)/\d+\s*(?:mg|g|ml|mcg)'
        ]
        if any(re.search(pattern, text) for pattern in dosage_patterns):
            post_quality['has_dosage'] = True
            quality_stats['posts_with_dosage_info'] += 1
        
        # Check for duration information
        duration_patterns = [
            r'\d+\s*(?:day|week|month|year)s?',
            r'\d+\s*(?:d|w|m|y)',
            r'(?:since|for|taking)\s+\d+\s*(?:day|week|month|year)s?'
        ]
        if any(re.search(pattern, text) for pattern in duration_patterns):
            post_quality['has_duration'] = True
            quality_stats['posts_with_duration_info'] += 1
        
        # Assess context quality
        context_score = sum([
            post_quality['has_exact_name'],
            post_quality['has_abbreviation'],
            post_quality['has_dosage'],
            post_quality['has_duration']
        ])
        
        if context_score >= 3:
            post_quality['context_quality'] = 'high'
        elif context_score >= 1:
            post_quality['context_quality'] = 'medium'
        
        quality_stats['medication_context_quality'].append(post_quality)
    
    # Calculate percentages
    total_posts = len(df)
    quality_stats['percentage_exact_names'] = (quality_stats['posts_with_exact_med_names'] / total_posts) * 100
    quality_stats['percentage_with_abbreviations'] = (quality_stats['posts_with_abbreviations'] / total_posts) * 100
    quality_stats['percentage_with_dosage'] = (quality_stats['posts_with_dosage_info'] / total_posts) * 100
    quality_stats['percentage_with_duration'] = (quality_stats['posts_with_duration_info'] / total_posts) * 100
    
    # Calculate context quality distribution
    context_quality_counts = Counter(
        item['context_quality'] for item in quality_stats['medication_context_quality']
    )
    quality_stats['context_quality_distribution'] = dict(context_quality_counts)
    
    return quality_stats

def generate_visualizations(dist_stats: Dict, quality_stats: Dict, output_dir: str):
    """Generate visualizations for the analysis."""
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    
    # 1. Top medications bar plot
    plt.figure(figsize=(12, 6))
    top_meds = dict(sorted(dist_stats['medication_counts'].items(), 
                          key=lambda x: x[1], 
                          reverse=True)[:20])
    plt.bar(top_meds.keys(), top_meds.values())
    plt.xticks(rotation=45, ha='right')
    plt.title('Top 20 Most Mentioned Medications')
    plt.tight_layout()
    plt.savefig(output_path / 'top_medications.png')
    plt.close()
    
    # 2. Context quality pie chart
    plt.figure(figsize=(8, 8))
    context_quality = quality_stats['context_quality_distribution']
    plt.pie(context_quality.values(), 
            labels=context_quality.keys(),
            autopct='%1.1f%%')
    plt.title('Distribution of Medication Context Quality')
    plt.savefig(output_path / 'context_quality.png')
    plt.close()
    
    # 3. Quality metrics bar plot
    plt.figure(figsize=(10, 6))
    quality_metrics = {
        'Exact Names': quality_stats['percentage_exact_names'],
        'Abbreviations': quality_stats['percentage_with_abbreviations'],
        'Dosage Info': quality_stats['percentage_with_dosage'],
        'Duration Info': quality_stats['percentage_with_duration']
    }
    plt.bar(quality_metrics.keys(), quality_metrics.values())
    plt.title('Percentage of Posts with Different Quality Metrics')
    plt.ylabel('Percentage')
    plt.xticks(rotation=45, ha='right')
    plt.tight_layout()
    plt.savefig(output_path / 'quality_metrics.png')
    plt.close()

def main():
    # Load data
    data_path = "data/processed/reddit_posts_clean.csv"
    df = load_cleaned_data(data_path)
    
    # Analyze distribution
    dist_stats = analyze_medication_distribution(df)
    
    # Analyze quality
    quality_stats = analyze_medication_quality(df)
    
    # Generate visualizations
    generate_visualizations(dist_stats, quality_stats, "data/analysis/medication")
    
    # Save statistics
    stats = {
        'distribution': dist_stats,
        'quality': quality_stats
    }
    
    def convert(o):
        if isinstance(o, dict):
            return {k: convert(v) for k, v in o.items()}
        elif isinstance(o, list):
            return [convert(i) for i in o]
        elif isinstance(o, (np.integer, np.floating)):
            return o.item()
        else:
            return o
    stats = convert(stats)
    with open("data/analysis/medication/analysis_stats.json", 'w') as f:
        json.dump(stats, f, indent=2)
    
    # Print summary
    print("\nMedication Distribution Summary:")
    print(f"Total posts: {dist_stats['total_posts']}")
    print(f"Posts with medications: {dist_stats['posts_with_medications']} ({dist_stats['percentage_with_medications']:.1f}%)")
    print(f"Unique medications: {dist_stats['unique_medications']}")
    print(f"Average medications per post: {dist_stats['avg_medications_per_post']:.2f}")
    
    print("\nMedication Quality Summary:")
    print(f"Posts with exact medication names: {quality_stats['posts_with_exact_med_names']} ({quality_stats['percentage_exact_names']:.1f}%)")
    print(f"Posts with dosage information: {quality_stats['posts_with_dosage_info']} ({quality_stats['percentage_with_dosage']:.1f}%)")
    print(f"Posts with duration information: {quality_stats['posts_with_duration_info']} ({quality_stats['percentage_with_duration']:.1f}%)")
    
    print("\nContext Quality Distribution:")
    for quality, count in quality_stats['context_quality_distribution'].items():
        print(f"{quality}: {count} posts ({(count/len(df))*100:.1f}%)")

if __name__ == "__main__":
    main() 
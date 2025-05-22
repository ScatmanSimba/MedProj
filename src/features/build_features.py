#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""build_features.py - Feature engineering for medication response prediction

This module builds features for the medication response prediction model.
It combines user timeline data with drug receptor data and creates
features for the model.

Usage:
    python -m src.features.build_features \
        --input data/processed/annotated_responses.parquet \
        --drug_data data/external/drug_receptor_affinity.csv \
        --output data/processed/features_dataset.parquet
"""

import os
import pandas as pd
import numpy as np
import argparse
import re
from typing import Dict, List, Optional, Set
from sklearn.preprocessing import StandardScaler
import joblib
import json
import logging
from sklearn.feature_extraction.text import TfidfVectorizer

# Configure logging
logger = logging.getLogger(__name__)

class FeatureBuilder:
    """Feature builder for medication response prediction."""
    
    def __init__(self, receptor_data: pd.DataFrame, config: Optional[Dict] = None):
        """Initialize feature builder with receptor data.
        
        Args:
            receptor_data: DataFrame with drug receptor affinities
            config: Optional configuration dictionary
        """
        self.receptor_data = receptor_data
        self.receptor_columns = [
            'd2_affinity', '5ht2a_affinity', 'h1_affinity', 
            'alpha1_affinity', 'm1_affinity'
        ]
        self._med_cache = None  # Cache for median values
        
        # Load configuration with defaults
        self.config = config or {}
        self.use_tfidf = self.config.get('use_tfidf', True)
        self.scale_tfidf = self.config.get('scale_tfidf', False)  # Default to False for tree models
        self.debug_mode = self.config.get('debug_mode', False)
        self.debug_dir = self.config.get('debug_dir', 'debug/features')
        self.strict_vectorizer = self.config.get('strict_vectorizer', False)
        
        # TF-IDF vectorizer configuration
        self.load_existing_vectorizer = self.config.get('load_existing_vectorizer', False)
        self.vectorizer_path = self.config.get('vectorizer_path', 'models/vectorizers/tfidf_vectorizer.joblib')
        self.vectorizer = None
        
        # Initialize spaCy and matchers
        try:
            import spacy
            from spacy.matcher import PhraseMatcher
            self.nlp = spacy.load("en_core_web_sm")
            self.has_spacy = True
            logger.info("Successfully loaded spaCy model")
        except Exception as e:
            logger.warning(f"Failed to load spaCy model: {e}. Falling back to regex-based matching.")
            self.nlp = None
            self.has_spacy = False
        
        # Word to number mapping for dosage extraction
        self.word2num = {
            'one': 1, 'two': 2, 'three': 3, 'four': 4, 'five': 5,
            'six': 6, 'seven': 7, 'eight': 8, 'nine': 9, 'ten': 10,
            'eleven': 11, 'twelve': 12, 'thirteen': 13, 'fourteen': 14,
            'fifteen': 15, 'sixteen': 16, 'seventeen': 17, 'eighteen': 18,
            'nineteen': 19, 'twenty': 20, 'thirty': 30, 'forty': 40,
            'fifty': 50, 'sixty': 60, 'seventy': 70, 'eighty': 80,
            'ninety': 90, 'hundred': 100, 'half': 0.5
        }
        
        # Pre-compile dosage regex patterns
        self.dosage_patterns = [
            r'{med}\s+(\d+(?:\.\d+)?)\s*mg',
            r'(\d+(?:\.\d+)?)\s*mg\s+{med}',
            r'(\d+(?:\.\d+)?)\s*mg\s+of\s+{med}',
            r'started\s+{med}.*?(\d+(?:\.\d+)?)\s*mg',
            r'on\s+(\d+(?:\.\d+)?)\s*mg\s+{med}',
            r'(\d+(?:\.\d+)?)\s+(?:pills?|tablets?|capsules?)\s+(?:of\s+)?(\d+(?:\.\d+)?)\s*mg\s+{med}',
            r'(?:taking|on|using)\s+(\d+(?:\.\d+)?)\s+(\d+(?:\.\d+)?)\s*mg\s+(?:pills?|tablets?|capsules?)\s+{med}',
            r'{med}\s+(\d+(?:\.\d+)?)\s*g',
            r'(\d+(?:\.\d+)?)\s*g\s+{med}',
            # Additional patterns for common Reddit phrasings
            r'(?:on|taking)\s+{med}\s+(?:at\s+)?(\d+(?:\.\d+)?)\s*mg',  # More specific pattern
            r'(\d+(?:\.\d+)?)\s*mg.*?{med}.*?(?:daily|per day|once|twice)',
            r'{med}\s+(\d+(?:\.\d+)?)\s*(?:mg|milligrams)'
        ]
        
        # Initialize scaler for numeric features
        self.scaler = StandardScaler()
        
        # Initialize matchers if spaCy is available
        if self.has_spacy:
            self.matchers = {}
            self._initialize_matchers()
        
    def _initialize_matchers(self) -> None:
        """Initialize spaCy matchers for each category."""
        if not self.has_spacy:
            return
            
        # Define keyword dictionaries with polarity weights
        symptom_keywords = {
            'activation': {
                # Activation (positive)
                'energetic': 0.9, 'alert': 0.8, 'stimulated': 0.85, 'wired': 0.9,
                'restless': 0.7, 'jittery': 0.75, 'amped': 0.8, 'motivated': 0.85,
                'awake': 0.8, 'insomnia': 0.7, 'can\'t sleep': 0.7,
                'energy': 0.85, 'drive': 0.8, 'focus': 0.75, 'concentration': 0.75,
                'productive': 0.8, 'active': 0.85,
                
                # Sedation (negative)
                'sedated': -0.9, 'drowsy': -0.8, 'tired': -0.7, 'sleepy': -0.8,
                'lethargic': -0.85, 'zonked': -0.9, 'sluggish': -0.75,
                'fatigued': -0.8, 'calm': -0.6, 'relaxed': -0.5
            },
            'emotional': {
                # Blunting (negative)
                'numb': -0.9, 'flat': -0.85, 'blunted': -0.9, 'empty': -0.8,
                'no emotions': -0.9, 'can\'t feel': -0.85, 'dull': -0.75,
                'apathetic': -0.8, 'indifferent': -0.7,
                
                # Restoration (positive)
                'feel again': 0.9, 'emotions back': 0.85, 'more emotional': 0.8,
                'crying': 0.7, 'happy': 0.8, 'sad': 0.7, 'joy': 0.85,
                'excited': 0.8, 'emotional range': 0.75, 'vibrant': 0.85,
                'alive': 0.8, 'responsive': 0.75, 'mood': 0.7, 'emotion': 0.7,
                'anxious': 0.6, 'depressed': 0.6, 'irritable': 0.65, 'angry': 0.65,
                'moody': 0.7
            },
            'metabolic': {
                # Appetite increase (positive)
                'hungry': 0.8, 'craving': 0.85, 'eating more': 0.75,
                'weight gain': 0.7, 'increased appetite': 0.8,
                'always hungry': 0.85, 'snacking': 0.7, 'appetite': 0.7,
                'hunger': 0.8, 'eating': 0.7, 'food': 0.6,
                
                # Appetite decrease (negative)
                'no appetite': -0.9, 'not hungry': -0.8, 'eating less': -0.75,
                'weight loss': -0.7, 'decreased appetite': -0.8,
                'forget to eat': -0.75, 'nausea': -0.6
            }
        }
        
        # Initialize matchers for each category
        for category, keywords in symptom_keywords.items():
            matcher = PhraseMatcher(self.nlp.vocab)
            # Add patterns for multi-word keywords
            multi_word_patterns = [self.nlp(text) for text in keywords.keys() if " " in text]
            if multi_word_patterns:
                matcher.add("MULTI_WORD", multi_word_patterns)
            self.matchers[category] = matcher
        
    def extract_dosage(self, text: str, medication: str) -> Optional[float]:
        """Extract dosage from text for a specific medication.
        
        Args:
            text: Text containing dosage information
            medication: Medication name to look for
            
        Returns:
            Extracted dosage in mg or None if not found
        """
        if not text or not medication:
            return None
            
        text = text.lower()
        med_lower = medication.lower()
        
        # Convert word numbers to digits
        for word, num in self.word2num.items():
            text = re.sub(r'\b' + re.escape(word) + r'\b', str(num), text)
        
        # Track consumed spans to avoid double-counting
        consumed_spans = set()
        
        # Look for med mention in proximity to dosage
        for pattern in self.dosage_patterns:
            pattern = pattern.format(med=re.escape(med_lower))
            matches = re.finditer(pattern, text)
            
            for match in matches:
                # Skip if this span overlaps with a previously consumed span
                span = match.span()
                if any(span[0] <= end and span[1] >= start 
                      for start, end in consumed_spans):
                    continue
                
                # Handle patterns with two numbers (pills * mg)
                if len(match.groups()) > 1:
                    num_pills = float(match.group(1))
                    mg_per_pill = float(match.group(2))
                    dosage = num_pills * mg_per_pill
                else:
                    dosage_str = match.group(1)
                    dosage = float(dosage_str)
                
                # Convert grams to mg if needed
                if 'g' in match.group(0) and 'mg' not in match.group(0):
                    dosage *= 1000
                
                # Sanity check: reasonable range for psychiatric meds
                if 0.1 <= dosage <= 2000:
                    consumed_spans.add(span)
                    return dosage
        
        return None
    
    def build_receptor_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Build receptor-based features.
        
        Args:
            df: DataFrame with medication information
            
        Returns:
            DataFrame with receptor features added
        """
        # Merge with receptor data
        df = df.merge(
            self.receptor_data,
            on='medication',
            how='left'
        )
        
        # Add missing flags and confidence scores before imputation
        for col in self.receptor_columns:
            # Add missing flag
            df[f'{col}_missing'] = df[col].isna().astype(int)
            
            # Add confidence score (1.0 for original values, 0.5 for imputed)
            df[f'{col}_confidence'] = np.where(
                df[col].isna(),
                0.5,  # Lower confidence for values that will be imputed
                1.0   # Full confidence for original values
            )
        
        # Handle missing receptor values using class medians
        try:
            # Calculate class medians once
            class_medians = df.groupby('medication_class').agg({
                col: 'median' for col in self.receptor_columns
            })
            
            # Apply imputation using class medians
            for col in self.receptor_columns:
                # Create mask for missing values
                missing_mask = df[col].isna()
                if missing_mask.sum() == 0:
                    continue
                    
                # Get class-specific medians for missing values
                class_meds = df.loc[missing_mask, 'medication_class'].map(
                    class_medians[col]
                )
                
                # Count and warn about global fallbacks
                global_fallbacks = class_meds.isna().sum()
                if global_fallbacks > 0:
                    missing_classes = df.loc[missing_mask & class_meds.isna(), 'medication_class'].unique()
                    logger.warning(f"Column {col}: {global_fallbacks} values ({global_fallbacks/missing_mask.sum():.1%}) "
                                 f"falling back to global median. Missing classes: {missing_classes}")
                
                # Fill missing values with class medians, fallback to global median
                df.loc[missing_mask, col] = class_meds.fillna(df[col].median())
                
                # Check for extreme outliers in imputed values
                imputed_mask = df[f'{col}_missing'] == 1
                if imputed_mask.sum() > 0:
                    imputed_values = df.loc[imputed_mask, col]
                    valid_values = df.loc[~df[col].isna() & ~imputed_mask, col]
                    
                    if len(valid_values) > 10:  # Only if we have enough data
                        p01, p99 = np.percentile(valid_values, [1, 99])
                        
                        # Check for extreme outliers
                        extreme_low = imputed_values < p01
                        extreme_high = imputed_values > p99
                        
                        if extreme_low.any() or extreme_high.any():
                            logger.warning(f"Column {col}: {extreme_low.sum()} imputed values below 1st percentile, "
                                         f"{extreme_high.sum()} above 99th percentile")
                            
                            # Cap extreme values
                            df.loc[imputed_mask & (df[col] < p01), col] = p01
                            df.loc[imputed_mask & (df[col] > p99), col] = p99
                            logger.info(f"Capped extreme imputed values for {col}")
                
        except Exception as e:
            logger.warning(f"Error during receptor imputation: {e}")
            # Fallback to simple median imputation
            for col in self.receptor_columns:
                df[col] = df[col].fillna(df[col].median())
        
        # Log distribution summaries for each receptor
        logger.info("Receptor data distribution summary:")
        for col in self.receptor_columns:
            # Get distribution of original values
            orig_values = df.loc[df[f'{col}_missing'] == 0, col]
            logger.info(f"\n{col} ORIGINAL (n={len(orig_values)}):\n{orig_values.describe()}")
            
            # Get distribution of imputed values
            imp_values = df.loc[df[f'{col}_missing'] == 1, col]
            if len(imp_values) > 0:
                logger.info(f"\n{col} IMPUTED (n={len(imp_values)}):\n{imp_values.describe()}")
                
                # Calculate distribution difference if enough data
                if len(orig_values) > 10 and len(imp_values) > 10:
                    from scipy.stats import entropy
                    from scipy.spatial.distance import jensenshannon
                    
                    # Create histograms with same bins
                    min_val = min(orig_values.min(), imp_values.min())
                    max_val = max(orig_values.max(), imp_values.max())
                    bins = np.linspace(min_val, max_val, 20)
                    
                    orig_hist, _ = np.histogram(orig_values, bins=bins, density=True)
                    imp_hist, _ = np.histogram(imp_values, bins=bins, density=True)
                    
                    # Add small epsilon to avoid division by zero
                    orig_hist = orig_hist + 1e-10
                    imp_hist = imp_hist + 1e-10
                    
                    # Normalize
                    orig_hist = orig_hist / orig_hist.sum()
                    imp_hist = imp_hist / imp_hist.sum()
                    
                    # Calculate JS distance
                    js_dist = jensenshannon(orig_hist, imp_hist)
                    logger.info(f"{col} distribution difference (Jensen-Shannon distance): {js_dist:.4f}")
        
        # Log transform receptor affinities more safely
        for col in self.receptor_columns:
            if col in df.columns:
                # Handle zero or negative values before log transform
                zero_or_negative = df[col] <= 0
                if zero_or_negative.any():
                    logger.warning(f"Column {col}: {zero_or_negative.sum()} zero or negative values detected before log transform")
                    
                # Apply log transform with protection against invalid values
                df[f'{col}_log'] = np.where(df[col] <= 0, 
                                          np.nan,  # Set to NaN for zero/negative values
                                          np.log10(df[col]))
                
                # Fill NaN values with minimum valid log value - 1
                min_valid_log = df[f'{col}_log'].min()
                if pd.isna(min_valid_log):
                    min_valid_log = 0
                df[f'{col}_log'] = df[f'{col}_log'].fillna(min_valid_log - 1)
                
                # Mark these values as low confidence
                df[f'{col}_log_confidence'] = np.where(zero_or_negative, 0.3, df[f'{col}_confidence'])
        
        # Create receptor interaction features
        df['d2_5ht2a_ratio'] = df['d2_affinity'] / df['5ht2a_affinity'].replace(0, np.nan)
        df['h1_alpha1_sum'] = df['h1_affinity'] + df['alpha1_affinity']
        
        # Flag when both components are imputed
        df['d2_5ht2a_ratio_fully_imputed'] = (df['d2_affinity_missing'] & df['5ht2a_affinity_missing']).astype(int)
        df['h1_alpha1_sum_fully_imputed'] = (df['h1_affinity_missing'] & df['alpha1_affinity_missing']).astype(int)
        
        # Add confidence scores for interaction features with compounded uncertainty
        df['d2_5ht2a_ratio_confidence'] = df[['d2_affinity_confidence', '5ht2a_affinity_confidence']].min(axis=1)
        df['h1_alpha1_sum_confidence'] = df[['h1_affinity_confidence', 'alpha1_affinity_confidence']].min(axis=1)
        
        # Adjust confidence for fully imputed interactions
        df['d2_5ht2a_ratio_confidence'] = np.where(
            df['d2_5ht2a_ratio_fully_imputed'] == 1,
            df['d2_5ht2a_ratio_confidence'] * 0.5,  # Further reduce confidence when both are imputed
            df['d2_5ht2a_ratio_confidence']
        )
        df['h1_alpha1_sum_confidence'] = np.where(
            df['h1_alpha1_sum_fully_imputed'] == 1,
            df['h1_alpha1_sum_confidence'] * 0.5,  # Further reduce confidence when both are imputed
            df['h1_alpha1_sum_confidence']
        )
        
        # Track posts with multiple imputed receptors
        imputation_counts = {
            '0_imputed': 0,
            '1_imputed': 0,
            '2+_imputed': 0,
            'all_imputed': 0
        }
        
        # Calculate number of imputed values per post
        missing_cols = [f'{col}_missing' for col in self.receptor_columns]
        df['num_imputed_receptors'] = df[missing_cols].sum(axis=1)
        
        # Count posts in each category
        imputation_counts['0_imputed'] = (df['num_imputed_receptors'] == 0).sum()
        imputation_counts['1_imputed'] = (df['num_imputed_receptors'] == 1).sum()
        imputation_counts['2+_imputed'] = ((df['num_imputed_receptors'] >= 2) & 
                                        (df['num_imputed_receptors'] < len(self.receptor_columns))).sum()
        imputation_counts['all_imputed'] = (df['num_imputed_receptors'] == len(self.receptor_columns)).sum()
        
        # Calculate percentages
        total_posts = len(df)
        imputation_percentages = {k: f"{v} ({v/total_posts:.1%})" for k, v in imputation_counts.items()}
        
        logger.info(f"Receptor imputation summary (n={total_posts}):")
        for k, v in imputation_percentages.items():
            logger.info(f"  {k}: {v}")
        
        # Add special warning if many posts have multiple imputations
        if imputation_counts['2+_imputed'] / total_posts > 0.3:  # If >30% have 2+ imputed
            logger.warning(f"HIGH IMPUTATION RATE: {imputation_counts['2+_imputed']/total_posts:.1%} of posts have 2+ imputed receptor values")
        
        # Drop raw receptor columns to avoid collinearity
        # Keep the missing flags and confidence scores
        cols_to_drop = [col for col in self.receptor_columns if col in df.columns]
        df.drop(columns=cols_to_drop, inplace=True)
        
        return df
    
    def build_dosage_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Build dosage-related features.
        
        Args:
            df: DataFrame with post text and medication
            
        Returns:
            DataFrame with dosage features added
        """
        # Extract dosage from text
        df['extracted_dosage'] = df.apply(
            lambda row: self.extract_dosage(row['post_text'], row['medication'])
            if 'post_text' in row and 'medication' in row else None,
            axis=1
        )
        
        # Use extracted dosage if original not available
        df['final_dosage'] = df['extracted_dosage'].fillna(df.get('dosage', np.nan))
        
        # Create has_dosage indicator
        df['has_dosage'] = df['final_dosage'].notna().astype(int)
        
        # Normalize dosage by typical dose, using -1 as indicator for missing values
        df['dosage_ratio'] = df.apply(
            lambda row: row['final_dosage'] / row['typical_dose'] 
            if pd.notna(row['final_dosage']) and pd.notna(row['typical_dose']) and row['typical_dose'] > 0
            else -1.0,  # Use -1 as indicator value
            axis=1
        )
        
        # Create dosage categories
        def categorize_dosage(ratio):
            if ratio == -1.0:  # Missing value indicator
                return 'unknown'
            elif ratio < 0.5:
                return 'low'
            elif ratio < 1.0:
                return 'below_typical'
            elif ratio < 1.5:
                return 'typical'
            elif ratio < 2.0:
                return 'above_typical'
            else:
                return 'high'
        
        df['dosage_category'] = df['dosage_ratio'].apply(categorize_dosage)
        
        # Log dosage category distribution
        category_counts = df['dosage_category'].value_counts()
        category_percentages = df['dosage_category'].value_counts(normalize=True) * 100
        logger.info("Dosage category distribution:")
        for category, count in category_counts.items():
            logger.info(f"  {category}: {count} ({category_percentages[category]:.1f}%)")
        
        # One-hot encode categories
        dosage_dummies = pd.get_dummies(df['dosage_category'], prefix='dose')
        df = pd.concat([df, dosage_dummies], axis=1)
        
        return df
    
    def build_lexical_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Build lexical features from text.
        
        Args:
            df: DataFrame with post text
            
        Returns:
            DataFrame with lexical features added
        """
        # Initialize result DataFrame
        result_df = df.copy()
        
        # Define keyword dictionaries with polarity weights
        symptom_keywords = {
            'activation': {
                # Activation (positive)
                'energetic': 0.9, 'alert': 0.8, 'stimulated': 0.85, 'wired': 0.9,
                'restless': 0.7, 'jittery': 0.75, 'amped': 0.8, 'motivated': 0.85,
                'awake': 0.8, 'insomnia': 0.7, 'can\'t sleep': 0.7,
                'energy': 0.85, 'drive': 0.8, 'focus': 0.75, 'concentration': 0.75,
                'productive': 0.8, 'active': 0.85,
                
                # Sedation (negative)
                'sedated': -0.9, 'drowsy': -0.8, 'tired': -0.7, 'sleepy': -0.8,
                'lethargic': -0.85, 'zonked': -0.9, 'sluggish': -0.75,
                'fatigued': -0.8, 'calm': -0.6, 'relaxed': -0.5
            },
            'emotional': {
                # Blunting (negative)
                'numb': -0.9, 'flat': -0.85, 'blunted': -0.9, 'empty': -0.8,
                'no emotions': -0.9, 'can\'t feel': -0.85, 'dull': -0.75,
                'apathetic': -0.8, 'indifferent': -0.7,
                
                # Restoration (positive)
                'feel again': 0.9, 'emotions back': 0.85, 'more emotional': 0.8,
                'crying': 0.7, 'happy': 0.8, 'sad': 0.7, 'joy': 0.85,
                'excited': 0.8, 'emotional range': 0.75, 'vibrant': 0.85,
                'alive': 0.8, 'responsive': 0.75, 'mood': 0.7, 'emotion': 0.7,
                'anxious': 0.6, 'depressed': 0.6, 'irritable': 0.65, 'angry': 0.65,
                'moody': 0.7
            },
            'metabolic': {
                # Appetite increase (positive)
                'hungry': 0.8, 'craving': 0.85, 'eating more': 0.75,
                'weight gain': 0.7, 'increased appetite': 0.8,
                'always hungry': 0.85, 'snacking': 0.7, 'appetite': 0.7,
                'hunger': 0.8, 'eating': 0.7, 'food': 0.6,
                
                # Appetite decrease (negative)
                'no appetite': -0.9, 'not hungry': -0.8, 'eating less': -0.75,
                'weight loss': -0.7, 'decreased appetite': -0.8,
                'forget to eat': -0.75, 'nausea': -0.6
            }
        }
        
        # Define intensity modifiers
        intensifiers = {
            'very': 1.5, 'extremely': 2.0, 'really': 1.3, 'quite': 1.2,
            'incredibly': 1.8, 'absolutely': 1.7, 'totally': 1.4,
            'completely': 1.6, 'highly': 1.5, 'intensely': 1.7
        }
        
        diminishers = {
            'slightly': 0.7, 'somewhat': 0.8, 'a bit': 0.75, 'a little': 0.8,
            'moderately': 0.9, 'relatively': 0.85, 'fairly': 0.9,
            'kind of': 0.8, 'sort of': 0.8, 'rather': 0.85
        }
        
        # Calculate word count for normalization
        result_df['word_count'] = result_df['post_text'].apply(lambda x: len(str(x).split()))
        
        # Add keyword-based features
        for category, keywords in symptom_keywords.items():
            # Count raw keyword occurrences with phrase matching
            def count_keywords(text):
                text = str(text).lower()
                count = 0
                hits = []  # Track matched keywords for debugging
                
                if self.has_spacy:
                    doc = self.nlp(text)
                    # Get matches from spaCy
                    matches = self.matchers[category](doc)
                    # Convert matches to set of spans for deduplication
                    matched_spans = set()
                    for _, start, end in matches:
                        span = doc[start:end].text.lower()
                        if span in keywords:
                            matched_spans.add((start, end))
                            count += 1
                            hits.append(span)
                    
                    # Check remaining single words
                    for token in doc:
                        if token.text.lower() in keywords and not any(start <= token.i < end for start, end in matched_spans):
                            count += 1
                            hits.append(token.text.lower())
                else:
                    # Fallback to regex-based matching
                    # Sort keywords by length (descending) to match phrases first
                    sorted_keywords = sorted(keywords.items(), key=lambda x: len(x[0].split()), reverse=True)
                    for kw, _ in sorted_keywords:
                        if " " in kw:  # It's a phrase
                            matches = re.finditer(r'\b' + re.escape(kw) + r'\b', text)
                        else:  # Single word
                            matches = re.finditer(r'\b' + re.escape(kw) + r'\b', text)
                        for match in matches:
                            count += 1
                            hits.append(kw)
                
                if self.debug_mode:
                    return count, hits
                return count
            
            if self.debug_mode:
                # Store both count and hits for debugging
                result_df[f'{category}_keyword_count'], result_df[f'{category}_hits'] = zip(*result_df['post_text'].apply(count_keywords))
            else:
                result_df[f'{category}_keyword_count'] = result_df['post_text'].apply(count_keywords)
            
            # Calculate weighted keyword scores with phrase matching
            def calculate_weighted_score(text):
                text = str(text).lower()
                score = 0.0
                count = 0
                
                if self.has_spacy:
                    doc = self.nlp(text)
                    # Get matches from spaCy
                    matches = self.matchers[category](doc)
                    # Convert matches to set of spans for deduplication
                    matched_spans = set()
                    for _, start, end in matches:
                        span = doc[start:end].text.lower()
                        if span in keywords:
                            matched_spans.add((start, end))
                            weight = keywords[span]
                            
                            # Check for intensifiers/diminishers
                            modifier = 1.0
                            # Look at tokens before the match
                            for token in doc[max(0, start-3):start]:
                                if token.text.lower() in intensifiers:
                                    modifier *= intensifiers[token.text.lower()]
                                    break
                                elif token.text.lower() in diminishers:
                                    modifier *= diminishers[token.text.lower()]
                                    break
                            
                            # Check for negation
                            for token in doc[max(0, start-3):start]:
                                if token.text.lower() in ['not', 'no', 'never', 'didn\'t', 'doesn\'t', 
                                                        'isn\'t', 'wasn\'t', 'weren\'t', 'haven\'t', 'hasn\'t']:
                                    modifier *= -1
                                    break
                            
                            score += weight * modifier
                            count += 1
                    
                    # Check remaining single words
                    for token in doc:
                        if token.text.lower() in keywords and not any(start <= token.i < end for start, end in matched_spans):
                            weight = keywords[token.text.lower()]
                            
                            # Check for intensifiers/diminishers
                            modifier = 1.0
                            for prev_token in doc[max(0, token.i-3):token.i]:
                                if prev_token.text.lower() in intensifiers:
                                    modifier *= intensifiers[prev_token.text.lower()]
                                    break
                                elif prev_token.text.lower() in diminishers:
                                    modifier *= diminishers[prev_token.text.lower()]
                                    break
                            
                            # Check for negation
                            for prev_token in doc[max(0, token.i-3):token.i]:
                                if prev_token.text.lower() in ['not', 'no', 'never', 'didn\'t', 'doesn\'t', 
                                                             'isn\'t', 'wasn\'t', 'weren\'t', 'haven\'t', 'hasn\'t']:
                                    modifier *= -1
                                    break
                            
                            score += weight * modifier
                            count += 1
                else:
                    # Fallback to regex-based matching
                    sorted_keywords = sorted(keywords.items(), key=lambda x: len(x[0].split()), reverse=True)
                    for kw, weight in sorted_keywords:
                        if " " in kw:  # It's a phrase
                            matches = re.finditer(r'\b' + re.escape(kw) + r'\b', text)
                        else:  # Single word
                            matches = re.finditer(r'\b' + re.escape(kw) + r'\b', text)
                        
                        for match in matches:
                            # Check for intensifiers/diminishers
                            modifier = 1.0
                            start = max(0, match.start() - 20)  # Look back 20 chars
                            context = text[start:match.start()]
                            
                            # Check for intensifiers
                            for intensifier, mult in intensifiers.items():
                                if re.search(r'\b' + re.escape(intensifier) + r'\b', context):
                                    modifier *= mult
                                    break
                                    
                            # Check for diminishers
                            for diminisher, mult in diminishers.items():
                                if re.search(r'\b' + re.escape(diminisher) + r'\b', context):
                                    modifier *= mult
                                    break
                            
                            # Check for negation
                            if re.search(r'\b(not|no|never|didn\'t|doesn\'t|isn\'t|wasn\'t|weren\'t|haven\'t|hasn\'t)\s+', context):
                                modifier *= -1
                            
                            score += weight * modifier
                            count += 1
                
                # Handle zero counts explicitly
                if count == 0:
                    return 0.0
                return score / count
            
            # Add weighted score features
            result_df[f'{category}_weighted_score'] = result_df['post_text'].apply(calculate_weighted_score)
            
            # Add log-transformed weighted score
            result_df[f'{category}_log_weighted'] = np.log1p(result_df[f'{category}_weighted_score'].abs()) * np.sign(result_df[f'{category}_weighted_score'])
            
            # Add normalized keyword counts (per 100 words)
            result_df[f'{category}_norm_count'] = result_df[f'{category}_keyword_count'] * 100 / result_df['word_count']
            
            # Add positive/negative keyword counts
            result_df[f'{category}_positive_count'] = result_df['post_text'].apply(
                lambda x: sum(1 for kw, weight in keywords.items() 
                            if weight > 0 and re.search(r'\b' + re.escape(kw) + r'\b', str(x).lower()))
            )
            
            result_df[f'{category}_negative_count'] = result_df['post_text'].apply(
                lambda x: sum(1 for kw, weight in keywords.items() 
                            if weight < 0 and re.search(r'\b' + re.escape(kw) + r'\b', str(x).lower()))
            )
            
            # Add intensity features
            result_df[f'{category}_intensity_count'] = result_df['post_text'].apply(
                lambda x: sum(1 for intensifier in intensifiers 
                            if re.search(r'\b' + re.escape(intensifier) + r'\b', str(x).lower()))
            )
            
            result_df[f'{category}_diminisher_count'] = result_df['post_text'].apply(
                lambda x: sum(1 for diminisher in diminishers 
                            if re.search(r'\b' + re.escape(diminisher) + r'\b', str(x).lower()))
            )
        
        # Add TF-IDF features if enabled
        if self.use_tfidf:
            if self.load_existing_vectorizer:
                try:
                    # Load existing vectorizer
                    self.vectorizer = joblib.load(self.vectorizer_path)
                    logger.info(f"Loaded existing TF-IDF vectorizer from {self.vectorizer_path}")
                    
                    # Transform using existing vectorizer
                    tfidf_matrix = self.vectorizer.transform(result_df['post_text'].fillna(''))
                    feature_names = self.vectorizer.get_feature_names_out()
                except Exception as e:
                    error_msg = f"Failed to load existing vectorizer from {self.vectorizer_path}: {e}"
                    if self.strict_vectorizer:
                        logger.error(error_msg)
                        raise ValueError(error_msg)
                    else:
                        logger.warning(f"{error_msg}. Falling back to fitting new one.")
                        self.load_existing_vectorizer = False  # Fall back to fitting
                    
            if not self.load_existing_vectorizer:
                # Create and fit new vectorizer
                self.vectorizer = TfidfVectorizer(
                    max_features=50,
                    ngram_range=(1, 2),
                    stop_words='english'
                )
                
                # Fit and transform text
                tfidf_matrix = self.vectorizer.fit_transform(result_df['post_text'].fillna(''))
                feature_names = self.vectorizer.get_feature_names_out()
                
                # Save vectorizer for future use
                os.makedirs('models/vectorizers', exist_ok=True)
                joblib.dump(self.vectorizer, self.vectorizer_path)
            
            # Log selected features
            logger.info(f"TF-IDF using {len(feature_names)} features:")
            for i, term in enumerate(feature_names):
                logger.info(f"  {i+1}. {term}")
            
            # Add TF-IDF features to DataFrame
            for i, term in enumerate(feature_names):
                result_df[f'tfidf_{term}'] = tfidf_matrix[:, i].toarray().flatten()
            
            # Save feature names for reference
            with open('models/vectorizers/tfidf_features.json', 'w') as f:
                json.dump(feature_names.tolist(), f)
        
        # Add debugging information if enabled
        if self.debug_mode:
            os.makedirs(self.debug_dir, exist_ok=True)
            
            # For each category, save top 20 posts by weighted score
            for category in symptom_keywords:
                # Sort by absolute weighted score (to get both extremes)
                top_posts = result_df.loc[result_df[f'{category}_keyword_count'] > 0]
                top_posts = top_posts.sort_values(f'{category}_weighted_score', 
                                                key=lambda x: x.abs(), 
                                                ascending=False).head(20)
                
                # Create preview with post text, score, and hits
                debug_cols = ['post_text', f'{category}_weighted_score', 
                             f'{category}_hits', f'{category}_norm_count']
                
                # Add identifier columns if available
                for id_col in ['post_id', 'id', 'user_id']:
                    if id_col in top_posts.columns:
                        debug_cols.insert(0, id_col)
                        
                # Add medication column if available
                if 'medication' in top_posts.columns:
                    debug_cols.insert(1, 'medication')
                    
                preview = top_posts[debug_cols]
                preview.to_csv(f"{self.debug_dir}/{category}_top_posts.csv", index=False)
        
        return result_df

    def build_all_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Build all features for the model.
        
        Args:
            df: Input DataFrame
            
        Returns:
            DataFrame with all features built
            
        Raises:
            ValueError: If required columns are missing or feature building is incomplete
        """
        # Validate inputs
        required_cols = ['medication', 'post_text']
        missing = [col for col in required_cols if col not in df.columns]
        if missing:
            raise ValueError(f"Missing required columns: {missing}")
        
        logger.info("Building receptor features...")
        df = self.build_receptor_features(df)
        
        logger.info("Building dosage features...")
        df = self.build_dosage_features(df)
        
        logger.info("Building lexical features...")
        df = self.build_lexical_features(df)
        
        # Handle infinite values before scaling
        for col in ['d2_5ht2a_ratio', 'h1_alpha1_sum']:
            df[col] = df[col].replace([np.inf, -np.inf], np.nan)
        
        # Then fill NaNs with column-specific medians
        if df[['d2_5ht2a_ratio', 'h1_alpha1_sum']].isna().any().any():
            for col in ['d2_5ht2a_ratio', 'h1_alpha1_sum']:
                df[col] = df[col].fillna(df[col].median())
        
        # Define explicitly which features should be scaled
        scale_these = [
            'd2_5ht2a_ratio',    # Receptor interaction
            'h1_alpha1_sum',     # Receptor interaction
        ]
        
        # Add lexical features to scaling list
        lexical_cols = [col for col in df.columns if col.startswith(('activation_', 'emotional_', 'metabolic_'))]
        scale_these.extend(lexical_cols)
        
        # Add TF-IDF features to scaling list only if configured
        if self.use_tfidf and self.scale_tfidf:
            tfidf_cols = [col for col in df.columns if col.startswith('tfidf_')]
            scale_these.extend(tfidf_cols)
            logger.info(f"Including {len(tfidf_cols)} TF-IDF features in scaling")
        else:
            logger.info("TF-IDF features will remain unscaled")
        
        # Optionally include embedding features if they exist
        embedding_cols = [col for col in df.columns if col.startswith('embedding_')]
        scale_these.extend(embedding_cols)
        
        # Only scale features that are present in the DataFrame
        features_to_scale = [col for col in scale_these if col in df.columns]
        logger.info(f"Scaling features: {features_to_scale}")
        
        # Scale the selected features
        df[features_to_scale] = self.scaler.fit_transform(df[features_to_scale])
        
        # Handle dosage_ratio scaling separately to preserve indicator values
        if 'dosage_ratio' in df.columns:
            mask = df['dosage_ratio'] != -1
            if mask.any():
                # Convert to 2D array for scaling
                values_to_scale = df.loc[mask, 'dosage_ratio'].values.reshape(-1, 1)
                # Fit and transform
                scaled_values = self.scaler.fit_transform(values_to_scale)
                # Update only the non-indicator values
                df.loc[mask, 'dosage_ratio'] = scaled_values.flatten()
        
        # Validate scaling worked correctly
        for feature in features_to_scale:
            mean_val = df[feature].mean()
            std_val = df[feature].std()
            if abs(mean_val) > 0.1 or abs(std_val - 1.0) > 0.1:
                logger.warning(f"Scaling issue with {feature}: mean={mean_val:.3f}, std={std_val:.3f}")
        
        # Save scaler for reproducing transformations in inference
        if self.config.get('save_scaler', True):
            scaler_dir = self.config.get('scaler_dir', 'models/scalers')
            os.makedirs(scaler_dir, exist_ok=True)
            scaler_path = os.path.join(scaler_dir, 'feature_scaler.joblib')
            joblib.dump(self.scaler, scaler_path)
            logger.info(f"Saved feature scaler to {scaler_path}")
        
        # Validate core features exist
        expected_features = ['d2_5ht2a_ratio', 'h1_alpha1_sum', 'dosage_ratio']
        missing_features = [f for f in expected_features if f not in df.columns]
        if missing_features:
            logger.error(f"Missing expected features: {missing_features}")
            raise ValueError(f"Feature building incomplete: {missing_features}")
        
        # Group features by type for easier analysis
        feature_groups = {
            'receptor_features': [col for col in df.columns if any(x in col for x in 
                                ['_affinity', 'd2_', '5ht2a_', 'h1_', 'alpha1_', 'm1_'])],
            'lexical_features': [col for col in df.columns if any(x in col for x in 
                               ['_keyword_', 'tfidf_', '_weighted_', '_count', '_intensity_'])],
            'temporal_features': [col for col in df.columns if any(x in col for x in 
                                ['_temporal_', 'current_', 'past_', 'starting_', 'stopped_'])],
            'dosage_features': [col for col in df.columns if any(x in col for x in 
                              ['dosage_', 'dose_', '_mg_'])],
            'other_features': []
        }
        
        # Assign any remaining features to 'other_features'
        all_grouped = set()
        for group in feature_groups.values():
            all_grouped.update(group)
        feature_groups['other_features'] = [col for col in df.columns if col not in all_grouped]
        
        # Save feature groups as metadata
        os.makedirs('models/metadata', exist_ok=True)
        with open('models/metadata/feature_groups.json', 'w') as f:
            json.dump(feature_groups, f, indent=2)
        
        logger.info("Feature groups created and saved to models/metadata/feature_groups.json")
        
        logger.info(f"Feature building complete: {len(df.columns)} features created")
        return df

    def plot_shap_summary(self, model, X, feature_names=None, plot_type=None, group_features=False):
        """Plot SHAP summary with proper feature names.
        
        Args:
            model: Trained model to explain
            X: Feature matrix
            feature_names: Optional list of feature names
            plot_type: Optional plot type ('dot', 'bar', 'both', or 'grouped')
            group_features: Whether to group features by category (default: False)
        """
        try:
            import shap
            import matplotlib.pyplot as plt
        except ImportError:
            logger.error("SHAP or matplotlib not installed. Please install with: pip install shap matplotlib")
            return
        
        # Handle DMatrix input
        if hasattr(X, 'feature_names'):
            # XGBoost DMatrix case
            feature_data = X.get_data()
            if feature_names is None:
                feature_names = X.feature_names
        else:
            # Regular DataFrame/array case
            feature_data = X
        
        # Load TF-IDF feature names if they exist
        try:
            with open('models/vectorizers/tfidf_features.json', 'r') as f:
                tfidf_features = json.load(f)
                
            # Replace generic feature names with actual features
            if feature_names:
                for i, name in enumerate(feature_names):
                    if isinstance(name, str) and name.startswith('tfidf_'):
                        idx = int(name.replace('tfidf_', ''))
                        if idx < len(tfidf_features):
                            feature_names[i] = f"tfidf_{tfidf_features[idx]}"
        except:
            logger.warning("TF-IDF feature names not found. Using default feature names.")
        
        # Create SHAP plot with updated feature names
        explainer = shap.TreeExplainer(model)
        shap_values = explainer.shap_values(feature_data)
        
        # Load feature groups if available
        feature_groups = None
        if group_features:
            try:
                with open('models/metadata/feature_groups.json', 'r') as f:
                    feature_groups = json.load(f)
            except:
                logger.warning("Feature groups not found. Using ungrouped visualization.")
                group_features = False
        
        # Plot based on type
        if plot_type == 'grouped' or (group_features and plot_type is None):
            self._plot_grouped_shap(shap_values, feature_names, feature_groups)
        elif not plot_type or plot_type == 'dot':
            shap.summary_plot(shap_values, feature_data, feature_names=feature_names)
        elif plot_type == 'bar':
            shap.summary_plot(shap_values, feature_data, feature_names=feature_names, plot_type='bar')
        elif plot_type == 'both':
            plt.figure(figsize=(10, 8))
            shap.summary_plot(shap_values, feature_data, feature_names=feature_names)
            plt.figure(figsize=(10, 8))
            shap.summary_plot(shap_values, feature_data, feature_names=feature_names, plot_type='bar')
    
    def _plot_grouped_shap(self, shap_values, feature_names, feature_groups):
        """Plot SHAP values grouped by feature category.
        
        Args:
            shap_values: SHAP values from explainer
            feature_names: List of feature names
            feature_groups: Dictionary mapping feature names to groups
        """
        import matplotlib.pyplot as plt
        import numpy as np
        import pandas as pd
        
        if feature_groups is None:
            logger.warning("No feature groups provided. Using ungrouped visualization.")
            return
        
        # Create mapping from feature name to group
        feature_to_group = {}
        for group, features in feature_groups.items():
            for feature in features:
                feature_to_group[feature] = group
        
        # Sum absolute SHAP values by group
        group_importance = {}
        for i, feature in enumerate(feature_names):
            group = feature_to_group.get(feature, 'other')
            if group not in group_importance:
                group_importance[group] = 0
            group_importance[group] += np.abs(shap_values[:,i]).mean()
        
        # Sort groups by importance (ascending for reversed plot)
        sorted_groups = sorted(group_importance.items(), key=lambda x: x[1])
        
        # Apply top-K filtering if configured
        top_k = self.config.get("shap_group_top_k", None)
        if top_k and len(sorted_groups) > 5:
            sorted_groups = sorted_groups[-top_k:]
            logger.info(f"Showing top {top_k} feature groups by importance")
        
        groups = [g[0] for g in sorted_groups]
        values = [g[1] for g in sorted_groups]
        
        # Calculate normalized values (percentage contribution)
        total_importance = sum(values)
        normalized_values = [v/total_importance * 100 for v in values]
        
        # Export to CSV for research logs
        os.makedirs('plots', exist_ok=True)
        df = pd.DataFrame({
            "group": groups,
            "mean_abs_shap": values,
            "percent": normalized_values
        })
        df.to_csv("plots/feature_group_importance.csv", index=False)
        logger.info("Saved feature group importance to plots/feature_group_importance.csv")
        
        # Define consistent color scheme
        colors = {
            'receptor_features': '#2ecc71',  # Green
            'lexical_features': '#3498db',   # Blue
            'temporal_features': '#e74c3c',  # Red
            'dosage_features': '#f1c40f',    # Yellow
            'other_features': '#95a5a6'      # Gray
        }
        
        # Create plot
        plt.figure(figsize=(12, 6))
        
        # Create bars with consistent colors
        bars = plt.barh(groups, values, 
                       color=[colors.get(g, colors['other_features']) for g in groups])
        
        # Add grid lines for easier comparison
        plt.grid(True, axis='x', linestyle='--', alpha=0.3)
        
        # Add value labels (both raw and percentage)
        for bar, value, norm_value in zip(bars, values, normalized_values):
            width = bar.get_width()
            plt.text(width, bar.get_y() + bar.get_height()/2, 
                    f'{width:.3f} ({norm_value:.1f}%)', 
                    ha='left', va='center', fontsize=9)
        
        plt.title('Feature Group Importance (Mean |SHAP value|)')
        plt.xlabel('Mean |SHAP value|')
        plt.tight_layout()
        
        # Add group descriptions if available
        group_descriptions = {
            'receptor_features': 'Drug receptor binding affinities and interactions',
            'lexical_features': 'Text-based features from post content',
            'temporal_features': 'Time-related medication effects',
            'dosage_features': 'Medication dosage information',
            'other_features': 'Miscellaneous features'
        }
        
        # Add description text
        desc_text = "\n".join([f"{group}: {group_descriptions.get(group, '')}" 
                              for group in groups])
        plt.figtext(0.02, 0.02, desc_text, fontsize=8, 
                   bbox=dict(facecolor='white', alpha=0.8))
        
        # Save plot
        plt.savefig('plots/feature_group_importance.png', dpi=300, bbox_inches='tight')
        plt.close()
        
        # Log group importance (both raw and normalized)
        logger.info("Feature group importance:")
        for group, value, norm_value in zip(groups, values, normalized_values):
            logger.info(f"  {group}: {value:.3f} ({norm_value:.1f}%)")

def load_annotated_responses(filepath: str) -> pd.DataFrame:
    """Load annotated responses DataFrame."""
    return pd.read_parquet(filepath)

def load_receptor_data(filepath: str) -> pd.DataFrame:
    """Load drug receptor affinity data."""
    df = pd.read_csv(filepath)
    
    # Validate required columns
    required_cols = [
        'medication', 'medication_class', 'd2_affinity', 
        '5ht2a_affinity', 'h1_affinity', 'alpha1_affinity', 
        'm1_affinity', 'half_life', 'typical_dose'
    ]
    
    missing_cols = [col for col in required_cols if col not in df.columns]
    if missing_cols:
        raise ValueError(f"Missing required columns in receptor data: {missing_cols}")
    
    # Ensure medication names are lowercase for matching
    df['medication'] = df['medication'].str.lower()
    
    return df

def save_features(df: pd.DataFrame, filepath: str, scaler: StandardScaler) -> None:
    """Save features to parquet file with comprehensive schema documentation.
    
    Args:
        df: DataFrame containing features
        filepath: Path to save parquet file
        scaler: Fitted StandardScaler instance
    """
    # Create output directory if it doesn't exist
    os.makedirs(os.path.dirname(filepath), exist_ok=True)
    
    # Save the parquet file
    df.to_parquet(filepath)
    
    # Load the human-readable schema template
    schema_template_path = "configs/feature_schema.json"
    schema_template = {}
    
    if os.path.exists(schema_template_path):
        try:
            with open(schema_template_path, 'r') as f:
                schema_template = json.load(f)
        except Exception as e:
            logger.warning(f"Failed to load schema template: {e}")
            logger.info("Proceeding without template...")
    else:
        logger.warning(f"Schema template not found at {schema_template_path}")
    
    # Generate actual feature list with types
    actual_schema = {}
    for col in df.columns:
        # Find the feature in schema template
        feature_found = False
        for category, features in schema_template.get('features', {}).items():
            if col in features:
                actual_schema[col] = {
                    "description": features[col],
                    "category": category,
                    "dtype": str(df[col].dtype),
                    "is_scaled": col in schema_template.get('scaling_info', {}).get('scaled_features', []),
                    "stats": {
                        "n_unique": int(df[col].nunique()),
                        "n_missing": int(df[col].isna().sum()),
                        "mean": None,
                        "std": None
                    }
                }
                feature_found = True
                break
        
        if not feature_found:
            # Feature not found in template
            actual_schema[col] = {
                "description": "Unknown feature",
                "category": "unknown",
                "dtype": str(df[col].dtype),
                "is_scaled": False,
                "stats": {
                    "n_unique": int(df[col].nunique()),
                    "n_missing": int(df[col].isna().sum()),
                    "mean": None,
                    "std": None
                }
            }
        
        # Add numeric stats separately with error handling
        if df[col].dtype in ['float64', 'int64']:
            try:
                mean_val = df[col].mean()
                std_val = df[col].std()
                actual_schema[col]["stats"]["mean"] = float(mean_val) if pd.notna(mean_val) else None
                actual_schema[col]["stats"]["std"] = float(std_val) if pd.notna(std_val) else None
            except Exception as e:
                logger.warning(f"Failed to calculate stats for {col}: {e}")
    
    # Save both the template and actual schema
    schema_path = filepath.replace('.parquet', '_schema.json')
    with open(schema_path, 'w') as f:
        json.dump(actual_schema, f, indent=2)
    
    # Save scaler as pickle (for easy loading)
    scaler_pickle_path = filepath.replace('.parquet', '_scaler.pkl')
    joblib.dump(scaler, scaler_pickle_path)
    
    # Also save scaler parameters as JSON (for inspection)
    scaler_json_path = filepath.replace('.parquet', '_scaler.json')
    scaler_params = {
        "mean_": scaler.mean_.tolist() if hasattr(scaler, 'mean_') else None,
        "scale_": scaler.scale_.tolist() if hasattr(scaler, 'scale_') else None,
        "var_": scaler.var_.tolist() if hasattr(scaler, 'var_') else None,
        "n_features_in_": scaler.n_features_in_ if hasattr(scaler, 'n_features_in_') else None,
        "feature_names_in_": scaler.feature_names_in_.tolist() if hasattr(scaler, 'feature_names_in_') else None,
        "n_samples_seen_": int(scaler.n_samples_seen_) if hasattr(scaler, 'n_samples_seen_') else None
    }
    with open(scaler_json_path, 'w') as f:
        json.dump(scaler_params, f, indent=2)
    
    # Save summary info
    info_path = filepath.replace('.parquet', '_info.json')
    info = {
        "n_samples": len(df),
        "n_features": len(df.columns),
        "feature_categories": {},
        "scaled_features": [col for col, info in actual_schema.items() if info.get('is_scaled', False)],
        "missing_from_template": [col for col, info in actual_schema.items() if info['category'] == 'unknown']
    }
    
    # Count features by category
    for col, feat_info in actual_schema.items():
        category = feat_info['category']
        if category not in info["feature_categories"]:
            info["feature_categories"][category] = 0
        info["feature_categories"][category] += 1
    
    with open(info_path, 'w') as f:
        json.dump(info, f, indent=2)
    
    logger.info(f"Saved features to {filepath}")
    logger.info(f"Saved schema documentation to {schema_path}")
    logger.info(f"Saved scaler to {scaler_pickle_path} and {scaler_json_path}")
    
    # Log any features missing from template
    if info["missing_from_template"]:
        logger.warning(f"Features missing from schema template: {info['missing_from_template']}")

def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(description="Build features for medication response prediction")
    parser.add_argument("--input", required=True, help="Path to annotated responses parquet")
    parser.add_argument("--drug_data", required=True, help="Path to drug receptor CSV")
    parser.add_argument("--output", required=True, help="Path for output features parquet")
    
    args = parser.parse_args()
    
    try:
        # Load data
        logger.info(f"Loading annotated responses from {args.input}")
        responses_df = load_annotated_responses(args.input)
        
        logger.info(f"Loading receptor data from {args.drug_data}")
        receptor_df = load_receptor_data(args.drug_data)
        
        # Initialize feature builder
        feature_builder = FeatureBuilder(receptor_df)
        
        # Build all features
        features_df = feature_builder.build_all_features(responses_df)
        
        # Save features
        logger.info(f"Saving features to {args.output}")
        save_features(features_df, args.output, feature_builder.scaler)
        
        print(f"\nFeature building complete:")
        print(f"- Input samples: {len(responses_df)}")
        print(f"- Output samples: {len(features_df)}")
        print(f"- Total features: {len(features_df.columns)}")
        print(f"- Output saved to: {args.output}")
        
    except Exception as e:
        logger.error(f"Error during feature building: {e}", exc_info=True)
        raise

if __name__ == "__main__":
    main() 
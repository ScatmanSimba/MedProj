#!/usr/bin/env python3
"""Script to generate fuzzy cache for symptom matching.

This script precomputes fuzzy matches for all terms defined in the YAML configuration
and saves them to a pickle file for faster loading at runtime.
"""

import argparse
import logging
import pickle
from pathlib import Path
import yaml
from rapidfuzz import process
import sys

# Add src to Python path
src_path = Path(__file__).parent.parent
sys.path.append(str(src_path))

from src.features.symptom_matcher import load_dimension_mappings, load_semantic_expansions

logger = logging.getLogger(__name__)

def generate_fuzzy_cache(config_path: Path, output_path: Path, threshold: float = 80) -> None:
    """Generate fuzzy cache for symptom matching.
    
    Args:
        config_path: Path to YAML configuration file
        output_path: Path to save pickle file
        threshold: Minimum similarity threshold (0-100)
    """
    # Load terms from YAML
    terms = set()
    
    # Add terms from dimension mappings
    for dimension, mapping in load_dimension_mappings().items():
        for term, info in mapping['terms'].items():
            terms.add(term.lower())
            # Add synonyms
            if 'synonyms' in info:
                terms.update(s.lower() for s in info['synonyms'])
    
    # Add terms from semantic expansions
    for dimension, expansions in load_semantic_expansions().items():
        for term, synonyms in expansions.items():
            terms.add(term.lower())
            terms.update(s.lower() for s in synonyms)
    
    logger.info(f"Loaded {len(terms)} terms from configuration")
    
    # Generate common misspellings
    misspellings = set()
    for term in terms:
        # Add common typos (limited to 1-2 character changes)
        for i in range(len(term)):
            # Character deletion
            misspellings.add(term[:i] + term[i+1:])
            # Character substitution (limited to common typos)
            for c in 'aeiou':  # Common vowel substitutions
                misspellings.add(term[:i] + c + term[i+1:])
    
    logger.info(f"Generated {len(misspellings)} potential misspellings")
    
    # Precompute scores
    fuzzy_cache = {}
    logger.info(f"Precomputing fuzzy matches with threshold {threshold}...")
    
    for misspelling in misspellings:
        if misspelling not in terms:  # Skip if it's already a valid term
            match = process.extractOne(misspelling, terms)
            if match and match[1] >= threshold:
                fuzzy_cache[misspelling] = match
    
    logger.info(f"Found {len(fuzzy_cache)} high-confidence fuzzy matches")
    
    # Save cache
    try:
        with open(output_path, 'wb') as f:
            pickle.dump(fuzzy_cache, f)
        logger.info(f"Saved fuzzy cache to {output_path}")
    except Exception as e:
        logger.error(f"Failed to save fuzzy cache: {e}")
        sys.exit(1)

def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(description="Generate fuzzy cache for symptom matching")
    parser.add_argument("--config", type=Path, default=src_path / "config" / "feature_config.yaml",
                      help="Path to YAML configuration file")
    parser.add_argument("--output", type=Path, default=src_path / "features" / "fuzzy_cache.pkl",
                      help="Path to save pickle file")
    parser.add_argument("--threshold", type=float, default=80,
                      help="Minimum similarity threshold (0-100)")
    parser.add_argument("--verbose", action="store_true",
                      help="Enable verbose logging")
    
    args = parser.parse_args()
    
    # Configure logging
    level = logging.DEBUG if args.verbose else logging.INFO
    logging.basicConfig(level=level,
                      format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    
    # Generate cache
    generate_fuzzy_cache(args.config, args.output, args.threshold)

if __name__ == "__main__":
    main() 
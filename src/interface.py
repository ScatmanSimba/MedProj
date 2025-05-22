"""interface.py - Streamlit interface for medication response annotation

This module provides a Streamlit interface for annotating medication responses
in Reddit posts. It includes functionality for displaying posts, collecting
annotations, and saving them to a file.
"""

import streamlit as st
import pandas as pd
from pathlib import Path
from typing import Dict, Any, Optional
import json
from datetime import datetime

from src.data.gold_set import GoldSetCreator

def initialize_session_state():
    """Initialize session state variables"""
    if 'annotator_id' not in st.session_state:
        st.session_state.annotator_id = None
    if 'current_post_idx' not in st.session_state:
        st.session_state.current_post_idx = 0
    if 'med_scores' not in st.session_state:
        st.session_state.med_scores = {}

def display_post(post: pd.Series):
    """Display a post with its medication responses"""
    st.write("### Post Text")
    st.write(post['post_text'])
    
    st.write("### Medications")
    meds = post.get('medications', [])
    st.write(", ".join(meds))
    
    st.write("## Medication Responses")
    
    # Get dimension scores and sentence mappings from post
    dimension_scores = post.get('response_dimension_scores', {})
    med_to_sentences = post.get('med_to_sentences', {})
    
    # Display sliders for each medication
    for med_name in meds:
        st.write(f"### {med_name}")
        
        # Add mentioning sentences section
        if med_name in med_to_sentences and med_to_sentences[med_name]:
            with st.expander("Mentioning Sentences"):
                for sentence in med_to_sentences[med_name]:
                    st.markdown(f"- {sentence}")
        
        # Get scores for this medication
        med_scores = dimension_scores.get(med_name, {})
        
        # Create sliders for each dimension
        activation_score = st.slider(
            f"{med_name} - Activation ↔ Sedation",
            0.0, 1.0, med_scores.get('activation', 0.5), 0.01,
            help="0 = Very Sedated, 1 = Very Activated"
        )
        
        emotional_score = st.slider(
            f"{med_name} - Emotional Blunting ↔ Restoration",
            0.0, 1.0, med_scores.get('emotional', 0.5), 0.01,
            help="0 = Emotionally Blunted, 1 = Emotions Restored"
        )
        
        metabolic_score = st.slider(
            f"{med_name} - Appetite/Metabolic Impact",
            0.0, 1.0, med_scores.get('metabolic', 0.5), 0.01,
            help="0 = Appetite Suppressed, 1 = Appetite Increased"
        )
        
        # Store the per-medication scores
        med_scores = {
            'activation': activation_score,
            'emotional': emotional_score,
            'metabolic': metabolic_score
        }
        
        # Update the dictionary of scores for this medication
        st.session_state.med_scores[med_name] = med_scores

def main():
    """Main entry point for the Streamlit interface"""
    st.title("Medication Response Annotation")
    
    # Initialize session state
    initialize_session_state()
    
    # Get annotator ID
    if not st.session_state.annotator_id:
        st.session_state.annotator_id = st.text_input("Enter your annotator ID:")
        if not st.session_state.annotator_id:
            st.stop()
    
    # Initialize gold set creator
    creator = GoldSetCreator()
    
    # Get unannotated posts
    try:
        unannotated = creator.get_unannotated_posts()
    except FileNotFoundError:
        st.error("No gold set found. Please create one first.")
        st.stop()
    
    if len(unannotated) == 0:
        st.success("All posts have been annotated!")
        st.stop()
    
    # Get current post
    current_post = unannotated.iloc[st.session_state.current_post_idx]
    
    # Display post
    display_post(current_post)
    
    # Add confidence slider
    confidence = st.slider(
        "Confidence in annotation",
        0.0, 1.0, 0.5, 0.1,
        help="How confident are you in your annotation?"
    )
    
    # Add notes field
    notes = st.text_area("Notes", help="Add any notes about the annotation")
    
    # Add submit button
    if st.button("Submit and Next"):
        # Build complete medication-specific ratings
        med_ratings = {}
        for med_name in current_post.get('medications', []):
            if med_name in st.session_state.med_scores:
                med_ratings[med_name] = st.session_state.med_scores[med_name]
        
        # Save annotation
        creator.save_annotation(current_post.name, {
            'post_id': current_post.name,
            'response_dimension_scores': med_ratings,
            'confidence': confidence,
            'notes': notes,
            'annotator_id': st.session_state.annotator_id
        })
        
        # Move to next post
        st.session_state.current_post_idx += 1
        if st.session_state.current_post_idx >= len(unannotated):
            st.success("All posts have been annotated!")
            st.stop()
        
        # Clear medication scores
        st.session_state.med_scores = {}
        
        # Rerun to show next post
        st.experimental_rerun()

if __name__ == "__main__":
    main() 
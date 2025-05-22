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
    if 'current_post_id' not in st.session_state:
        st.session_state.current_post_id = None

def display_post(post: pd.Series):
    """Display a post with its medication responses"""
    st.write("### Post Text")
    st.write(post['post_text'])
    
    st.write("### Medications")
    medications_in_post = post.get('medications', [])
    if not isinstance(medications_in_post, list):  # Ensure it's a list
        medications_in_post = []
    st.write(", ".join(medications_in_post))
    
    st.write("## Medication Responses")
    
    # Get dimension scores and sentence mappings from post
    med_to_sentences = post.get('med_to_sentences', {})
    
    # Ensure st.session_state.med_scores is initialized for this post
    if 'current_post_id' not in st.session_state or st.session_state.current_post_id != post.name:
        st.session_state.med_scores = {}  # Reset for new post
        st.session_state.current_post_id = post.name

    # Display sliders for EACH medication
    for med_name in medications_in_post:
        st.markdown(f"---")  # Separator
        st.write(f"### Rate effects for: **{med_name}**")
        
        # Display sentences where THIS medication is mentioned to guide annotation
        if med_name in med_to_sentences and med_to_sentences[med_name]:
            with st.expander(f"Sentences mentioning {med_name}"):
                for sentence in med_to_sentences[med_name]:
                    st.markdown(f"- {sentence}")
        
        # Get pre-filled scores for THIS medication from the automated system (if available)
        automated_med_specific_scores = {}
        if isinstance(post.get('response_dimension_scores'), dict):
            automated_med_specific_scores = post.get('response_dimension_scores', {}).get(med_name, {})

        # Initialize scores for this med in session_state if not present
        if med_name not in st.session_state.med_scores:
            st.session_state.med_scores[med_name] = {
                'activation': automated_med_specific_scores.get('activation', 0.5),
                'emotional': automated_med_specific_scores.get('emotional', 0.5),
                'metabolic': automated_med_specific_scores.get('metabolic', 0.5)
            }

        # Create unique keys for sliders for each medication
        activation_key = f"{med_name}_activation"
        emotional_key = f"{med_name}_emotional"
        metabolic_key = f"{med_name}_metabolic"

        st.write(f"**For {med_name}:**")
        activation_score = st.slider(
            f"Activation ↔ Sedation",  # Label is now generic, context is provided by med_name header
            0.0, 1.0, 
            st.session_state.med_scores[med_name]['activation'],  # Use session state for stickiness within post
            0.01,
            help=f"Rate Activation/Sedation specifically for {med_name}. 0=Very Sedated, 1=Very Activated",
            key=activation_key
        )
        
        emotional_score = st.slider(
            f"Emotional Blunting ↔ Restoration",
            0.0, 1.0, 
            st.session_state.med_scores[med_name]['emotional'],
            0.01,
            help=f"Rate Emotional Blunting/Restoration specifically for {med_name}. 0=Emotionally Blunted, 1=Emotions Restored",
            key=emotional_key
        )
        
        metabolic_score = st.slider(
            f"Appetite/Metabolic Impact",
            0.0, 1.0, 
            st.session_state.med_scores[med_name]['metabolic'],
            0.01,
            help=f"Rate Appetite/Metabolic Impact specifically for {med_name}. 0=Appetite Suppressed, 1=Appetite Increased",
            key=metabolic_key
        )
        
        # Update the session state for the current medication's scores as sliders are moved
        st.session_state.med_scores[med_name]['activation'] = activation_score
        st.session_state.med_scores[med_name]['emotional'] = emotional_score
        st.session_state.med_scores[med_name]['metabolic'] = metabolic_score

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
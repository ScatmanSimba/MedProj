# -*- coding: utf-8 -*-
"""predictor.py - Model prediction interface for medication response

This module provides an interface for making predictions with the trained model.
It loads the trained XGBoost models and provides methods for predicting
responses to medications based on receptor profiles.

Usage:
    python -m src.models.predictor \
        --model models/receptor_predictor.pkl \
        --affinity data/external/drug_receptor_affinity.csv \
        --medication lexapro
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
import xgboost as xgb

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler("predictor.log"),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger("predictor")

class MedicationResponsePredictor:
    """Interface for predicting medication responses."""
    
    def __init__(self, model_path: str, drug_data_path: str):
        """
        Initialize the predictor.

        Args:
            model_path: Path to trained model pickle file
            drug_data_path: Path to drug receptor affinity CSV
        """
        self.model_path = model_path
        self.drug_data_path = drug_data_path
        
        # Load model
        logger.info(f"Loading model from {model_path}")
        with open(model_path, 'rb') as f:
            self.model_results = pickle.load(f)
        
        self.models = self.model_results['models']
        self.feature_info = self.model_results['feature_info']
        
        # Load drug data
        logger.info(f"Loading drug data from {drug_data_path}")
        self.drug_data = pd.read_csv(drug_data_path)
        
        # Convert medication names to lowercase for consistent matching
        self.drug_data['medication'] = self.drug_data['medication'].str.lower()
        
        logger.info(f"Loaded model and data for {len(self.drug_data)} medications")
    
    def get_receptor_profile(self, medication: str) -> Optional[Dict[str, float]]:
        """
        Get receptor profile for a medication.

        Args:
            medication: Medication name (case insensitive)

        Returns:
            Dictionary with receptor values or None if not found
        """
        medication = medication.lower()
        
        # Look for exact match
        med_data = self.drug_data[self.drug_data['medication'] == medication]
        
        # If not found, try partial match
        if len(med_data) == 0:
            med_data = self.drug_data[self.drug_data['medication'].str.contains(medication, case=False)]
        
        if len(med_data) == 0:
            logger.warning(f"Medication '{medication}' not found in database")
            return None
        
        if len(med_data) > 1:
            logger.warning(f"Multiple matches found for '{medication}', using first match")
            med_data = med_data.iloc[0:1]
        
        # Extract receptor profile
        receptor_features = self.feature_info['receptor_features']
        receptor_profile = {feature: med_data[feature].values[0] for feature in receptor_features}
        
        return receptor_profile
    
    def predict_response(self, 
                        medication: str, 
                        dosage: Optional[float] = None) -> Dict[str, Any]:
        """
        Predict response to a medication.

        Args:
            medication: Medication name
            dosage: Dosage in mg (optional)

        Returns:
            Dictionary with predicted responses
        """
        # Get receptor profile
        receptor_profile = self.get_receptor_profile(medication)
        if receptor_profile is None:
            return {
                'error': f"Medication '{medication}' not found in database",
                'found': False
            }
        
        # Get medication data
        med_data = self.drug_data[self.drug_data['medication'] == medication.lower()]
        if len(med_data) == 0:
            med_data = self.drug_data[self.drug_data['medication'].str.contains(medication.lower(), case=False)]
        
        if len(med_data) == 0:
            return {
                'error': f"Medication '{medication}' not found in database",
                'found': False
            }
        
        # Create feature vector
        feature_values = []
        for feature in self.feature_info['feature_columns']:
            if feature in receptor_profile:
                feature_values.append(receptor_profile[feature])
            elif feature == 'normalized_dosage' and dosage is not None:
                if 'typical_dose' in med_data.columns:
                    typical_dose = med_data['typical_dose'].values[0]
                    if pd.notna(typical_dose) and typical_dose > 0:
                        feature_values.append(dosage / typical_dose)
                    else:
                        feature_values.append(0)
                else:
                    feature_values.append(0)
            elif feature == 'has_dosage':
                feature_values.append(1 if dosage is not None else 0)
            elif feature.startswith('class_'):
                # Handle drug class one-hot features
                class_name = feature.replace('class_', '')
                if 'medication_class' in med_data.columns and med_data['medication_class'].values[0] == class_name:
                    feature_values.append(1)
                else:
                    feature_values.append(0)
            else:
                # Default value for unknown features
                feature_values.append(0)
        
        # Create DMatrix for prediction
        dmatrix = xgb.DMatrix(np.array([feature_values]))
        
        # Make predictions for each dimension
        predictions = {}
        for target, model in self.models.items():
            prediction = model.predict(dmatrix)[0]
            predictions[target] = float(prediction)
        
        # Format response
        result = {
            'medication': medication,
            'found': True,
            'dosage': dosage,
            'predictions': {
                'activation_sedation_score': predictions['activation'],
                'emotional_blunting_restoration_score': predictions['emotional'],
                'appetite_metabolic_score': predictions['metabolic']
            },
            'medication_info': {
                'class': med_data['medication_class'].values[0] if 'medication_class' in med_data.columns else 'unknown',
                'receptor_profile': receptor_profile
            }
        }
        
        return result
    
    def batch_predict(self, medications: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """
        Batch predict responses for multiple medications.

        Args:
            medications: List of dictionaries with medication name and optional dosage

        Returns:
            List of prediction results
        """
        results = []
        for med_info in medications:
            medication = med_info.get('medication', '')
            dosage = med_info.get('dosage')
            
            result = self.predict_response(medication, dosage)
            results.append(result)
        
        return results
    
    def interpret_prediction(self, prediction: Dict[str, Any]) -> Dict[str, str]:
        """
        Interpret prediction results in plain language.

        Args:
            prediction: Prediction results from predict_response

        Returns:
            Dictionary with interpretations for each dimension
        """
        if not prediction.get('found', False):
            return {
                'error': prediction.get('error', 'Medication not found in database')
            }
        
        scores = prediction['predictions']
        
        # Interpret activation-sedation axis
        act_score = scores['activation_sedation_score']
        if act_score > 0.7:
            activation = "Very activating - likely to cause significant energy increase and potential sleep disruption"
        elif act_score > 0.6:
            activation = "Moderately activating - may increase energy and alertness"
        elif act_score > 0.4:
            activation = "Neutral effect on energy/sedation - balanced impact"
        elif act_score > 0.3:
            activation = "Mildly sedating - may cause some drowsiness"
        else:
            activation = "Strongly sedating - likely to cause significant drowsiness"
        
        # Interpret emotional blunting-restoration axis
        emo_score = scores['emotional_blunting_restoration_score']
        if emo_score > 0.7:
            emotional = "Strong emotional restoration - likely to improve emotional range and responsiveness"
        elif emo_score > 0.6:
            emotional = "Moderate emotional restoration - may enhance emotional experience"
        elif emo_score > 0.4:
            emotional = "Neutral effect on emotions - balanced impact"
        elif emo_score > 0.3:
            emotional = "Mild emotional blunting - may slightly dampen emotional response"
        else:
            emotional = "Strong emotional blunting - likely to cause significant emotional numbing"
        
        # Interpret appetite/metabolic axis
        met_score = scores['appetite_metabolic_score']
        if met_score > 0.7:
            metabolic = "Strong appetite increase - likely to cause significant weight gain"
        elif met_score > 0.6:
            metabolic = "Moderate appetite increase - may lead to some weight gain"
        elif met_score > 0.4:
            metabolic = "Neutral effect on appetite/metabolism - balanced impact"
        elif met_score > 0.3:
            metabolic = "Mild appetite suppression - may cause slight weight loss"
        else:
            metabolic = "Strong appetite suppression - likely to cause significant weight loss"
        
        # Return interpretations
        return {
            'activation_sedation': activation,
            'emotional_blunting_restoration': emotional,
            'appetite_metabolic': metabolic
        }

def main():
    """Main entry point"""
    parser = argparse.ArgumentParser(description="Predict medication responses")
    parser.add_argument("--model", required=True, help="Path to trained model pickle file")
    parser.add_argument("--affinity", required=True, help="Path to drug receptor affinity CSV")
    parser.add_argument("--medication", required=True, help="Medication name to predict for")
    parser.add_argument("--dosage", type=float, help="Medication dosage in mg (optional)")
    parser.add_argument("--interpret", action="store_true", help="Provide interpretation of results")
    
    args = parser.parse_args()
    
    try:
        # Initialize predictor
        predictor = MedicationResponsePredictor(args.model, args.affinity)
        
        # Predict response
        prediction = predictor.predict_response(args.medication, args.dosage)
        
        if prediction.get('found', False):
            print(f"Predictions for {prediction['medication']}:")
            print(f"  Activation-Sedation Score: {prediction['predictions']['activation_sedation_score']:.3f}")
            print(f"  Emotional Blunting-Restoration Score: {prediction['predictions']['emotional_blunting_restoration_score']:.3f}")
            print(f"  Appetite/Metabolic Score: {prediction['predictions']['appetite_metabolic_score']:.3f}")
            
            if args.interpret:
                interpretation = predictor.interpret_prediction(prediction)
                print("\nInterpretation:")
                print(f"  Activation-Sedation: {interpretation['activation_sedation']}")
                print(f"  Emotional Effect: {interpretation['emotional_blunting_restoration']}")
                print(f"  Appetite/Metabolic Effect: {interpretation['appetite_metabolic']}")
        else:
            print(f"Error: {prediction.get('error', 'Unknown error')}")
        
    except Exception as e:
        logger.error(f"Error making prediction: {e}", exc_info=True)
        print(f"Error: {str(e)}")
        exit(1)

if __name__ == "__main__":
    main() 
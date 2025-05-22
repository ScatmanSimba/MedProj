#!/bin/bash

# Create necessary directories
mkdir -p data/{raw,processed,external}
mkdir -p models
mkdir -p logs

# Set up Python environment
python -m venv venv
source venv/bin/activate
pip install -r requirements.txt

# Run data collection and processing
echo "Running data collection and processing..."
python src/data_collection/snapshot.py
python src/data/data_loader.py

# Run feature engineering
echo "Running feature engineering..."
python src/features/build_features.py

# Train model
echo "Training model..."
python src/models/train_model.py

# Generate evaluation report
echo "Generating evaluation report..."
python src/evaluation/generate_report.py

echo "Pipeline completed successfully!" 
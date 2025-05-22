# Project Requirement Document (PRD)

## Project Title

**Psychiatric Medication Response Predictor: Receptor-Based Phenotyping**

## Project Vision

Develop a computational system that predicts how patients will respond to antidepressants and atypical antipsychotics by inferring a compact 5-receptor sensitivity profile (D2, 5-HT2A, H1, α1, M1) from Reddit self-reports. The goal is to outperform diagnosis-only or sentiment-only baselines on three clinically salient response axes:

1. Activation ↔ Sedation
2. Emotional Blunting ↔ Restoration
3. Appetite / Metabolic impact

## Core Hypothesis

Stable patterns in the three response dimensions can be mapped to a five-receptor vector. Knowing that vector allows for forecasting response to any new drug whose binding affinities are known.

---

## File Structure

```
/
├── data/                               # Data directory
│   ├── raw/                            # Raw collected data
│   │   └── reddit_posts.parquet        # Raw Reddit posts
│   ├── processed/                      # Processed datasets
│   │   ├── user_timelines.parquet      # Timeline data by user
│   │   └── annotated_responses.parquet # Posts with response annotations
│   └── external/                       # External reference data
│       └── drug_receptor_affinity.csv  # Drug receptor binding data
├── embeddings/                         # Text embeddings (if needed)
│   └── transformer_embeddings.parquet  # DistilBERT embeddings for classification
├── src/                                # Python source files
│   ├── snapshot.py                     # Reddit data collector
│   ├── med_dictionary1803.py          # Medication dictionary and NER
│   ├── features/                       # Feature engineering
│   │   ├── symptom_matcher.py         # Symptom matching and classification
│   │   ├── temporal_parser.py         # Temporal relationship analysis
│   │   ├── emoji_processor.py         # Emoji signal processing
│   │   ├── response_attribution.py    # Response dimension attribution
│   │   ├── build_features.py          # Main feature builder
│   │   └── response_classifier.py     # Response axis classifier
│   ├── models/                         # Model implementation
│   │   ├── train_model.py             # Multi-task GBT trainer
│   │   └── predictor.py               # Model prediction interface
│   ├── evaluation/                     # Evaluation utilities
│   │   ├── metrics.py                 # Performance metrics calculators
│   │   └── shap_analysis.py           # SHAP-based feature importance analysis
│   └── utils/                          # Helper utilities
│       ├── logger.py                  # Logging configuration
│       └── constants.py               # Project constants and configurations
├── notebooks/                          # Jupyter notebooks
│   ├── eda.ipynb                      # Exploratory data analysis
│   ├── response_annotation.ipynb      # Response dimension annotation development
│   └── model_evaluation.ipynb         # Model performance analysis
├── tests/                              # Test suite
│   ├── test_snapshot.py               # Test Reddit data collection
│   ├── test_parsers.py                # Test feature extraction parsers
│   ├── test_response_classifier.py    # Test response classification
│   └── test_model.py                  # Test model training/prediction
├── configs/                            # Configuration files
│   ├── model_config.yaml              # Model hyperparameters
│   └── feature_config.yaml            # Feature engineering parameters
├── .cursorrules                        # Cursor generation rules
├── requirements.txt                    # Python dependencies
├── run_all.sh                          # End-to-end pipeline script
├── evaluation.md                       # Evaluation results and analysis
├── PRD.md                              # This project description file
└── README.md                           # Project documentation
```

## Enhanced Temporal Understanding

To address a critical challenge in medication response prediction, we've implemented a sophisticated temporal analysis system that can distinguish between current, past, and prospective medication usage. This system:

1. Builds a medication-time-effect graph representation (`med_temporal_graph.py`):
   - Nodes represent medications, effects, and temporal markers
   - Edges capture causal relationships and temporal dependencies
   - Implements pattern matching for direct effects and side effects
   - Supports emoji-based emotional signals with polarity detection
   - Handles both sentence-level and document-level relationships

2. Uses advanced NLP techniques:
   - Dependency parsing for temporal relationship identification
   - Coreference resolution for medication mentions
   - Negation detection with temporal scope
   - Causal pattern matching with confidence scoring
   - Emoji pattern recognition for emotional signals

3. Implements sophisticated response attribution (`response_attribution.py`):
   - Multi-dimensional response tracking (activation, emotional, metabolic)
   - Proximity-based attribution with distance decay
   - Causal link detection with confidence scoring
   - Intensity modifier detection
   - Emoji signal integration with polarity normalization
   - Duplicate detection prevention
   - Confidence-aware scoring system

4. Provides comprehensive confidence scoring:
   - Temporal clarity assessment
   - Attribution strength measurement
   - Detail level evaluation
   - Language pattern analysis
   - Emoji signal confidence
   - Causal relationship strength

5. Features advanced pattern matching:
   - Direct effect patterns (e.g., "has me", "makes me", "is making me")
   - Side effect patterns with confidence scoring
   - Reddit emoji patterns with polarity detection
   - Bidirectional emoji-medication relationships
   - Temporal marker patterns

The system also includes a robust attribution framework that links response dimensions to specific medications based on:
- Textual proximity with configurable distance tolerance
- Explicit causality statements with confidence scoring
- Linguistic coreference patterns
- Emoji-based emotional signals
- Temporal relationship strength

Each medication-response association receives a confidence score based on:
- Temporal clarity
- Attribution strength
- Detail level
- Language patterns
- Emoji signal presence and strength
- Causal relationship confidence

This confidence score is propagated through the model, allowing it to learn more effectively from high-confidence examples while still incorporating information from lower-confidence data points.

---

## Component A: Data Collection & Pre-processing

### Input
- Reddit API credentials
- Target subreddits list (r/antidepressants, r/Abilify, etc.)
- Date range (2022-present)

### Process
- Implement `snapshot.py` to collect posts from target subreddits
- Apply text preprocessing (remove formatting, URLs, normalize text)
- Implement medication NER using:
  - Primary: Dictionary-based regex matching with med_dictionary
  - Fallback: SciSpacy for missed medications
- Extract dosage information where available
- Group posts by user to build medication timelines
- Store user, date, drug, dose, and text information

### Output
- `user_timelines.parquet`: Contains user-level medication timelines with timestamps
- Data schema:
  ```
  user_id: string (anonymized)
  post_date: timestamp
  medication: string
  dosage: float (standardized to mg)
  dosage_unit: string
  post_text: string
  subreddit: string
  post_id: string
  ```

### Validation
- Verify medication extraction precision on 100 random posts
- Check for temporal consistency in user timelines
- Validate data completeness across required fields

---

## Component B: Response-Dimension Annotation

### Input
- `user_timelines.parquet`
- Keyword lists for each response dimension

### Process
- Develop keyword-based rules for initial annotation
- Train a small DistilBERT classifier for each response axis
- Apply classifiers to generate continuous scores (0-1) for each axis
- Implement temporal analysis to determine medication status
- Calculate confidence scores based on:
  - Temporal clarity of medication usage
  - Attribution strength to specific medications
  - Level of detail in response description
  - Linguistic patterns indicating certainty
- Manually validate precision on 500 random posts
- Refine classifiers based on validation results

### Output
- `annotated_responses.parquet`: Timeline data with response dimension scores
- Data schema:
  ```
  [all fields from user_timelines.parquet]
  activation_sedation_score: float (0-1)
  emotional_blunting_restoration_score: float (0-1)
  appetite_metabolic_score: float (0-1)
  confidence_scores: array(float) [3]
  medication_status: string (current/past/brief/prospective)
  attribution_confidence: float (0-1)
  temporal_confidence: float (0-1)
  ```

### Validation
- Inter-rater reliability metrics for manual annotations
- Classifier performance metrics (precision, recall, F1)
- Distribution analysis of scores for each axis
- Temporal classification accuracy assessment
- Confidence score correlation with prediction accuracy

---

## Component C: Pharmacology Knowledge Base

### Input
- External pharmacological databases or literature
- List of medications identified in the dataset

### Process
- Compile binding affinity (Ki) data for all included medications
- Focus on five target receptors: D2, 5-HT2A, H1, α1, M1
- Normalize values to log(Ki)
- Create a standardized reference table

### Output
- `drug_receptor_affinity.csv`: Reference table of receptor binding data
- Schema:
  ```
  medication: string
  medication_class: string
  d2_affinity: float (log Ki)
  5ht2a_affinity: float (log Ki)
  h1_affinity: float (log Ki)
  alpha1_affinity: float (log Ki)
  m1_affinity: float (log Ki)
  half_life: float (hours)
  typical_dose: float (mg)
  ```

### Validation
- Cross-reference values with multiple sources
- Check completeness for all medications in the dataset
- Validate normalization approach

---

## Component D: Receptor-Aware Predictor (MVP Model)

### Input
- `annotated_responses.parquet`
- `drug_receptor_affinity.csv`

### Process
- Merge response data with receptor profiles
- Engineer features:
  - Drug receptor vector (5 dimensions)
  - Basic text features from posts
  - Dosage information
  - User demographic features (if available)
  - Temporal features (medication status, duration)
  - Confidence metrics (attribution, temporal)
- Implement multi-task Gradient Boosted Trees using LightGBM or XGBoost
- Use confidence scores as sample weights during training
- Implement stratified evaluation by confidence tiers
- Train model to predict three continuous response scores
- Implement 5-fold cross-validation

### Output
- `model.pkl`: Trained multi-task GBT model
- `feature_importance.csv`: Feature importance metrics
- `predictions.csv`: Cross-validation predictions
- `confidence_analysis.csv`: Performance metrics by confidence tier

### Validation
- Learning curves to detect overfitting
- Feature importance analysis
- Cross-validation metrics (R², RMSE)
- Performance analysis by confidence tier
- Confidence score correlation with prediction accuracy

---

## Component E: Evaluation & Interpretability

### Input
- Trained model
- Test dataset (held-out portion)
- Baseline models for comparison

### Process
- Evaluate model performance using 5-fold CV
- Compare against baselines:
  1. Majority baseline
  2. Sentiment-only model
  3. Drug-class dummy predictor
- Generate SHAP plots for feature attribution
- Analyze receptor contribution to predictions
- Document findings and limitations

### Output
- `evaluation.md`: Comprehensive evaluation report
- `metrics.csv`: Performance metrics for main model and baselines
- SHAP visualization plots
- Error analysis documentation

### Validation
- Statistical significance of improvements over baselines
- Verification that ≥3 of top-5 features are receptor dimensions
- Segment-based evaluation by drug class

---

## Optional Stretch Goals (Post-MVP)

- Unsupervised clustering of user sensitivity profiles
- Side-effect lexical layer implementation
- Fine-tuning of response dimension classifiers
- Temporal analysis of response patterns

---

## Technical Requirements

- Python package structure (src/)
- Reproducible random seeds for all stochastic operations
- requirements.txt for dependency management
- LightGBM or XGBoost for model implementation
- SHAP for feature attribution analysis
- Ethics documentation addressing:
  - Use of public data
  - Absence of PHI
  - Takedown policy for sensitive content

---

## Implementation Priorities

The enhanced temporal understanding components should be implemented in this order:

1. **Core Temporal Extraction** - Implement basic medication status detection (current/past)
2. **Confidence Scoring** - Add multi-factor confidence calculation
3. **Response Attribution** - Link effects to specific medications with confidence
4. **Enhanced Features** - Build temporal and confidence features
5. **Confidence-Aware Training** - Use confidence as sample weights
6. **Stratified Evaluation** - Analyze performance by confidence tier

This staged approach ensures we can realize benefits quickly while building toward the complete solution.

## Success Criteria (MVP)

| Metric | Target |
|--------|--------|
| Macro-R² across three axes | ≥ 0.25 (beats sentiment baseline by ≥ 5 points) |
| SHAP top-5 features | ≥ 3 are receptor dimensions (demonstrates pharmacology signal) |
| Reproducibility | run_all.sh reproduces metrics end-to-end |
| **High-confidence R²** | **≥ 0.40 for predictions with confidence score > 0.7** |
| **Temporal accuracy** | **≥ 85% accuracy on manual inspection of 100 temporal classifications** |
| **Confidence correlation** | **Negative correlation (≤ -0.3) between confidence and prediction error** |

These additional criteria ensure we're properly measuring the quality improvements from our enhanced temporal understanding.

---

## Implementation Guidelines

- All Python source code must reside in `/src`
- Use a modular design with clear separation of concerns
- Each module must include docstrings and examples
- Implement proper logging throughout the pipeline
- Handle exceptions gracefully with informative messages
- Keep computational requirements reasonable for Colab environment
- Document all hyperparameters and configuration options
- Implement unit tests for critical components

---

## Timeline (90 Days)

| Week | Milestone |
|------|-----------|
| 1-2 | Finalize med_dictionary; collect 50k raw posts |
| 3-4 | Build response-dimension keyword lists; label sample; train DistilBERT tagger |
| 5-6 | Assemble receptor affinity CSV; merge with timelines |
| 7-8 | Train GBT model, run CV, log metrics |
| 9 | SHAP interpretability, error analysis |
| 10 | Polish docs, push repo, record demo video |

---

## CLI Flow Example

```bash
# Collect Reddit data
python -m src.data.snapshot --subreddits r/antidepressants,r/Abilify --start_date 2022-01-01 --output data/raw/reddit_posts.parquet

# Build user timelines with medication extraction
python -m src.data.med_ner --input data/raw/reddit_posts.parquet --output data/processed/user_timelines.parquet

# Generate response annotations
python -m src.features.response_classifier --input data/processed/user_timelines.parquet --output data/processed/annotated_responses.parquet

# Train model
python -m src.models.train_model --data data/processed/annotated_responses.parquet --affinity data/external/drug_receptor_affinity.csv --output models/receptor_predictor.pkl

# Evaluate model
python -m src.evaluation.metrics --model models/receptor_predictor.pkl --data data/processed/test_set.parquet --baselines
```

---

## Assumptions & Notes

- Reddit data is publicly available and does not contain PHI
- The 5-receptor model is a simplification but sufficient for MVP
- Response dimensions can be reliably extracted from self-reports
- User medication reporting is assumed to be generally accurate
- Sufficient data exists for the included medications
- Project scope is limited to two drug classes for feasibility

This PRD serves as the blueprint for the Psychiatric Medication Response Predictor and will guide Cursor in generating appropriate code for the implementation.

"""Response dimension classifier for psychiatric medication posts.

This module implements classifiers for the three response dimensions:
1. Activation ↔ Sedation
2. Emotional Blunting ↔ Restoration
3. Appetite / Metabolic impact

The implementation uses both keyword-based rules and DistilBERT-based classification
to generate continuous scores (0-1) for each dimension.
"""

import os
from typing import Dict, List, Tuple, Optional, Union, Any
import pandas as pd
import numpy as np
from transformers import DistilBertTokenizer, DistilBertModel
import torch
from torch import nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score
from sklearn.isotonic import IsotonicRegression
import logging
from pathlib import Path
import spacy
from dataclasses import dataclass
import json
import yaml

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

@dataclass
class ResponseDimensionKeywords:
    """Keywords and their polarities for each response dimension."""
    activation_keywords: Dict[str, float]  # word -> polarity
    emotional_keywords: Dict[str, float]
    metabolic_keywords: Dict[str, float]
    
    def __post_init__(self):
        # Configure logging
        self.logger = logging.getLogger(__name__)
        
        try:
            # Load spaCy model for negation detection
            self.nlp = spacy.load("en_core_web_sm")
        except Exception as e:
            self.logger.error(f"Failed to load spaCy model: {e}")
            raise
        
        # Initialize counters for rule tracking
        self.rule_counts = {
            'slang_matches': 0,
            'intensifier_matches': 0,
            'diminisher_matches': 0,
            'negation_matches': 0,
            'cant_sleep_matches': 0
        }
        
        # Load or initialize slang mapping
        self.slang_map = self._load_vocabulary('slang_map.json', {
            "wired af": "wired",
            "knackered": "tired",
            "zonked": "tired",
            "amped": "energetic",
            "wiped": "tired",
            "dead": "tired",
            "beat": "tired",
            "pooped": "tired",
            "buzzed": "energetic",
            "jacked": "energetic",
            "stoked": "energetic",
            "wrecked": "tired",
            "exhausted af": "tired",
            "tired af": "tired",
            "energetic af": "energetic"
        })
        
        # Load or initialize intensifiers and diminishers
        self.intensifiers = self._load_vocabulary('intensifiers.json', {
            "very": 1.3,
            "super": 1.3,
            "extremely": 1.3,
            "really": 1.2,
            "quite": 1.2,
            "totally": 1.3,
            "completely": 1.3,
            "absolutely": 1.3
        })
        
        self.diminishers = self._load_vocabulary('diminishers.json', {
            "slightly": 0.7,
            "bit": 0.7,
            "kinda": 0.7,
            "somewhat": 0.7,
            "rather": 0.8,
            "fairly": 0.8,
            "moderately": 0.8
        })
        
        # Compile regex patterns
        import re
        self.cant_sleep_pattern = re.compile(r"(can'?t|cannot)\s+(?:get|fall|go)\s+to\s+sleep", re.IGNORECASE)
        
    def _load_vocabulary(self, filename: str, default_vocab: Dict[str, Any]) -> Dict[str, Any]:
        """Load vocabulary from file or use default if file doesn't exist.
        
        Args:
            filename: Name of the vocabulary file
            default_vocab: Default vocabulary to use if file doesn't exist
            
        Returns:
            Loaded or default vocabulary
        """
        try:
            vocab_path = Path('data/vocabulary') / filename
            if vocab_path.exists():
                with open(vocab_path, 'r') as f:
                    return json.load(f)
            else:
                # Save default vocabulary
                vocab_path.parent.mkdir(parents=True, exist_ok=True)
                with open(vocab_path, 'w') as f:
                    json.dump(default_vocab, f, indent=2)
                return default_vocab
        except Exception as e:
            self.logger.warning(f"Failed to load vocabulary from {filename}: {e}")
            return default_vocab
            
    def save_vocabulary(self) -> None:
        """Save current vocabulary to files."""
        try:
            vocab_dir = Path('data/vocabulary')
            vocab_dir.mkdir(parents=True, exist_ok=True)
            
            with open(vocab_dir / 'slang_map.json', 'w') as f:
                json.dump(self.slang_map, f, indent=2)
            with open(vocab_dir / 'intensifiers.json', 'w') as f:
                json.dump(self.intensifiers, f, indent=2)
            with open(vocab_dir / 'diminishers.json', 'w') as f:
                json.dump(self.diminishers, f, indent=2)
                
            self.logger.info("Saved vocabulary files successfully")
        except Exception as e:
            self.logger.error(f"Failed to save vocabulary: {e}")
            
    def get_lemma(self, token: spacy.tokens.Token) -> str:
        """Get lemma of token, handling slang and special cases."""
        try:
            # Check for multi-word slang first
            if token.i > 0:
                prev_token = token.doc[token.i - 1]
                bigram = f"{prev_token.text.lower()} {token.text.lower()}"
                if bigram in self.slang_map:
                    self.rule_counts['slang_matches'] += 1
                    self.logger.debug(f"Matched slang bigram: {bigram} -> {self.slang_map[bigram]}")
                    return self.slang_map[bigram]
            
            # Check for single-word slang
            if token.text.lower() in self.slang_map:
                self.rule_counts['slang_matches'] += 1
                self.logger.debug(f"Matched slang word: {token.text} -> {self.slang_map[token.text.lower()]}")
                return self.slang_map[token.text.lower()]
                
            # Return lemma for regular words
            return token.lemma_.lower()
        except Exception as e:
            self.logger.warning(f"Error in get_lemma for token '{token.text}': {e}")
            return token.text.lower()
        
    def get_intensity_modifier(self, doc: spacy.tokens.Doc, token_idx: int) -> float:
        """Get intensity modifier based on nearby intensifiers/diminishers."""
        try:
            modifier = 1.0
            
            # Check tokens in window
            for i in range(max(0, token_idx - 2), min(len(doc), token_idx + 3)):
                token = doc[i]
                if token.text.lower() in self.intensifiers:
                    self.rule_counts['intensifier_matches'] += 1
                    modifier *= self.intensifiers[token.text.lower()]
                    self.logger.debug(f"Applied intensifier: {token.text} ({modifier})")
                elif token.text.lower() in self.diminishers:
                    self.rule_counts['diminisher_matches'] += 1
                    modifier *= self.diminishers[token.text.lower()]
                    self.logger.debug(f"Applied diminisher: {token.text} ({modifier})")
                    
            return modifier
        except Exception as e:
            self.logger.warning(f"Error in get_intensity_modifier: {e}")
            return 1.0
        
    def detect_negation(self, doc: spacy.tokens.Doc, token_idx: int) -> bool:
        """Enhanced negation detection with context window and sentence-level search."""
        try:
            token = doc[token_idx]
            
            # Check for "no longer" pattern
            if token_idx > 0 and doc[token_idx-1].text.lower() == "no" and token.text.lower() == "longer":
                self.rule_counts['negation_matches'] += 1
                self.logger.debug("Matched 'no longer' pattern")
                return True
            
            # Check for "without" pattern
            if token_idx > 0 and doc[token_idx-1].text.lower() == "without":
                self.rule_counts['negation_matches'] += 1
                self.logger.debug("Matched 'without' pattern")
                return True
                
            # Check for negation in ancestors
            for ancestor in token.ancestors:
                if ancestor.dep_ == "neg":
                    self.rule_counts['negation_matches'] += 1
                    self.logger.debug(f"Found negation in ancestor: {ancestor.text}")
                    return True
                    
            # Check for hedge words in preceding tokens
            hedge_words = {"barely", "hardly", "not", "scarcely", "seldom", "rarely", "never"}
            for i in range(max(0, token_idx - 3), token_idx):
                if doc[i].text.lower() in hedge_words:
                    self.rule_counts['negation_matches'] += 1
                    self.logger.debug(f"Found hedge word: {doc[i].text}")
                    return True
            
            # Check for negation within the same sentence
            sent = token.sent
            for t in sent:
                if t.dep_ == "neg":
                    self.rule_counts['negation_matches'] += 1
                    self.logger.debug(f"Found negation in sentence: {t.text}")
                    return True
                    
            # Check for "couldn't sleep" pattern
            if self.cant_sleep_pattern.search(sent.text):
                self.rule_counts['cant_sleep_matches'] += 1
                self.logger.debug("Matched 'couldn't sleep' pattern")
                return True
                    
            return False
        except Exception as e:
            self.logger.warning(f"Error in detect_negation: {e}")
            return False
        
    def get_rule_stats(self) -> Dict[str, int]:
        """Get statistics about rule matches.
        
        Returns:
            Dictionary with counts of each rule type
        """
        return self.rule_counts.copy()
        
    def reset_rule_stats(self) -> None:
        """Reset rule match statistics."""
        self.rule_counts = {k: 0 for k in self.rule_counts}
        
    def get_keyword_scores_batch(self, texts: List[str]) -> Dict[str, List[float]]:
        """Process keyword scores in batch for efficiency.
        
        Args:
            texts: List of input texts
            
        Returns:
            Dictionary of scores for each dimension
        """
        # Filter out invalid inputs
        valid_texts = [text for text in texts if self._validate_inputs(text)]
        if not valid_texts:
            return {dim: [0.5] * len(texts) for dim in self.classifiers.keys()}
        
        # Process texts in batch
        docs = list(self.nlp.pipe(valid_texts))
        
        # Initialize results
        scores = {dim: [0.0] * len(texts) for dim in self.classifiers.keys()}
        counts = {dim: [0] * len(texts) for dim in self.classifiers.keys()}
        
        # Map dimensions to their keyword dictionaries
        dim_keywords = {
            'activation_sedation': self.activation_keywords,
            'emotional_blunting': self.emotional_keywords,
            'appetite_metabolic': self.metabolic_keywords
        }
        
        # Process each document
        for doc_idx, doc in enumerate(docs):
            for token in doc:
                # Get lemma and check if it's a keyword
                lemma = self.get_lemma(token)
                
                for dim, keywords in dim_keywords.items():
                    if lemma in keywords:
                        polarity = keywords[lemma]
                        
                        # Apply intensity modifier
                        intensity = self.get_intensity_modifier(doc, token.i)
                        polarity *= intensity
                        
                        # Apply negation if detected
                        if self.detect_negation(doc, token.i):
                            polarity *= -1
                            
                        scores[dim][doc_idx] += polarity
                        counts[dim][doc_idx] += 1
        
        # Normalize scores
        for dim in scores:
            for i in range(len(texts)):
                if counts[dim][i] > 0:
                    scores[dim][i] = (scores[dim][i] / counts[dim][i] + 1) / 2  # Scale to 0-1
                else:
                    scores[dim][i] = 0.5  # Default score for no keywords
                
        return scores

class ResponseDataset(Dataset):
    """PyTorch dataset for response dimension classification."""
    
    def __init__(self, texts: List[str], labels: List[float], tokenizer: DistilBertTokenizer, max_length: int = 512):
        """Initialize the dataset.
        
        Args:
            texts: List of text samples
            labels: List of continuous labels (0-1)
            tokenizer: DistilBERT tokenizer
            max_length: Maximum sequence length
        """
        self.texts = texts
        self.labels = labels
        self.tokenizer = tokenizer
        self.max_length = max_length
        
    def __len__(self) -> int:
        return len(self.texts)
    
    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        text = str(self.texts[idx])
        label = self.labels[idx]
        
        encoding = self.tokenizer(
            text,
            add_special_tokens=True,
            max_length=self.max_length,
            padding='max_length',
            truncation=True,
            return_tensors='pt'
        )
        
        return {
            'input_ids': encoding['input_ids'].flatten(),
            'attention_mask': encoding['attention_mask'].flatten(),
            'label': torch.tensor(label, dtype=torch.float),
            'text': text  # Include text for keyword processing
        }

class ResponseClassifier(nn.Module):
    """DistilBERT-based classifier for response dimensions."""
    
    def __init__(self, 
                 model_name: str = 'distilbert-base-uncased',
                 dropout_rate: float = 0.1,
                 mc_samples: int = 20):
        """Initialize the classifier.
        
        Args:
            model_name: Name of the pre-trained DistilBERT model
            dropout_rate: Dropout rate for MC-Dropout
            mc_samples: Number of Monte Carlo samples for uncertainty
        """
        super().__init__()
        self.bert = DistilBertModel.from_pretrained(model_name)
        self.dropout = nn.Dropout(dropout_rate)
        self.classifier = nn.Linear(768, 1)  # Regression head
        self.mc_samples = mc_samples
        
        # Initialize calibration model
        self.calibration = IsotonicRegression(out_of_bounds='clip')
        self.is_calibrated = False
        
    def calibrate(self, predictions: np.ndarray, targets: np.ndarray) -> None:
        """Calibrate predictions using isotonic regression."""
        self.calibration.fit(predictions, targets)
        self.is_calibrated = True
        
    def forward(self, 
                input_ids: torch.Tensor, 
                attention_mask: torch.Tensor,
                return_uncertainty: bool = False) -> Dict[str, torch.Tensor]:
        """Forward pass.
        
        Args:
            input_ids: Input token IDs
            attention_mask: Attention mask
            return_uncertainty: Whether to return uncertainty estimates
            
        Returns:
            Dictionary with predictions and optional uncertainty
        """
        # Keep in training mode for MC-Dropout
        self.train()
        
        # Get BERT embeddings
        outputs = self.bert(input_ids=input_ids, attention_mask=attention_mask)
        pooled_output = outputs[0][:, 0, :]  # Use [CLS] token
        pooled_output = self.dropout(pooled_output)
        
        # MC-Dropout sampling
        predictions = []
        for _ in range(self.mc_samples):
            pred = self.classifier(pooled_output)
            predictions.append(pred)
            
        predictions = torch.stack(predictions)
        mean_pred = predictions.mean(dim=0)
        std_pred = predictions.std(dim=0)
        
        # Apply calibration if trained
        if self.is_calibrated:
            mean_pred = torch.tensor(
                self.calibration.predict(mean_pred.detach().cpu().numpy()),
                device=mean_pred.device
            )
        
        result = {'prediction': mean_pred}
        if return_uncertainty:
            result['uncertainty'] = std_pred
            
        return result

class ResponseDimensionClassifier:
    """Main class for response dimension classification."""
    
    def __init__(self, model_dir: str = 'models/response_classifiers'):
        """Initialize the classifier.
        
        Args:
            model_dir: Directory to save/load models
        """
        self.model_dir = Path(model_dir)
        self.model_dir.mkdir(parents=True, exist_ok=True)
        
        self.tokenizer = DistilBertTokenizer.from_pretrained('distilbert-base-uncased')
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
        # Load dimension weights from config
        config_path = Path('configs/feature_config.yaml')
        if config_path.exists():
            with open(config_path, 'r') as f:
                config = yaml.safe_load(f)
                self.dimension_weights = config.get('dimension_weights', {
                    'activation': 1.0,
                    'emotional': 1.0,
                    'metabolic': 1.0
                })
        else:
            self.dimension_weights = {
                'activation': 1.0,
                'emotional': 1.0,
                'metabolic': 1.0
            }
        
        # Initialize keyword matcher with improved polarities
        self.keywords = ResponseDimensionKeywords(
            activation_keywords={
                # Activation (positive)
                'energetic': 0.9, 'alert': 0.8, 'stimulated': 0.85, 'wired': 0.9,
                'restless': 0.7, 'jittery': 0.75, 'amped': 0.8, 'motivated': 0.85,
                'awake': 0.8, 'insomnia': 0.7, 'can\'t sleep': 0.7,
                
                # Sedation (negative)
                'sedated': -0.9, 'drowsy': -0.8, 'tired': -0.7, 'sleepy': -0.8,
                'lethargic': -0.85, 'zonked': -0.9, 'sluggish': -0.75,
                'fatigued': -0.8, 'calm': -0.6, 'relaxed': -0.5
            },
            emotional_keywords={
                # Blunting (negative)
                'numb': -0.9, 'flat': -0.85, 'blunted': -0.9, 'empty': -0.8,
                'no emotions': -0.9, 'can\'t feel': -0.85, 'dull': -0.75,
                'apathetic': -0.8, 'indifferent': -0.7,
                
                # Restoration (positive)
                'feel again': 0.9, 'emotions back': 0.85, 'more emotional': 0.8,
                'crying': 0.7, 'happy': 0.8, 'sad': 0.7, 'joy': 0.85,
                'excited': 0.8, 'emotional range': 0.75, 'vibrant': 0.85,
                'alive': 0.8, 'responsive': 0.75
            },
            metabolic_keywords={
                # Appetite increase (positive)
                'hungry': 0.8, 'craving': 0.85, 'eating more': 0.75,
                'weight gain': 0.7, 'increased appetite': 0.8,
                'always hungry': 0.85, 'snacking': 0.7,
                
                # Appetite decrease (negative)
                'no appetite': -0.9, 'not hungry': -0.8, 'eating less': -0.75,
                'weight loss': -0.7, 'decreased appetite': -0.8,
                'forget to eat': -0.75, 'nausea': -0.6
            }
        )
        
        # Initialize classifiers for each dimension
        self.classifiers = {
            'activation_sedation': ResponseClassifier(),
            'emotional_blunting': ResponseClassifier(),
            'appetite_metabolic': ResponseClassifier()
        }
        
        # Initialize stacking weights
        self.stacking_weights = nn.ParameterDict({
            dim: nn.Parameter(torch.tensor([0.7, 0.3]))  # Initial weights for BERT/keyword
            for dim in self.classifiers.keys()
        })
        
        # Move models to device
        for model in self.classifiers.values():
            model.to(self.device)
            
        # Initialize metrics tracking
        self.metrics_history = {
            'weighted_loss': [],
            'unweighted_loss': [],
            'mean_weight': [],
            'r2': []
        }
    
    def calculate_sample_weights(self, uncertainties: torch.Tensor, alpha: float = 0.9, beta: float = 0.1) -> torch.Tensor:
        """Calculate smooth sample weights based on uncertainty.
        
        Args:
            uncertainties: MC-Dropout standard deviations
            alpha: Weight scaling factor (default 0.9)
            beta: Minimum weight floor (default 0.1)
        
        Returns:
            Weights in range [beta, alpha + beta]
        """
        # Normalize uncertainties to [0,1] if needed
        uncertainties = uncertainties / uncertainties.max()
        
        # Apply smooth weighting formula
        weights = (1 - uncertainties) * alpha + beta
        
        return weights
    
    def log_metrics(self, metrics: Dict[str, float]) -> None:
        """Log training metrics.
        
        Args:
            metrics: Dictionary of metric values
        """
        for key, value in metrics.items():
            if key in self.metrics_history:
                self.metrics_history[key].append(value)
    
    def _validate_inputs(self, text: str) -> bool:
        """Validate input text for edge cases.
        
        Args:
            text: Input text to validate
            
        Returns:
            Whether the input is valid
        """
        if not text or len(text.strip()) < 5:
            return False
        if text.isspace():
            return False
        return True
    
    def get_keyword_scores_batch(self, texts: List[str]) -> Dict[str, List[float]]:
        """Process keyword scores in batch for efficiency.
        
        Args:
            texts: List of input texts
            
        Returns:
            Dictionary of scores for each dimension
        """
        # Use the keyword matcher's implementation
        return self.keywords.get_keyword_scores_batch(texts)
    
    def _evaluate_with_beta(self, 
                          data_loader: DataLoader,
                          dimension: str,
                          beta: float) -> float:
        """Evaluate model performance with specific beta value.
        
        Args:
            data_loader: DataLoader for evaluation
            dimension: Response dimension
            beta: Weight floor value
            
        Returns:
            R² score
        """
        self.classifiers[dimension].eval()
        all_preds = []
        all_labels = []
        all_uncertainties = []
        
        with torch.no_grad():
            for batch in data_loader:
                input_ids = batch['input_ids'].to(self.device)
                attention_mask = batch['attention_mask'].to(self.device)
                labels = batch['label']
                
                # Get BERT predictions with uncertainty
                bert_outputs = self.classifiers[dimension](
                    input_ids, attention_mask, return_uncertainty=True
                )
                bert_preds = bert_outputs['prediction']
                uncertainties = bert_outputs['uncertainty']
                
                # Get keyword scores in batch
                keyword_scores = self.get_keyword_scores_batch(batch['text'])[dimension]
                keyword_scores = torch.tensor(keyword_scores, device=self.device)
                
                # Stack predictions
                weights = F.softmax(self.stacking_weights[dimension], dim=0)
                final_preds = weights[0] * bert_preds + weights[1] * keyword_scores
                
                # Store predictions, labels and uncertainties for analysis
                all_preds.extend(final_preds.cpu().numpy())
                all_labels.extend(labels.numpy())
                all_uncertainties.extend(uncertainties.cpu().numpy())
        
        # Calculate R² score on unweighted predictions
        r2 = r2_score(all_labels, all_preds)
        
        # Log uncertainty statistics for analysis
        mean_uncertainty = np.mean(all_uncertainties)
        logger.info(f"Mean uncertainty: {mean_uncertainty:.4f}")
        
        return r2
    
    def optimize_weight_floor(self, 
                            val_loader: DataLoader, 
                            dimension: str, 
                            beta_values: List[float] = [0.0, 0.05, 0.1, 0.2]) -> Tuple[float, float]:
        """Find optimal beta value for weight floor.
        
        Args:
            val_loader: Validation data loader
            dimension: Response dimension
            beta_values: List of beta values to try
            
        Returns:
            Tuple of (best_beta, best_r2)
        """
        best_beta = 0.1
        best_r2 = -float('inf')
        
        for beta in beta_values:
            # Evaluate with this beta
            r2 = self._evaluate_with_beta(val_loader, dimension, beta)
            
            logger.info(f"Beta {beta:.2f}: R² = {r2:.4f}")
            
            if r2 > best_r2:
                best_r2 = r2
                best_beta = beta
        
        return best_beta, best_r2
    
    def train(self, 
              texts: List[str],
              labels: Dict[str, List[float]],
              batch_size: int = 16,
              epochs: int = 3,
              learning_rate: float = 2e-5,
              alpha: float = 0.9,
              beta: float = 0.1,
              optimize_beta: bool = True) -> Dict[str, float]:
        """Train the classifiers.
        
        Args:
            texts: List of training texts
            labels: Dictionary of labels for each dimension
            batch_size: Batch size for training
            epochs: Number of training epochs
            learning_rate: Learning rate
            alpha: Weight scaling factor for uncertainty weighting
            beta: Minimum weight floor for uncertainty weighting
            optimize_beta: Whether to optimize beta value
            
        Returns:
            Dictionary of validation metrics
        """
        metrics = {}
        
        for dimension, dimension_labels in labels.items():
            logger.info(f"Training classifier for {dimension}")
            
            # Split data
            train_texts, val_texts, train_labels, val_labels = train_test_split(
                texts, dimension_labels, test_size=0.2, random_state=42
            )
            
            # Create datasets
            train_dataset = ResponseDataset(train_texts, train_labels, self.tokenizer)
            val_dataset = ResponseDataset(val_texts, val_labels, self.tokenizer)
            
            train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
            val_loader = DataLoader(val_dataset, batch_size=batch_size)
            
            # Optimize beta if requested
            if optimize_beta:
                logger.info("Optimizing beta value...")
                best_beta, best_r2 = self.optimize_weight_floor(val_loader, dimension)
                beta = best_beta
                logger.info(f"Selected beta = {beta:.2f} with R² = {best_r2:.4f}")
            
            # Initialize optimizer
            optimizer = torch.optim.AdamW(
                list(self.classifiers[dimension].parameters()) + 
                [self.stacking_weights[dimension]],  # Include stacking weights in optimization
                lr=learning_rate
            )
            
            # Training loop
            best_val_r2 = -float('inf')
            for epoch in range(epochs):
                self.classifiers[dimension].train()
                total_weighted_loss = 0
                total_unweighted_loss = 0
                total_bert_only_loss = 0
                total_combined_loss = 0
                
                for batch in train_loader:
                    input_ids = batch['input_ids'].to(self.device)
                    attention_mask = batch['attention_mask'].to(self.device)
                    labels = batch['label'].to(self.device)
                    
                    # Get BERT predictions with uncertainty
                    bert_outputs = self.classifiers[dimension](
                        input_ids, attention_mask, return_uncertainty=True
                    )
                    bert_preds = bert_outputs['prediction']
                    uncertainties = bert_outputs['uncertainty']
                    
                    # Get keyword scores in batch
                    keyword_scores = self.get_keyword_scores_batch(batch['text'])[dimension]
                    keyword_scores = torch.tensor(keyword_scores, device=self.device)
                    
                    # Stack predictions
                    weights = F.softmax(self.stacking_weights[dimension], dim=0)
                    final_preds = weights[0] * bert_preds + weights[1] * keyword_scores
                    
                    # Calculate sample weights based on uncertainty
                    sample_weights = self.calculate_sample_weights(uncertainties, alpha, beta)
                    
                    # Calculate losses
                    unweighted_loss = F.mse_loss(final_preds, labels)
                    weighted_loss = F.mse_loss(final_preds, labels, reduction='none')
                    weighted_loss = (weighted_loss * sample_weights).mean()
                    
                    # Calculate BERT-only and combined losses for comparison
                    bert_only_loss = F.mse_loss(bert_preds, labels)
                    combined_loss = F.mse_loss(final_preds, labels)
                    
                    # Log metrics
                    self.log_metrics({
                        'weighted_loss': weighted_loss.item(),
                        'unweighted_loss': unweighted_loss.item(),
                        'mean_weight': sample_weights.mean().item(),
                        'bert_only_loss': bert_only_loss.item(),
                        'combined_loss': combined_loss.item()
                    })
                    
                    # Backward pass
                    optimizer.zero_grad()
                    weighted_loss.backward()
                    optimizer.step()
                    
                    total_weighted_loss += weighted_loss.item()
                    total_unweighted_loss += unweighted_loss.item()
                    total_bert_only_loss += bert_only_loss.item()
                    total_combined_loss += combined_loss.item()
                
                # Log stacking weights after each epoch
                weights = F.softmax(self.stacking_weights[dimension], dim=0)
                logger.info(f"Epoch {epoch + 1}/{epochs} - {dimension} stacking weights:")
                logger.info(f"  BERT weight: {weights[0].item():.3f}")
                logger.info(f"  Keyword weight: {weights[1].item():.3f}")
                
                # Validation
                val_metrics = self._evaluate(val_loader, dimension)
                metrics[dimension] = val_metrics
                
                # Compare BERT-only vs combined performance
                bert_only_r2 = val_metrics.get('bert_only_r2', 0)
                combined_r2 = val_metrics.get('r2', 0)
                logger.info(f"Validation R² - BERT-only: {bert_only_r2:.4f}, Combined: {combined_r2:.4f}")
                
                # Save best model
                if val_metrics['r2'] > best_val_r2:
                    best_val_r2 = val_metrics['r2']
                    self._save_model(dimension)
                
                logger.info(f"Epoch {epoch + 1}/{epochs}")
                logger.info(f"  Weighted Loss: {total_weighted_loss / len(train_loader):.4f}")
                logger.info(f"  Unweighted Loss: {total_unweighted_loss / len(train_loader):.4f}")
                logger.info(f"  BERT-only Loss: {total_bert_only_loss / len(train_loader):.4f}")
                logger.info(f"  Combined Loss: {total_combined_loss / len(train_loader):.4f}")
                logger.info(f"  Val R²: {val_metrics['r2']:.4f}")
            
            # Calibrate model on validation set
            val_preds = []
            val_targets = []
            with torch.no_grad():
                for batch in val_loader:
                    input_ids = batch['input_ids'].to(self.device)
                    attention_mask = batch['attention_mask'].to(self.device)
                    labels = batch['label']
                    
                    outputs = self.classifiers[dimension](input_ids, attention_mask)
                    val_preds.extend(outputs['prediction'].cpu().numpy())
                    val_targets.extend(labels.numpy())
            
            self.classifiers[dimension].calibrate(
                np.array(val_preds),
                np.array(val_targets)
            )
        
        return metrics
    
    def _evaluate(self, 
                 data_loader: DataLoader,
                 dimension: str) -> Dict[str, float]:
        """Evaluate model performance.
        
        Args:
            data_loader: DataLoader for evaluation
            dimension: Response dimension
            
        Returns:
            Dictionary of metrics
        """
        self.classifiers[dimension].eval()
        all_preds = []
        all_labels = []
        all_bert_preds = []
        
        with torch.no_grad():
            for batch in data_loader:
                input_ids = batch['input_ids'].to(self.device)
                attention_mask = batch['attention_mask'].to(self.device)
                labels = batch['label']
                
                # Get BERT predictions
                bert_outputs = self.classifiers[dimension](
                    input_ids, attention_mask, return_uncertainty=True
                )
                bert_preds = bert_outputs['prediction']
                
                # Get keyword scores in batch
                keyword_scores = self.get_keyword_scores_batch(batch['text'])[dimension]
                keyword_scores = torch.tensor(keyword_scores, device=self.device)
                
                # Stack predictions
                weights = F.softmax(self.stacking_weights[dimension], dim=0)
                final_preds = weights[0] * bert_preds + weights[1] * keyword_scores
                
                all_preds.extend(final_preds.cpu().numpy())
                all_labels.extend(labels.numpy())
                all_bert_preds.extend(bert_preds.cpu().numpy())
        
        # Calculate metrics
        r2 = r2_score(all_labels, all_preds)
        bert_only_r2 = r2_score(all_labels, all_bert_preds)
        
        return {
            'r2': r2,
            'bert_only_r2': bert_only_r2
        }
    
    def _save_model(self, dimension: str) -> None:
        """Save model state.
        
        Args:
            dimension: Response dimension
        """
        save_path = self.model_dir / f"{dimension}.pt"
        torch.save({
            'model_state_dict': self.classifiers[dimension].state_dict(),
            'stacking_weights': self.stacking_weights[dimension].state_dict(),
            'is_calibrated': self.classifiers[dimension].is_calibrated,
            'calibration': self.classifiers[dimension].calibration if self.classifiers[dimension].is_calibrated else None
        }, save_path)
    
    def _load_model(self, dimension: str) -> None:
        """Load model state.
        
        Args:
            dimension: Response dimension
        """
        load_path = self.model_dir / f"{dimension}.pt"
        if load_path.exists():
            checkpoint = torch.load(load_path)
            self.classifiers[dimension].load_state_dict(checkpoint['model_state_dict'])
            self.stacking_weights[dimension].load_state_dict(checkpoint['stacking_weights'])
            
            if checkpoint.get('is_calibrated', False):
                self.classifiers[dimension].is_calibrated = True
                self.classifiers[dimension].calibration = checkpoint['calibration']
    
    def predict(self, 
                texts: Union[str, List[str]],
                use_keywords: bool = True,
                return_uncertainty: bool = False) -> Dict[str, Union[float, List[float]]]:
        """Make predictions.
        
        Args:
            texts: Input text(s)
            use_keywords: Whether to use keyword matching
            return_uncertainty: Whether to return uncertainty estimates
            
        Returns:
            Dictionary with predictions and optional uncertainty
        """
        if isinstance(texts, str):
            texts = [texts]
            
        # Validate inputs
        valid_texts = [text for text in texts if self._validate_inputs(text)]
        if not valid_texts:
            results = {
                dim: 0.5 if isinstance(texts, str) else [0.5] * len(texts)
                for dim in self.classifiers.keys()
            }
            if return_uncertainty:
                results['uncertainties'] = {
                    dim: 1.0 if isinstance(texts, str) else [1.0] * len(texts)
                    for dim in self.classifiers.keys()
                }
            return results
            
        results = {
            'activation_sedation': [],
            'emotional_blunting': [],
            'appetite_metabolic': []
        }
        
        if return_uncertainty:
            uncertainties = {
                'activation_sedation': [],
                'emotional_blunting': [],
                'appetite_metabolic': []
            }
        
        # Get keyword scores in batch
        if use_keywords:
            keyword_scores = self.get_keyword_scores_batch(valid_texts)
        
        # Process each text
        for i, text in enumerate(texts):
            if not self._validate_inputs(text):
                # Use default scores for invalid inputs
                for dim in results:
                    results[dim].append(0.5)
                    if return_uncertainty:
                        uncertainties[dim].append(1.0)
                continue
            
            # Get BERT predictions
            inputs = self.tokenizer(
                text,
                return_tensors="pt",
                padding=True,
                truncation=True
            ).to(self.device)
            
            for dimension in self.classifiers:
                # Get BERT predictions
                bert_outputs = self.classifiers[dimension](
                    inputs['input_ids'],
                    inputs['attention_mask'],
                    return_uncertainty=return_uncertainty
                )
                bert_pred = bert_outputs['prediction'].item()
                
                if return_uncertainty:
                    uncertainty = bert_outputs['uncertainty'].item()
                
                if use_keywords:
                    # Get keyword score
                    keyword_score = keyword_scores[dimension][i]
                    
                    # Stack predictions
                    weights = F.softmax(self.stacking_weights[dimension], dim=0)
                    final_pred = weights[0].item() * bert_pred + weights[1].item() * keyword_score
                else:
                    final_pred = bert_pred
                
                results[dimension].append(final_pred)
                if return_uncertainty:
                    uncertainties[dimension].append(uncertainty)
        
        if len(texts) == 1:
            # Return single predictions
            results = {k: v[0] for k, v in results.items()}
            if return_uncertainty:
                uncertainties = {k: v[0] for k, v in uncertainties.items()}
        
        # Always add uncertainties to results if requested
        if return_uncertainty:
            results['uncertainties'] = uncertainties
        
        return results

def process_timeline_data(timeline_file: str,
                         output_file: str,
                         batch_size: int = 32) -> None:
    """Process timeline data and add response dimension scores.
    
    Args:
        timeline_file: Path to timeline parquet file
        output_file: Path to output parquet file
        batch_size: Batch size for processing
    """
    # Load timeline data
    df = pd.read_parquet(timeline_file)
    
    # Initialize classifier
    classifier = ResponseDimensionClassifier()
    
    # Process in batches
    all_scores = []
    for i in range(0, len(df), batch_size):
        batch = df.iloc[i:i+batch_size]
        scores = classifier.predict(batch['post_text'].tolist())
        
        batch_scores = pd.DataFrame({
            'activation_sedation_score': scores['activation_sedation'],
            'emotional_blunting_restoration_score': scores['emotional_blunting'],
            'appetite_metabolic_score': scores['appetite_metabolic']
        })
        
        all_scores.append(batch_scores)
    
    # Combine scores with original data
    scores_df = pd.concat(all_scores, ignore_index=True)
    result_df = pd.concat([df, scores_df], axis=1)
    
    # Add confidence scores (placeholder for now)
    result_df['confidence_scores'] = [[0.8, 0.8, 0.8] for _ in range(len(result_df))]
    
    # Save results
    result_df.to_parquet(output_file)
    logger.info(f"Processed {len(df)} posts and saved to {output_file}")

if __name__ == "__main__":
    # Example usage
    timeline_file = "data/processed/user_timelines.parquet"
    output_file = "data/processed/annotated_responses.parquet"
    
    process_timeline_data(timeline_file, output_file) 
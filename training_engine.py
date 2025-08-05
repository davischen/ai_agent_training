# -*- coding: utf-8 -*-
"""
Training Engine for ATLAS Agent
Handles model training requests
"""

import logging
import time
import pandas as pd
import numpy as np
from typing import Dict, List, Any, Optional
from dataclasses import dataclass
from datetime import datetime
from sklearn.model_selection import train_test_split
from sentence_transformers import SentenceTransformer, InputExample, losses
from torch.utils.data import DataLoader
import torch

from .model_manager import ModelManager

logger = logging.getLogger(__name__)

@dataclass 
class TrainingRequest:
    """Request object for training operations"""
    request_id: str
    task_type: str
    priority: int = 1
    data_path: Optional[str] = None
    model_config: Dict[str, Any] = None
    training_params: Dict[str, Any] = None
    callback_url: Optional[str] = None
    created_at: datetime = None
    scheduled_at: Optional[datetime] = None
    
    def __post_init__(self):
        if self.created_at is None:
            self.created_at = datetime.now()
        if self.model_config is None:
            self.model_config = {}
        if self.training_params is None:
            self.training_params = {}

class DataGenerator:
    """Generates synthetic training data and augments existing datasets"""
    
    def __init__(self):
        self.phishing_patterns = [
            "urgent action required",
            "verify your account", 
            "click here immediately",
            "your account will be closed",
            "suspicious activity detected",
            "confirm your identity",
            "limited time offer",
            "act now or lose access",
            "security alert",
            "update payment information"
        ]
        
        self.safe_patterns = [
            "meeting scheduled",
            "project update",
            "weekly report",
            "team collaboration", 
            "document shared",
            "calendar invitation",
            "quarterly review",
            "system maintenance",
            "newsletter",
            "thank you for your purchase"
        ]
    
    def generate_synthetic_emails(self, num_samples: int = 100) -> pd.DataFrame:
        """Generate synthetic email dataset"""
        data = []
        
        for i in range(num_samples // 2):
            # Generate phishing emails
            pattern = np.random.choice(self.phishing_patterns)
            phishing_variations = [
                f"Dear user, {pattern}. Please visit our secure portal to complete verification.",
                f"URGENT: {pattern}. Click the link below to avoid account suspension.",
                f"Security notice: {pattern}. Immediate action required to protect your account.",
                f"Alert: {pattern}. Follow this link to update your information now."
            ]
            phishing_text = np.random.choice(phishing_variations)
            
            data.append({
                'Email Text': phishing_text,
                'Email Type': 'Phishing Email',
                'synthetic': True,
                'pattern_used': pattern
            })
            
            # Generate safe emails
            pattern = np.random.choice(self.safe_patterns)
            safe_variations = [
                f"Hello team, this is regarding {pattern}. Please find the details in the attachment.",
                f"Hi, I wanted to update you on the {pattern}. Let me know if you have questions.",
                f"Good morning, please note the {pattern} scheduled for next week.",
                f"Team update: The {pattern} has been completed successfully."
            ]
            safe_text = np.random.choice(safe_variations)
            
            data.append({
                'Email Text': safe_text,
                'Email Type': 'Safe Email',
                'synthetic': True,
                'pattern_used': pattern
            })
        
        df = pd.DataFrame(data)
        df['Label'] = df['Email Type'].map({'Safe Email': 0, 'Phishing Email': 1})
        return df
    
    def augment_dataset(self, original_df: pd.DataFrame, 
                       augmentation_factor: float = 0.2) -> pd.DataFrame:
        """Augment existing dataset with synthetic data"""
        num_synthetic = int(len(original_df) * augmentation_factor)
        synthetic_df = self.generate_synthetic_emails(num_synthetic)
        return pd.concat([original_df, synthetic_df], ignore_index=True)

class TrainingEngine:
    """Handles model training requests"""
    
    def __init__(self, model_manager: ModelManager, data_generator: DataGenerator = None):
        self.model_manager = model_manager
        self.data_generator = data_generator or DataGenerator()
        
    async def process_training(self, request: TrainingRequest) -> Dict[str, Any]:
        """Process training request"""
        try:
            # Load or generate training data
            if request.data_path:
                df = pd.read_csv(request.data_path)
                logger.info(f"Loaded training data from {request.data_path}: {len(df)} samples")
            else:
                df = self.data_generator.generate_synthetic_emails(1000)
                logger.info(f"Generated synthetic training data: {len(df)} samples")
            
            # Data preprocessing
            df = self._preprocess_data(df)
            
            # Model configuration
            model_name = request.model_config.get('base_model', 'all-MiniLM-L6-v2')
            epochs = request.training_params.get('epochs', 3)
            batch_size = request.training_params.get('batch_size', 16)
            learning_rate = request.training_params.get('learning_rate', 2e-5)
            
            # Load base model
            model = self.model_manager.load_model(model_name)
            logger.info(f"Loaded base model: {model_name}")
            
            # Prepare training data
            train_examples = self._prepare_training_examples(df)
            
            # Create data loader
            train_dataloader = DataLoader(train_examples, shuffle=True, batch_size=batch_size)
            
            # Setup training loss
            train_loss = losses.MultipleNegativesRankingLoss(model)
            
            # Train the model
            logger.info(f"Starting training with {len(train_examples)} examples for {epochs} epochs")
            
            model.fit(
                train_objectives=[(train_dataloader, train_loss)],
                epochs=epochs,
                warmup_steps=int(0.1 * len(train_dataloader)),
                show_progress_bar=True,
                output_path=None  # We'll save manually
            )
            
            # Save trained model
            trained_model_name = f"trained_{request.request_id}_{int(time.time())}"
            self.model_manager.save_model(model, trained_model_name)
            
            # Evaluate model performance
            performance_metrics = await self._evaluate_model(model, df)
            
            return {
                'model_name': trained_model_name,
                'training_samples': len(train_examples),
                'epochs': epochs,
                'batch_size': batch_size,
                'learning_rate': learning_rate,
                'performance_metrics': performance_metrics,
                'status': 'completed'
            }
            
        except Exception as e:
            logger.error(f"Training failed: {e}")
            raise
    
    def _preprocess_data(self, df: pd.DataFrame) -> pd.DataFrame:
        """Preprocess training data"""
        # Clean column names
        df.columns = df.columns.str.strip()
        
        # Create label column if it doesn't exist
        if 'Label' not in df.columns and 'Email Type' in df.columns:
            df['Label'] = df['Email Type'].map({'Safe Email': 0, 'Phishing Email': 1})
        
        # Drop rows with missing essential data
        df = df.dropna(subset=['Email Text', 'Label'])
        
        # Clean text data
        df['Email Text'] = df['Email Text'].astype(str).str.strip()
        
        # Remove empty texts
        df = df[df['Email Text'].str.len() > 0]
        
        logger.info(f"Preprocessed data: {len(df)} samples, "
                   f"{sum(df['Label'] == 1)} phishing, {sum(df['Label'] == 0)} safe")
        
        return df
    
    def _prepare_training_examples(self, df: pd.DataFrame) -> List[InputExample]:
        """Prepare training examples for sentence transformer"""
        examples = []
        
        # Split by labels
        phishing_texts = df[df['Label'] == 1]['Email Text'].tolist()
        safe_texts = df[df['Label'] == 0]['Email Text'].tolist()
        
        # Create positive pairs (same label)
        for texts in [phishing_texts, safe_texts]:
            for i, text1 in enumerate(texts):
                for j, text2 in enumerate(texts[i+1:], i+1):
                    if i != j:  # Different texts but same label
                        examples.append(InputExample(texts=[text1, text2], label=1.0))
        
        # Create negative pairs (different labels)
        import random
        for phishing_text in phishing_texts[:50]:  # Limit for performance
            safe_text = random.choice(safe_texts)
            examples.append(InputExample(texts=[phishing_text, safe_text], label=0.0))
        
        random.shuffle(examples)
        logger.info(f"Created {len(examples)} training examples")
        
        return examples
    
    async def _evaluate_model(self, model: SentenceTransformer, 
                            df: pd.DataFrame) -> Dict[str, float]:
        """Evaluate trained model performance"""
        try:
            # Split data for evaluation
            train_df, test_df = train_test_split(df, test_size=0.2, random_state=42, 
                                               stratify=df['Label'])
            
            # Generate embeddings for test set
            test_texts = test_df['Email Text'].tolist()
            test_labels = test_df['Label'].tolist()
            
            embeddings = model.encode(test_texts, convert_to_numpy=True)
            
            # Simple classification based on similarity
            # This is a placeholder - in practice you'd use a proper classifier
            predictions = []
            for embedding in embeddings:
                # Simplified prediction logic
                prob = np.sum(np.abs(embedding)) % 1.0
                pred = 1 if prob > 0.5 else 0
                predictions.append(pred)
            
            # Calculate metrics
            from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
            
            accuracy = accuracy_score(test_labels, predictions)
            precision = precision_score(test_labels, predictions, zero_division=0)
            recall = recall_score(test_labels, predictions, zero_division=0)
            f1 = f1_score(test_labels, predictions, zero_division=0)
            
            metrics = {
                'accuracy': accuracy,
                'precision': precision,
                'recall': recall,
                'f1_score': f1,
                'test_samples': len(test_labels)
            }
            
            logger.info(f"Model evaluation - Accuracy: {accuracy:.3f}, "
                       f"Precision: {precision:.3f}, Recall: {recall:.3f}, F1: {f1:.3f}")
            
            return metrics
            
        except Exception as e:
            logger.error(f"Model evaluation failed: {e}")
            return {'error': str(e)}
    
    def prepare_training_data_from_csv(self, csv_path: str) -> pd.DataFrame:
        """Load and prepare training data from CSV file"""
        try:
            df = pd.read_csv(csv_path)
            df = self._preprocess_data(df)
            return df
        except Exception as e:
            logger.error(f"Failed to load training data from {csv_path}: {e}")
            raise
    
    def get_training_statistics(self, df: pd.DataFrame) -> Dict[str, Any]:
        """Get statistics about training data"""
        if df.empty:
            return {}
        
        stats = {
            'total_samples': len(df),
            'phishing_samples': sum(df['Label'] == 1),
            'safe_samples': sum(df['Label'] == 0),
            'phishing_percentage': (sum(df['Label'] == 1) / len(df)) * 100,
            'average_text_length': df['Email Text'].str.len().mean(),
            'min_text_length': df['Email Text'].str.len().min(),
            'max_text_length': df['Email Text'].str.len().max()
        }
        
        return stats

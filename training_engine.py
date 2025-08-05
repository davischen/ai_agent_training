# -*- coding: utf-8 -*-
"""
Training Engine for ATLAS Agent
Unified with preprocess_phishing_data.py training methodology
"""

import logging
import time
import pandas as pd
import numpy as np
import random
from typing import Dict, List, Any, Optional
from dataclasses import dataclass
from datetime import datetime
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.metrics.pairwise import cosine_similarity
from sentence_transformers import SentenceTransformer, InputExample, losses
from torch.utils.data import DataLoader
import torch

# 同一層目錄導入
from model_manager import ModelManager

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
    """Generates synthetic training data (same as preprocess_phishing_data.py style)"""
    
    def __init__(self):
        self.phishing_patterns = [
            "urgent action required", "verify your account", "click here immediately",
            "your account will be closed", "suspicious activity detected",
            "confirm your identity", "limited time offer", "act now or lose access",
            "security alert", "update payment information"
        ]
        
        self.safe_patterns = [
            "meeting scheduled", "project update", "weekly report",
            "team collaboration", "document shared", "calendar invitation",
            "quarterly review", "system maintenance", "newsletter",
            "thank you for your purchase"
        ]
    
    def generate_synthetic_emails(self, num_samples: int = 100) -> pd.DataFrame:
        """Generate synthetic email dataset"""
        data = []
        
        for i in range(num_samples // 2):
            # Generate phishing emails
            pattern = np.random.choice(self.phishing_patterns)
            phishing_variations = [
                f"URGENT: {pattern} at secure-bank-verify.com",
                f"Dear user, {pattern}. Please visit our secure portal to complete verification.",
                f"SECURITY ALERT: {pattern}. Confirm your identity immediately.",
                f"Alert: {pattern}. Follow this link to update your information now."
            ]
            phishing_text = np.random.choice(phishing_variations)
            
            data.append({
                'Email Text': phishing_text,
                'Email Type': 'Phishing Email'
            })
            
            # Generate safe emails
            pattern = np.random.choice(self.safe_patterns)
            safe_variations = [
                f"Hi team, please find the {pattern} attached for your review.",
                f"Hello, this is regarding {pattern}. Please find the details in the attachment.",
                f"Good morning, please note the {pattern} scheduled for next week.",
                f"Team update: The {pattern} has been completed successfully."
            ]
            safe_text = np.random.choice(safe_variations)
            
            data.append({
                'Email Text': safe_text,
                'Email Type': 'Safe Email'
            })
        
        df = pd.DataFrame(data)
        df['Label'] = df['Email Type'].map({'Safe Email': 0, 'Phishing Email': 1})
        return df

class TrainingEngine:
    """Training Engine using the same methodology as preprocess_phishing_data.py"""
    
    def __init__(self, model_manager: ModelManager, data_generator: DataGenerator = None):
        self.model_manager = model_manager
        self.data_generator = data_generator or DataGenerator()
        
    async def process_training(self, request: TrainingRequest) -> Dict[str, Any]:
        """Process training request using unified methodology"""
        try:
            # Step 1: Load and preprocess data (same as preprocess_phishing_data.py)
            df = await self._load_and_preprocess_data(request)
            
            # Step 2: Split data
            train_df, test_df = self._split_data(df)
            
            # Step 3: Prepare training data (same method as preprocess_phishing_data.py)
            train_emails, test_emails, train_labels, test_labels = self._prepare_training_data(train_df, test_df)
            
            # Step 4: Create training examples (same as preprocess_phishing_data.py)
            train_examples = self._create_training_examples(train_emails, train_labels)
            
            # Step 5: Train model (same method as preprocess_phishing_data.py)
            model = await self._train_model(train_examples, request)
            
            # Step 6: Evaluate model (same method as preprocess_phishing_data.py)
            evaluation_results = await self._evaluate_model(model, test_emails, test_labels)
            
            # Step 7: Save model
            trained_model_name = f"trained_{request.request_id}_{int(time.time())}"
            self.model_manager.save_model(model, trained_model_name)
            
            return {
                'model_name': trained_model_name,
                'training_samples': len(train_examples),
                'test_samples': len(test_emails),
                'epochs': request.training_params.get('epochs', 5),
                'batch_size': request.training_params.get('batch_size', 16),
                'evaluation_results': evaluation_results,
                'status': 'completed',
                'method': 'unified_similarity_learning'
            }
            
        except Exception as e:
            logger.error(f"Training failed: {e}")
            raise
    
    async def _load_and_preprocess_data(self, request: TrainingRequest) -> pd.DataFrame:
        """Load and preprocess data (same as preprocess_phishing_data.py)"""
        
        if request.data_path:
            # Load from CSV
            df = pd.read_csv(request.data_path)
            logger.info(f"Loaded {len(df)} records from {request.data_path}")
        else:
            # Generate synthetic data
            df = self.data_generator.generate_synthetic_emails(1000)
            logger.info(f"Generated {len(df)} synthetic records")
        
        # Ensure correct column names and drop NaN values (same as preprocess_phishing_data.py)
        df.columns = df.columns.str.strip()
        df['Label'] = df['Email Type'].map({'Safe Email': 0, 'Phishing Email': 1})
        df['preprocessed_text_combined'] = df['Email Text']  # Keeping raw text
        df = df.rename(columns={'preprocessed_text_combined': 'text', 'Label': 'label'})

        # Drop rows where 'text' is NaN or contains only whitespace
        df = df.dropna(subset=['text', 'label'])

        # Apply stricter filtering (same as preprocess_phishing_data.py)
        # - Empty or whitespace-only texts
        # - Texts that contain the word "empty" (case insensitive)
        df = df[df['text'].astype(str).apply(lambda x: isinstance(x, str) and x.strip() != "" and "empty" not in x.lower())]

        logger.info(f"After filtering: {len(df)} records")
        logger.info(f"Label distribution: {df['label'].value_counts().to_dict()}")
        
        return df
    
    def _split_data(self, df: pd.DataFrame) -> tuple:
        """Split data (same as preprocess_phishing_data.py)"""
        train_df, test_df = train_test_split(df, test_size=0.2, random_state=42, stratify=df['label'])
        logger.info(f"Train Set: {len(train_df)} emails, Test Set: {len(test_df)} emails")
        return train_df, test_df
    
    def _prepare_training_data(self, train_df: pd.DataFrame, test_df: pd.DataFrame) -> tuple:
        """Prepare training data (same as preprocess_phishing_data.py)"""
        train_emails = train_df['text'].astype(str).tolist()
        test_emails = test_df['text'].astype(str).tolist()
        train_labels = train_df['label'].astype(int).tolist()
        test_labels = test_df['label'].astype(int).tolist()

        # Identify remaining emails that still contain "empty" (for debugging)
        remaining_empty_indices = [i for i, email in enumerate(test_emails) if "empty" in email.lower()]
        if remaining_empty_indices:
            logger.warning(f"Emails containing 'empty' still exist at indices: {remaining_empty_indices}")
        else:
            logger.info("No emails containing 'empty' found.")
        
        return train_emails, test_emails, train_labels, test_labels
    
    def _create_training_examples(self, train_emails: List[str], train_labels: List[int]) -> List[InputExample]:
        """Create training examples (same method as preprocess_phishing_data.py)"""
        train_examples = []
        for email, label in zip(train_emails, train_labels):
            # 找到相同類別的正樣本 (same as preprocess_phishing_data.py)
            positive_example = random.choice(
                [e for e, l in zip(train_emails, train_labels) if l == label and e != email]
            )  # Ensure positive example is from the same category
            train_examples.append(InputExample(texts=[email, positive_example]))

        logger.info(f"Created {len(train_examples)} training examples")
        return train_examples
    
    async def _train_model(self, train_examples: List[InputExample], request: TrainingRequest) -> SentenceTransformer:
        """Train model (same method as preprocess_phishing_data.py)"""
        
        # Get training parameters
        epochs = request.training_params.get('epochs', 5)
        batch_size = request.training_params.get('batch_size', 16)
        model_name = request.model_config.get('base_model', 'all-MiniLM-L6-v2')
        
        # Setup device (same as preprocess_phishing_data.py)
        device = "cuda" if torch.cuda.is_available() else "cpu"
        logger.info(f"Using device: {device}")
        
        # Load model (same as preprocess_phishing_data.py)
        model = self.model_manager.load_model(model_name).to(device)
        
        # Setup training (same as preprocess_phishing_data.py)
        train_dataloader = DataLoader(train_examples, shuffle=True, batch_size=batch_size)
        train_loss = losses.MultipleNegativesRankingLoss(model)

        logger.info(f"Starting training with {len(train_examples)} examples for {epochs} epochs")
        
        # Train model (same as preprocess_phishing_data.py)
        model.fit(
            train_objectives=[(train_dataloader, train_loss)],
            epochs=epochs,
            warmup_steps=int(0.1 * len(train_dataloader)),
            show_progress_bar=True,
            output_path=None  # We'll save manually using ModelManager
        )
        
        logger.info("Training completed")
        return model
    
    async def _evaluate_model(self, model: SentenceTransformer, test_emails: List[str], test_labels: List[int]) -> Dict[str, Any]:
        """Evaluate model (same method as preprocess_phishing_data.py)"""
        
        logger.info("Evaluating model...")
        
        # Generate embeddings (same as preprocess_phishing_data.py)
        test_embeddings = model.encode(test_emails, batch_size=32, convert_to_numpy=True, show_progress_bar=True)
        
        # Create query-to-document ranking (same as preprocess_phishing_data.py)
        query_id_to_ranked_doc_ids = {}
        for idx, query_embedding in enumerate(test_embeddings):
            similarities = cosine_similarity([query_embedding], test_embeddings)[0]
            ranked_indices = np.argsort(similarities)[::-1]  # Rank documents by similarity
            query_id_to_ranked_doc_ids[idx] = ranked_indices.tolist()

        logger.info(f"Example ranking for query 0 (within test set): {query_id_to_ranked_doc_ids[0][:10]}")
        
        # Calculate MAP (same as preprocess_phishing_data.py)
        map_score = self._mean_average_precision(test_labels, query_id_to_ranked_doc_ids)
        logger.info(f"Mean Average Precision (MAP): {map_score}")
        
        # Calculate predictions (same as preprocess_phishing_data.py)
        y_pred = [test_labels[query_id_to_ranked_doc_ids[i][0]] for i in range(len(test_labels))]  # Predict top-1 match
        
        # Generate classification report (same as preprocess_phishing_data.py)
        logger.info("Classification Report:")
        logger.info(f"\n{classification_report(test_labels, y_pred)}")
        
        # Confusion matrix (same as preprocess_phishing_data.py)
        conf_matrix = confusion_matrix(test_labels, y_pred)
        logger.info(f"Confusion Matrix:\n{conf_matrix}")
        
        # Calculate accuracy
        accuracy = accuracy_score(test_labels, y_pred)
        logger.info(f"Accuracy: {accuracy:.4f}")
        
        # Analyze misclassified examples (same as preprocess_phishing_data.py)
        misclassified_analysis = self._analyze_misclassified_examples(test_emails, test_labels, y_pred)
        
        return {
            'accuracy': accuracy,
            'map_score': map_score,
            'confusion_matrix': conf_matrix.tolist(),
            'classification_report': classification_report(test_labels, y_pred, output_dict=True),
            'predictions': y_pred,
            'misclassified_analysis': misclassified_analysis,
            'total_test_samples': len(test_labels)
        }
    
    def _mean_average_precision(self, true_labels: List[int], query_id_to_ranked_doc_ids: Dict[int, List[int]], top_k: int = 10) -> float:
        """Calculate Mean Average Precision (same as preprocess_phishing_data.py)"""
        
        def average_precision(relevant_docs, candidate_docs):
            """Compute the Average Precision (AP) for a single query."""
            y_true = [1 if doc_id in relevant_docs else 0 for doc_id in candidate_docs]
            precisions = [np.mean(y_true[:k+1]) for k in range(len(y_true)) if y_true[k]]
            return np.mean(precisions) if precisions else 0

        average_precisions = []
        for query_id, ranked_docs in query_id_to_ranked_doc_ids.items():
            relevant_docs = [i for i, label in enumerate(true_labels) if label == true_labels[query_id]]  # Find correct class
            ap = average_precision(relevant_docs, ranked_docs[:top_k])  # Compute AP for top_k results
            average_precisions.append(ap)

        return np.mean(average_precisions)
    
    def _analyze_misclassified_examples(self, test_emails: List[str], test_labels: List[int], y_pred: List[int], max_examples: int = 10) -> Dict[str, Any]:
        """Analyze misclassified examples (same as preprocess_phishing_data.py)"""
        
        misclassified_indices = [
            i for i in range(len(test_labels))
            if test_labels[i] != y_pred[i] and test_emails[i].strip()  # Exclude empty text
        ]
        
        misclassified_examples = []
        for idx in misclassified_indices[:max_examples]:  # Limit to max_examples
            true_label = "Safe Email" if test_labels[idx] == 0 else "Phishing Email"
            predicted_label = "Safe Email" if y_pred[idx] == 0 else "Phishing Email"

            misclassified_examples.append({
                'index': idx,
                'true_label': true_label,
                'predicted_label': predicted_label,
                'email_text': test_emails[idx][:200] + "..." if len(test_emails[idx]) > 200 else test_emails[idx]
            })
        
        logger.info(f"Found {len(misclassified_indices)} misclassified examples")
        
        return {
            'total_misclassified': len(misclassified_indices),
            'examples': misclassified_examples
        }
    
    def prepare_training_data_from_csv(self, csv_path: str) -> pd.DataFrame:
        """Load and prepare training data from CSV file"""
        try:
            df = pd.read_csv(csv_path)
            # Apply the same preprocessing as in the training pipeline
            request = TrainingRequest(
                request_id="temp",
                task_type="data_prep",
                data_path=csv_path
            )
            return asyncio.run(self._load_and_preprocess_data(request))
        except Exception as e:
            logger.error(f"Failed to load training data from {csv_path}: {e}")
            raise
    
    def get_training_statistics(self, df: pd.DataFrame) -> Dict[str, Any]:
        """Get statistics about training data"""
        if df.empty:
            return {}
        
        stats = {
            'total_samples': len(df),
            'phishing_samples': sum(df['label'] == 1) if 'label' in df.columns else 0,
            'safe_samples': sum(df['label'] == 0) if 'label' in df.columns else 0,
            'phishing_percentage': (sum(df['label'] == 1) / len(df)) * 100 if 'label' in df.columns else 0,
            'average_text_length': df['text'].str.len().mean() if 'text' in df.columns else 0,
            'min_text_length': df['text'].str.len().min() if 'text' in df.columns else 0,
            'max_text_length': df['text'].str.len().max() if 'text' in df.columns else 0
        }
        
        return stats

# 便利函數
def create_unified_training_engine(model_manager: ModelManager = None) -> TrainingEngine:
    """Create TrainingEngine with unified methodology"""
    if model_manager is None:
        from model_manager import ModelManager
        model_manager = ModelManager()
    return TrainingEngine(model_manager)

async def test_unified_training():
    """Test the unified training engine"""
    print("🧪 Testing Unified Training Engine...")
    
    try:
        from model_manager import ModelManager
        
        # Create components
        mm = ModelManager("./test_models")
        te = TrainingEngine(mm)
        
        # Create a test training request
        request = TrainingRequest(
            request_id="test_001",
            task_type="training",
            training_params={'epochs': 2, 'batch_size': 8},  # Small for testing
            model_config={'base_model': 'all-MiniLM-L6-v2'}
        )
        
        # Run training
        result = await te.process_training(request)
        
        print(f"✅ Training completed successfully!")
        print(f"   Model: {result['model_name']}")
        print(f"   Accuracy: {result['evaluation_results']['accuracy']:.4f}")
        print(f"   MAP Score: {result['evaluation_results']['map_score']:.4f}")
        print(f"   Method: {result['method']}")
        
        return True
        
    except Exception as e:
        print(f"❌ Test failed: {e}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    import asyncio
    asyncio.run(test_unified_training())

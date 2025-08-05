# -*- coding: utf-8 -*-
"""
Inference Engine for ATLAS Agent
Handles real-time inference requests
"""

import logging
import numpy as np
from typing import Dict, List, Any
from dataclasses import dataclass
from datetime import datetime
from .model_manager import ModelManager

logger = logging.getLogger(__name__)

@dataclass
class InferenceRequest:
    """Request object for inference operations"""
    request_id: str
    text_data: List[str]
    model_name: str = "default"
    batch_size: int = 32
    return_embeddings: bool = False
    threshold: float = 0.5
    created_at: datetime = None
    
    def __post_init__(self):
        if self.created_at is None:
            self.created_at = datetime.now()

class InferenceEngine:
    """Handles real-time inference requests"""
    
    def __init__(self, model_manager: ModelManager):
        self.model_manager = model_manager
        
    async def process_inference(self, request: InferenceRequest) -> Dict[str, Any]:
        """Process inference request"""
        try:
            model = self.model_manager.load_model(request.model_name)
            
            # Generate embeddings
            embeddings = model.encode(
                request.text_data,
                batch_size=request.batch_size,
                convert_to_numpy=True,
                show_progress_bar=False
            )
            
            # Perform classification (simplified binary classification)
            predictions = []
            confidences = []
            
            for embedding in embeddings:
                # Simplified classification logic - replace with your actual classification method
                confidence = self._calculate_phishing_probability(embedding)
                prediction = 1 if confidence > request.threshold else 0
                predictions.append(prediction)
                confidences.append(confidence)
            
            result = {
                'predictions': predictions,
                'confidences': confidences,
                'text_data': request.text_data,
                'threshold_used': request.threshold,
                'model_used': request.model_name
            }
            
            if request.return_embeddings:
                result['embeddings'] = embeddings.tolist()
            
            return result
            
        except Exception as e:
            logger.error(f"Inference failed: {e}")
            raise
    
    def _calculate_phishing_probability(self, embedding: np.ndarray) -> float:
        """
        Calculate phishing probability from embedding
        This is a simplified placeholder - in real implementation,
        you would use a trained classifier
        """
        # Placeholder logic - replace with actual trained model
        # For now, using a simple heuristic based on embedding features
        feature_sum = np.sum(np.abs(embedding))
        normalized_score = (feature_sum % 1.0)  # Simple normalization
        
        # Add some randomness for demonstration
        noise = np.random.normal(0, 0.1)
        probability = max(0.0, min(1.0, normalized_score + noise))
        
        return probability
    
    def batch_inference(self, text_list: List[str], 
                       model_name: str = "default",
                       batch_size: int = 32) -> Dict[str, Any]:
        """Synchronous batch inference for multiple texts"""
        try:
            model = self.model_manager.load_model(model_name)
            
            # Process in batches
            all_predictions = []
            all_confidences = []
            
            for i in range(0, len(text_list), batch_size):
                batch_texts = text_list[i:i + batch_size]
                
                # Generate embeddings for this batch
                embeddings = model.encode(
                    batch_texts,
                    convert_to_numpy=True,
                    show_progress_bar=False
                )
                
                # Get predictions for this batch
                for embedding in embeddings:
                    confidence = self._calculate_phishing_probability(embedding)
                    prediction = 1 if confidence > 0.5 else 0
                    all_predictions.append(prediction)
                    all_confidences.append(confidence)
            
            return {
                'predictions': all_predictions,
                'confidences': all_confidences,
                'total_processed': len(text_list),
                'model_used': model_name
            }
            
        except Exception as e:
            logger.error(f"Batch inference failed: {e}")
            raise
    
    def get_model_predictions_summary(self, predictions: List[int], 
                                    confidences: List[float]) -> Dict[str, Any]:
        """Generate summary statistics for predictions"""
        if not predictions:
            return {}
        
        phishing_count = sum(predictions)
        safe_count = len(predictions) - phishing_count
        
        avg_confidence = np.mean(confidences)
        max_confidence = np.max(confidences)
        min_confidence = np.min(confidences)
        
        high_confidence_threshold = 0.8
        high_confidence_count = sum(1 for c in confidences if c > high_confidence_threshold)
        
        return {
            'total_emails': len(predictions),
            'phishing_detected': phishing_count,
            'safe_emails': safe_count,
            'phishing_percentage': (phishing_count / len(predictions)) * 100,
            'average_confidence': avg_confidence,
            'max_confidence': max_confidence,
            'min_confidence': min_confidence,
            'high_confidence_predictions': high_confidence_count,
            'high_confidence_percentage': (high_confidence_count / len(predictions)) * 100
        }

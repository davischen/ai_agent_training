# -*- coding: utf-8 -*-
"""
PII Recognition Inference Engine for ATLAS System
Real-time PII detection and entity recognition
"""

import torch
import numpy as np
from transformers import AutoTokenizer, AutoModelForTokenClassification, pipeline
import logging
import json
import os
from typing import Dict, List, Any, Optional, Tuple, Union
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path

logger = logging.getLogger(__name__)

@dataclass
class PIIEntity:
    """PII Entity detection result"""
    text: str
    label: str
    start: int
    end: int
    confidence: float
    entity_type: str

@dataclass
class PIIInferenceRequest:
    """Request object for PII inference operations"""
    request_id: str
    text_data: Union[str, List[str]]
    return_entities: bool = True
    return_masked_text: bool = False
    mask_token: str = "[REDACTED]"
    confidence_threshold: float = 0.5
    batch_size: int = 32
    created_at: datetime = None
    
    def __post_init__(self):
        if self.created_at is None:
            self.created_at = datetime.now()

@dataclass
class PIIInferenceResult:
    """Result object for PII inference"""
    request_id: str
    original_texts: List[str]
    entities: List[List[PIIEntity]]
    masked_texts: Optional[List[str]] = None
    processing_time: float = 0.0
    model_used: str = ""
    confidence_threshold: float = 0.5

class PIIInferenceEngine:
    """PII Recognition Inference Engine"""
    
    def __init__(self, 
                 model_path: str,
                 device: str = None,
                 batch_size: int = 32,
                 confidence_threshold: float = 0.5):
        """
        Initialize PII Inference Engine
        
        Args:
            model_path: Path to trained model
            device: Device to run inference on ('cpu', 'cuda', or None for auto)
            batch_size: Default batch size for inference
            confidence_threshold: Default confidence threshold
        """
        self.model_path = model_path
        self.batch_size = batch_size
        self.confidence_threshold = confidence_threshold
        
        # Setup device
        if device is None:
            self.device = "cuda" if torch.cuda.is_available() else "cpu"
        else:
            self.device = device
        
        logger.info(f"Using device: {self.device}")
        
        # Initialize components
        self.tokenizer = None
        self.model = None
        self.pipeline = None
        self.label_mapping = {}
        self.entity_types = set()
        
        # Load model and configuration
        self._load_model()
        self._load_config()
    
    def _load_model(self):
        """Load trained model and tokenizer"""
        try:
            logger.info(f"Loading model from: {self.model_path}")
            
            # Load tokenizer
            self.tokenizer = AutoTokenizer.from_pretrained(self.model_path)
            
            # Load model
            self.model = AutoModelForTokenClassification.from_pretrained(self.model_path)
            self.model.to(self.device)
            self.model.eval()
            
            # Create pipeline for easy inference
            self.pipeline = pipeline(
                "ner",
                model=self.model,
                tokenizer=self.tokenizer,
                device=0 if self.device == "cuda" else -1,
                aggregation_strategy="simple"
            )
            
            logger.info("Model loaded successfully")
            
        except Exception as e:
            logger.error(f"Failed to load model: {e}")
            raise
    
    def _load_config(self):
        """Load model configuration"""
        try:
            config_path = os.path.join(self.model_path, 'training_config.json')
            if os.path.exists(config_path):
                with open(config_path, 'r') as f:
                    config = json.load(f)
                
                self.label_mapping = config.get('id2label', {})
                # Convert string keys to int
                self.label_mapping = {int(k): v for k, v in self.label_mapping.items()}
                
                # Extract entity types from labels
                for label in self.label_mapping.values():
                    if label != 'O':  # Ignore 'O' (Outside) label
                        # Extract entity type from BIO format (e.g., B-PERSON -> PERSON)
                        entity_type = label.split('-')[-1] if '-' in label else label
                        self.entity_types.add(entity_type)
                
                logger.info(f"Loaded {len(self.label_mapping)} labels")
                logger.info(f"Entity types: {sorted(self.entity_types)}")
            else:
                logger.warning("No training config found, using default settings")
                
        except Exception as e:
            logger.warning(f"Failed to load config: {e}")
    
    def detect_pii_single(self, text: str, confidence_threshold: float = None) -> List[PIIEntity]:
        """
        Detect PII entities in a single text
        
        Args:
            text: Input text
            confidence_threshold: Confidence threshold (uses default if None)
            
        Returns:
            List of detected PII entities
        """
        if confidence_threshold is None:
            confidence_threshold = self.confidence_threshold
        
        try:
            # Run inference
            results = self.pipeline(text)
            
            # Convert to PIIEntity objects
            entities = []
            for result in results:
                if result['score'] >= confidence_threshold:
                    entity = PIIEntity(
                        text=result['word'],
                        label=result['entity_group'],
                        start=result['start'],
                        end=result['end'],
                        confidence=result['score'],
                        entity_type=result['entity_group']
                    )
                    entities.append(entity)
            
            return entities
            
        except Exception as e:
            logger.error(f"Error in PII detection: {e}")
            return []
    
    def detect_pii_batch(self, texts: List[str], confidence_threshold: float = None) -> List[List[PIIEntity]]:
        """
        Detect PII entities in multiple texts
        
        Args:
            texts: List of input texts
            confidence_threshold: Confidence threshold (uses default if None)
            
        Returns:
            List of lists of detected PII entities
        """
        if confidence_threshold is None:
            confidence_threshold = self.confidence_threshold
        
        try:
            all_entities = []
            
            # Process in batches
            for i in range(0, len(texts), self.batch_size):
                batch_texts = texts[i:i + self.batch_size]
                
                for text in batch_texts:
                    entities = self.detect_pii_single(text, confidence_threshold)
                    all_entities.append(entities)
            
            return all_entities
            
        except Exception as e:
            logger.error(f"Error in batch PII detection: {e}")
            return [[] for _ in texts]
    
    def mask_pii_text(self, text: str, entities: List[PIIEntity], mask_token: str = "[REDACTED]") -> str:
        """
        Mask PII entities in text
        
        Args:
            text: Original text
            entities: List of PII entities to mask
            mask_token: Token to replace PII with
            
        Returns:
            Text with PII entities masked
        """
        if not entities:
            return text
        
        # Sort entities by start position (reverse order for replacement)
        sorted_entities = sorted(entities, key=lambda x: x.start, reverse=True)
        
        masked_text = text
        for entity in sorted_entities:
            masked_text = (
                masked_text[:entity.start] + 
                mask_token + 
                masked_text[entity.end:]
            )
        
        return masked_text
    
    def get_entity_statistics(self, entities: List[List[PIIEntity]]) -> Dict[str, Any]:
        """
        Get statistics about detected entities
        
        Args:
            entities: List of entity lists from batch processing
            
        Returns:
            Statistics dictionary
        """
        total_entities = sum(len(entity_list) for entity_list in entities)
        entity_counts = {}
        confidence_scores = []
        
        for entity_list in entities:
            for entity in entity_list:
                entity_type = entity.entity_type
                entity_counts[entity_type] = entity_counts.get(entity_type, 0) + 1
                confidence_scores.append(entity.confidence)
        
        avg_confidence = np.mean(confidence_scores) if confidence_scores else 0.0
        max_confidence = np.max(confidence_scores) if confidence_scores else 0.0
        min_confidence = np.min(confidence_scores) if confidence_scores else 0.0
        
        return {
            'total_entities': total_entities,
            'entity_types_detected': len(entity_counts),
            'entity_type_counts': entity_counts,
            'average_confidence': avg_confidence,
            'max_confidence': max_confidence,
            'min_confidence': min_confidence,
            'texts_processed': len(entities),
            'texts_with_pii': sum(1 for entity_list in entities if entity_list)
        }
    
    async def process_inference_request(self, request: PIIInferenceRequest) -> PIIInferenceResult:
        """
        Process PII inference request
        
        Args:
            request: PII inference request
            
        Returns:
            PII inference result
        """
        start_time = datetime.now()
        
        try:
            # Handle single text or list of texts
            if isinstance(request.text_data, str):
                texts = [request.text_data]
            else:
                texts = request.text_data
            
            # Detect entities
            entities = self.detect_pii_batch(texts, request.confidence_threshold)
            
            # Create masked texts if requested
            masked_texts = None
            if request.return_masked_text:
                masked_texts = []
                for text, entity_list in zip(texts, entities):
                    masked_text = self.mask_pii_text(text, entity_list, request.mask_token)
                    masked_texts.append(masked_text)
            
            # Calculate processing time
            processing_time = (datetime.now() - start_time).total_seconds()
            
            # Create result
            result = PIIInferenceResult(
                request_id=request.request_id,
                original_texts=texts,
                entities=entities if request.return_entities else None,
                masked_texts=masked_texts,
                processing_time=processing_time,
                model_used=self.model_path,
                confidence_threshold=request.confidence_threshold
            )
            
            logger.info(f"Processed request {request.request_id} in {processing_time:.2f}s")
            return result
            
        except Exception as e:
            logger.error(f"Error processing inference request: {e}")
            raise
    
    def quick_detect(self, text: str, return_masked: bool = False, mask_token: str = "[REDACTED]") -> Dict[str, Any]:
        """
        Quick PII detection for single text
        
        Args:
            text: Input text
            return_masked: Whether to return masked text
            mask_token: Token for masking
            
        Returns:
            Detection results
        """
        entities = self.detect_pii_single(text)
        
        result = {
            'text': text,
            'entities_found': len(entities),
            'entities': [
                {
                    'text': entity.text,
                    'type': entity.entity_type,
                    'start': entity.start,
                    'end': entity.end,
                    'confidence': entity.confidence
                }
                for entity in entities
            ]
        }
        
        if return_masked:
            result['masked_text'] = self.mask_pii_text(text, entities, mask_token)
        
        return result
    
    def analyze_text_privacy_risk(self, text: str) -> Dict[str, Any]:
        """
        Analyze privacy risk level of text based on PII content
        
        Args:
            text: Input text
            
        Returns:
            Privacy risk analysis
        """
        entities = self.detect_pii_single(text)
        
        # Define risk levels for different entity types
        high_risk_types = {'PERSON', 'EMAIL', 'PHONE', 'SSN', 'CREDIT_CARD'}
        medium_risk_types = {'ADDRESS', 'DATE_OF_BIRTH', 'ORGANIZATION'}
        low_risk_types = {'LOCATION', 'DATE', 'TIME'}
        
        high_risk_count = sum(1 for entity in entities if entity.entity_type in high_risk_types)
        medium_risk_count = sum(1 for entity in entities if entity.entity_type in medium_risk_types)
        low_risk_count = sum(1 for entity in entities if entity.entity_type in low_risk_types)
        
        # Calculate risk score (0-100)
        risk_score = min(100, (high_risk_count * 30) + (medium_risk_count * 15) + (low_risk_count * 5))
        
        # Determine risk level
        if risk_score >= 70:
            risk_level = "HIGH"
        elif risk_score >= 30:
            risk_level = "MEDIUM"
        elif risk_score > 0:
            risk_level = "LOW"
        else:
            risk_level = "MINIMAL"
        
        return {
            'risk_level': risk_level,
            'risk_score': risk_score,
            'total_pii_entities': len(entities),
            'high_risk_entities': high_risk_count,
            'medium_risk_entities': medium_risk_count,
            'low_risk_entities': low_risk_count,
            'recommendations': self._get_privacy_recommendations(risk_level, entities)
        }
    
    def _get_privacy_recommendations(self, risk_level: str, entities: List[PIIEntity]) -> List[str]:
        """Get privacy recommendations based on risk level"""
        recommendations = []
        
        if risk_level == "HIGH":
            recommendations.extend([
                "Immediately review and mask sensitive information",
                "Consider removing or encrypting this content",
                "Implement additional access controls"
            ])
        elif risk_level == "MEDIUM":
            recommendations.extend([
                "Review content for sensitivity",
                "Consider partial masking of identified entities",
                "Monitor access to this content"
            ])
        elif risk_level == "LOW":
            recommendations.extend([
                "Monitor for additional PII disclosure",
                "Consider basic privacy measures"
            ])
        
        # Add specific recommendations based on entity types
        entity_types = {entity.entity_type for entity in entities}
        
        if 'EMAIL' in entity_types:
            recommendations.append("Email addresses detected - consider masking")
        if 'PHONE' in entity_types:
            recommendations.append("Phone numbers detected - verify if disclosure is necessary")
        if 'PERSON' in entity_types:
            recommendations.append("Personal names detected - ensure compliance with data protection")
        
        return recommendations
    
    def get_model_info(self) -> Dict[str, Any]:
        """Get information about the loaded model"""
        return {
            'model_path': self.model_path,
            'device': self.device,
            'entity_types': list(self.entity_types),
            'num_labels': len(self.label_mapping),
            'default_confidence_threshold': self.confidence_threshold,
            'default_batch_size': self.batch_size,
            'model_loaded': self.model is not None
        }

# Utility functions
def create_pii_inference_engine(model_path: str, 
                               device: str = None, 
                               confidence_threshold: float = 0.5) -> PIIInferenceEngine:
    """Create PII inference engine instance"""
    return PIIInferenceEngine(model_path, device, confidence_threshold=confidence_threshold)

def quick_pii_detection(model_path: str, text: str, confidence_threshold: float = 0.5) -> Dict[str, Any]:
    """Quick PII detection for single text"""
    engine = create_pii_inference_engine(model_path, confidence_threshold=confidence_threshold)
    return engine.quick_detect(text)

# Test function
def test_pii_inference_engine():
    """Test function for PII inference engine"""
    print("🧪 Testing PII Inference Engine...")
    
    # Sample texts with PII
    test_texts = [
        "Hi, my name is John Smith and my email is john.smith@example.com",
        "Please contact me at +1-555-123-4567 or visit 123 Main St, New York",
        "My social security number is 123-45-6789 and DOB is 01/15/1990"
    ]
    
    try:
        # Note: This would require an actual trained model
        print("Note: This test requires a trained PII model")
        print("Sample texts to test:")
        for i, text in enumerate(test_texts, 1):
            print(f"  {i}. {text}")
        
        return True
        
    except Exception as e:
        print(f"❌ Test failed: {e}")
        return False

if __name__ == "__main__":
    test_pii_inference_engine()
# -*- coding: utf-8 -*-
"""
Model Manager for ATLAS Agent
Handles model loading, saving, and lifecycle management
"""

import os
import logging
from typing import Dict, Optional
from sentence_transformers import SentenceTransformer

logger = logging.getLogger(__name__)

class ModelManager:
    """Manages AI models for training and inference"""
    
    def __init__(self, model_dir: str = "models"):
        self.model_dir = model_dir
        self.models = {}
        self.default_model_name = "all-MiniLM-L6-v2"
        os.makedirs(model_dir, exist_ok=True)
        
    def load_model(self, model_name: str) -> SentenceTransformer:
        """Load a model into memory"""
        if model_name not in self.models:
            try:
                model_path = os.path.join(self.model_dir, model_name)
                if os.path.exists(model_path):
                    self.models[model_name] = SentenceTransformer(model_path)
                else:
                    self.models[model_name] = SentenceTransformer(model_name)
                logger.info(f"Loaded model: {model_name}")
            except Exception as e:
                logger.error(f"Failed to load model {model_name}: {e}")
                raise
        return self.models[model_name]
    
    def save_model(self, model: SentenceTransformer, model_name: str):
        """Save a trained model"""
        save_path = os.path.join(self.model_dir, model_name)
        model.save(save_path)
        logger.info(f"Saved model to: {save_path}")
    
    def list_models(self) -> Dict[str, str]:
        """List all available models"""
        models_info = {}
        
        # Check local models
        if os.path.exists(self.model_dir):
            for item in os.listdir(self.model_dir):
                item_path = os.path.join(self.model_dir, item)
                if os.path.isdir(item_path):
                    models_info[item] = "local"
        
        # Check loaded models
        for model_name in self.models.keys():
            if model_name not in models_info:
                models_info[model_name] = "loaded"
        
        return models_info
    
    def unload_model(self, model_name: str) -> bool:
        """Unload a model from memory"""
        if model_name in self.models:
            del self.models[model_name]
            logger.info(f"Unloaded model: {model_name}")
            return True
        return False
    
    def get_model_info(self, model_name: str) -> Optional[Dict]:
        """Get information about a model"""
        if model_name in self.models:
            model = self.models[model_name]
            return {
                "name": model_name,
                "max_seq_length": getattr(model, 'max_seq_length', None),
                "device": str(model.device) if hasattr(model, 'device') else None,
                "status": "loaded"
            }
        return None

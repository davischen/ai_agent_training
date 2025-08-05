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
    
    def __init__(self, model_dir: str = "models"):  # 修正：正確的 __init__ 格式
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
                    logger.info(f"Loading model from local path: {model_path}")
                    self.models[model_name] = SentenceTransformer(model_path)
                else:
                    logger.info(f"Loading model from HuggingFace: {model_name}")
                    self.models[model_name] = SentenceTransformer(model_name)
                logger.info(f"Successfully loaded model: {model_name}")
            except Exception as e:
                logger.error(f"Failed to load model {model_name}: {e}")
                raise
        return self.models[model_name]
    
    def save_model(self, model: SentenceTransformer, model_name: str):
        """Save a trained model"""
        save_path = os.path.join(self.model_dir, model_name)
        try:
            model.save(save_path)
            logger.info(f"Saved model to: {save_path}")
        except Exception as e:
            logger.error(f"Failed to save model {model_name}: {e}")
            raise
    
    def list_models(self) -> Dict[str, str]:
        """List all available models"""
        models_info = {}
        
        # Check local models
        if os.path.exists(self.model_dir):
            for item in os.listdir(self.model_dir):
                item_path = os.path.join(self.model_dir, item)
                if os.path.isdir(item_path):
                    # 檢查是否是有效的模型目錄
                    if any(f.endswith('.json') for f in os.listdir(item_path)):
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
        else:
            logger.warning(f"Model {model_name} not found in loaded models")
            return False
    
    def get_model_info(self, model_name: str) -> Optional[Dict]:
        """Get information about a model"""
        if model_name in self.models:
            model = self.models[model_name]
            return {
                "name": model_name,
                "max_seq_length": getattr(model, 'max_seq_length', None),
                "device": str(model.device) if hasattr(model, 'device') else None,
                "status": "loaded",
                "model_type": type(model).__name__
            }
        else:
            # 檢查是否是本地模型
            model_path = os.path.join(self.model_dir, model_name)
            if os.path.exists(model_path):
                return {
                    "name": model_name,
                    "status": "available_local",
                    "path": model_path
                }
        return None
    
    def get_loaded_models(self) -> Dict[str, str]:
        """Get currently loaded models"""
        return {name: type(model).__name__ for name, model in self.models.items()}
    
    def clear_all_models(self):
        """Clear all loaded models from memory"""
        model_count = len(self.models)
        self.models.clear()
        logger.info(f"Cleared {model_count} models from memory")
    
    def model_exists(self, model_name: str) -> bool:
        """Check if a model exists (locally or can be downloaded)"""
        # 檢查是否已載入
        if model_name in self.models:
            return True
        
        # 檢查本地是否存在
        model_path = os.path.join(self.model_dir, model_name)
        if os.path.exists(model_path):
            return True
        
        # 對於標準模型名稱，假設可以從 HuggingFace 下載
        standard_models = [
            "all-MiniLM-L6-v2", "all-mpnet-base-v2", "all-distilroberta-v1",
            "all-MiniLM-L12-v2", "paraphrase-MiniLM-L6-v2"
        ]
        
        return model_name in standard_models

# 新增：便利函數
def create_model_manager(model_dir: str = "models") -> ModelManager:
    """Create and return a ModelManager instance"""
    return ModelManager(model_dir)

def test_model_manager():
    """Test function for ModelManager"""
    print("🧪 Testing ModelManager...")
    
    try:
        # 創建模型管理器
        mm = ModelManager("./test_models")
        print("✅ ModelManager created")
        
        # 測試載入模型
        model = mm.load_model("all-MiniLM-L6-v2")
        print("✅ Model loaded successfully")
        
        # 測試模型資訊
        info = mm.get_model_info("all-MiniLM-L6-v2")
        print(f"✅ Model info: {info}")
        
        # 測試列出模型
        models = mm.list_models()
        print(f"✅ Available models: {models}")
        
        # 測試模型存在檢查
        exists = mm.model_exists("all-MiniLM-L6-v2")
        print(f"✅ Model exists: {exists}")
        
        return True
        
    except Exception as e:
        print(f"❌ Test failed: {e}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    # Run test when file is executed directly
    test_model_manager()

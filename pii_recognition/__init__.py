# -*- coding: utf-8 -*-
"""
PII Recognition Module
Personal Identifiable Information detection and management
"""

try:
    from .pii_trainer import PIITrainer, create_pii_trainer, quick_train_pii
    PII_TRAINER_AVAILABLE = True
except ImportError:
    PII_TRAINER_AVAILABLE = False

try:
    from .pii_inference import PIIInferenceEngine, create_pii_inference_engine, quick_pii_detection
    PII_INFERENCE_AVAILABLE = True
except ImportError:
    PII_INFERENCE_AVAILABLE = False

try:
    from .pii_config import PIIConfig, PIIConfigManager, PIIConfigs
    PII_CONFIG_AVAILABLE = True
except ImportError:
    PII_CONFIG_AVAILABLE = False

__version__ = "1.0.0"
__all__ = [
    # Training
    "PIITrainer",
    "create_pii_trainer", 
    "quick_train_pii",
    
    # Inference
    "PIIInferenceEngine",
    "create_pii_inference_engine",
    "quick_pii_detection",
    
    # Configuration
    "PIIConfig",
    "PIIConfigManager", 
    "PIIConfigs"
]
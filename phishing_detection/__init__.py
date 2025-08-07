# -*- coding: utf-8 -*-
"""
Phishing Detection Module
Specialized phishing email detection components
"""

# Import preprocessing utilities
try:
    from .preprocess_phishing_data import (
        preprocess_phishing_dataset,
        create_training_examples,
        prepare_model_training,
        train_model,
        evaluate_model,
        complete_pipeline
    )
    PREPROCESS_AVAILABLE = True
except ImportError:
    PREPROCESS_AVAILABLE = False

try:
    from .simpleDataPreprocessor import (
        SimpleDataPreprocessor,
        preprocess_phishing_data
    )
    SIMPLE_PREPROCESSOR_AVAILABLE = True
except ImportError:
    SIMPLE_PREPROCESSOR_AVAILABLE = False

__version__ = "1.0.0"
__all__ = [
    "preprocess_phishing_dataset",
    "create_training_examples", 
    "complete_pipeline",
    "SimpleDataPreprocessor",
    "preprocess_phishing_data"
]
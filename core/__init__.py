# -*- coding: utf-8 -*-
"""
ATLAS Core Module
Main components for the ATLAS training system
"""

from .ai_training_agent import AITrainingAgent
from .model_manager import ModelManager
from .inference_engine import InferenceEngine
from .training_engine import TrainingEngine
from .task_scheduler import TaskScheduler

__version__ = "1.0.0"
__all__ = [
    "AITrainingAgent",
    "ModelManager", 
    "InferenceEngine",
    "TrainingEngine",
    "TaskScheduler"
]
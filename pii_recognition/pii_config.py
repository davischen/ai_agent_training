# -*- coding: utf-8 -*-
"""
PII Recognition Configuration for ATLAS System
Configuration management and settings for PII recognition module
"""

import os
import json
import yaml
from dataclasses import dataclass, asdict, field
from typing import Dict, List, Any, Optional
from pathlib import Path
import logging

logger = logging.getLogger(__name__)

@dataclass
class PIIModelConfig:
    """Configuration for PII model"""
    model_name: str = "bert-base-cased"
    model_path: str = "./models/pii"
    tokenizer_name: str = None  # Uses model_name if None
    max_length: int = 512
    device: str = "auto"  # 'cpu', 'cuda', or 'auto'
    use_fast_tokenizer: bool = True
    
    def __post_init__(self):
        if self.tokenizer_name is None:
            self.tokenizer_name = self.model_name

@dataclass
class PIITrainingConfig:
    """Configuration for PII training"""
    dataset_name: str = "ai4privacy/pii-masking-200k"
    output_dir: str = "./models/pii-trained"
    epochs: int = 3
    batch_size: int = 16
    learning_rate: float = 2e-5
    weight_decay: float = 0.01
    warmup_steps: int = 500
    save_steps: int = 500
    eval_steps: int = 500
    save_total_limit: int = 3
    evaluation_strategy: str = "steps"
    metric_for_best_model: str = "f1"
    load_best_model_at_end: bool = True
    greater_is_better: bool = True
    dataloader_num_workers: int = 2
    seed: int = 42
    max_samples: Optional[int] = None  # For testing with limited data
    test_size: float = 0.1

@dataclass
class PIIInferenceConfig:
    """Configuration for PII inference"""
    confidence_threshold: float = 0.5
    batch_size: int = 32
    return_entities: bool = True
    return_masked_text: bool = False
    mask_token: str = "[REDACTED]"
    aggregation_strategy: str = "simple"  # 'none', 'simple', 'first', 'average', 'max'
    
@dataclass
class PIIEntityConfig:
    """Configuration for PII entity types"""
    entity_types: List[str] = field(default_factory=lambda: [
        "PERSON",
        "EMAIL", 
        "PHONE",
        "ADDRESS",
        "SSN",
        "CREDIT_CARD",
        "DATE_OF_BIRTH",
        "ORGANIZATION",
        "LOCATION",
        "DATE",
        "TIME",
        "IP_ADDRESS",
        "URL"
    ])
    
    # Risk levels for different entity types
    high_risk_entities: List[str] = field(default_factory=lambda: [
        "PERSON", "EMAIL", "PHONE", "SSN", "CREDIT_CARD", "DATE_OF_BIRTH"
    ])
    
    medium_risk_entities: List[str] = field(default_factory=lambda: [
        "ADDRESS", "ORGANIZATION", "IP_ADDRESS"
    ])
    
    low_risk_entities: List[str] = field(default_factory=lambda: [
        "LOCATION", "DATE", "TIME", "URL"
    ])

@dataclass
class PIILoggingConfig:
    """Configuration for logging"""
    log_level: str = "INFO"
    log_dir: str = "./logs"
    log_file_prefix: str = "pii"
    max_log_files: int = 10
    max_log_size_mb: int = 100
    console_logging: bool = True
    file_logging: bool = True

@dataclass
class PIISystemConfig:
    """System-wide PII configuration"""
    enable_gpu: bool = True
    max_memory_gb: float = 8.0
    cache_models: bool = True
    cache_dir: str = "./cache"
    temp_dir: str = "./temp"
    max_concurrent_requests: int = 10
    request_timeout_seconds: int = 300

@dataclass
class PIIConfig:
    """Main PII configuration class"""
    model: PIIModelConfig = field(default_factory=PIIModelConfig)
    training: PIITrainingConfig = field(default_factory=PIITrainingConfig)
    inference: PIIInferenceConfig = field(default_factory=PIIInferenceConfig)
    entities: PIIEntityConfig = field(default_factory=PIIEntityConfig)
    logging: PIILoggingConfig = field(default_factory=PIILoggingConfig)
    system: PIISystemConfig = field(default_factory=PIISystemConfig)
    
    @classmethod
    def from_file(cls, config_path: str) -> 'PIIConfig':
        """Load configuration from file"""
        config_path = Path(config_path)
        
        if not config_path.exists():
            logger.warning(f"Config file not found: {config_path}, using defaults")
            return cls()
        
        try:
            with open(config_path, 'r', encoding='utf-8') as f:
                if config_path.suffix.lower() in ['.yaml', '.yml']:
                    config_dict = yaml.safe_load(f)
                else:
                    config_dict = json.load(f)
            
            return cls.from_dict(config_dict)
            
        except Exception as e:
            logger.error(f"Error loading config file: {e}")
            logger.info("Using default configuration")
            return cls()
    
    @classmethod
    def from_dict(cls, config_dict: Dict[str, Any]) -> 'PIIConfig':
        """Create configuration from dictionary"""
        try:
            config = cls()
            
            # Update model config
            if 'model' in config_dict:
                model_dict = config_dict['model']
                config.model = PIIModelConfig(**model_dict)
            
            # Update training config
            if 'training' in config_dict:
                training_dict = config_dict['training']
                config.training = PIITrainingConfig(**training_dict)
            
            # Update inference config
            if 'inference' in config_dict:
                inference_dict = config_dict['inference']
                config.inference = PIIInferenceConfig(**inference_dict)
            
            # Update entity config
            if 'entities' in config_dict:
                entities_dict = config_dict['entities']
                config.entities = PIIEntityConfig(**entities_dict)
            
            # Update logging config
            if 'logging' in config_dict:
                logging_dict = config_dict['logging']
                config.logging = PIILoggingConfig(**logging_dict)
            
            # Update system config
            if 'system' in config_dict:
                system_dict = config_dict['system']
                config.system = PIISystemConfig(**system_dict)
            
            return config
            
        except Exception as e:
            logger.error(f"Error creating config from dict: {e}")
            return cls()
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert configuration to dictionary"""
        return {
            'model': asdict(self.model),
            'training': asdict(self.training),
            'inference': asdict(self.inference),
            'entities': asdict(self.entities),
            'logging': asdict(self.logging),
            'system': asdict(self.system)
        }
    
    def save_to_file(self, config_path: str, format: str = 'json'):
        """Save configuration to file"""
        config_path = Path(config_path)
        config_path.parent.mkdir(parents=True, exist_ok=True)
        
        config_dict = self.to_dict()
        
        try:
            with open(config_path, 'w', encoding='utf-8') as f:
                if format.lower() == 'yaml':
                    yaml.dump(config_dict, f, default_flow_style=False, indent=2)
                else:
                    json.dump(config_dict, f, indent=2, ensure_ascii=False)
            
            logger.info(f"Configuration saved to: {config_path}")
            
        except Exception as e:
            logger.error(f"Error saving config file: {e}")
            raise
    
    def validate(self) -> List[str]:
        """Validate configuration and return list of issues"""
        issues = []
        
        # Validate model config
        if not self.model.model_name:
            issues.append("Model name cannot be empty")
        
        if self.model.max_length <= 0:
            issues.append("Max length must be positive")
        
        # Validate training config
        if self.training.epochs <= 0:
            issues.append("Training epochs must be positive")
        
        if self.training.batch_size <= 0:
            issues.append("Batch size must be positive")
        
        if not (0 < self.training.learning_rate < 1):
            issues.append("Learning rate must be between 0 and 1")
        
        if not (0 < self.training.test_size < 1):
            issues.append("Test size must be between 0 and 1")
        
        # Validate inference config
        if not (0 <= self.inference.confidence_threshold <= 1):
            issues.append("Confidence threshold must be between 0 and 1")
        
        if self.inference.batch_size <= 0:
            issues.append("Inference batch size must be positive")
        
        # Validate system config
        if self.system.max_memory_gb <= 0:
            issues.append("Max memory must be positive")
        
        if self.system.max_concurrent_requests <= 0:
            issues.append("Max concurrent requests must be positive")
        
        return issues
    
    def get_device(self) -> str:
        """Get the appropriate device based on configuration and availability"""
        if self.model.device == "auto":
            if self.system.enable_gpu and torch.cuda.is_available():
                return "cuda"
            else:
                return "cpu"
        else:
            return self.model.device
    
    def setup_directories(self):
        """Create necessary directories"""
        directories = [
            self.training.output_dir,
            self.logging.log_dir,
            self.system.cache_dir,
            self.system.temp_dir,
            os.path.dirname(self.model.model_path)
        ]
        
        for directory in directories:
            if directory:
                Path(directory).mkdir(parents=True, exist_ok=True)
                logger.debug(f"Created directory: {directory}")

# Predefined configurations
class PIIConfigs:
    """Predefined PII configurations for common use cases"""
    
    @staticmethod
    def quick_test() -> PIIConfig:
        """Configuration for quick testing"""
        config = PIIConfig()
        config.training.epochs = 1
        config.training.batch_size = 8
        config.training.max_samples = 100
        config.inference.batch_size = 16
        return config
    
    @staticmethod
    def production() -> PIIConfig:
        """Configuration for production use"""
        config = PIIConfig()
        config.training.epochs = 5
        config.training.batch_size = 32
        config.inference.batch_size = 64
        config.inference.confidence_threshold = 0.7
        config.system.max_concurrent_requests = 20
        return config
    
    @staticmethod
    def high_accuracy() -> PIIConfig:
        """Configuration for high accuracy requirements"""
        config = PIIConfig()
        config.training.epochs = 10
        config.training.learning_rate = 1e-5
        config.inference.confidence_threshold = 0.8
        config.inference.aggregation_strategy = "average"
        return config
    
    @staticmethod
    def low_resource() -> PIIConfig:
        """Configuration for low resource environments"""
        config = PIIConfig()
        config.model.device = "cpu"
        config.training.batch_size = 4
        config.inference.batch_size = 8
        config.system.enable_gpu = False
        config.system.max_memory_gb = 2.0
        return config

# Utility functions
def load_pii_config(config_path: str = None) -> PIIConfig:
    """Load PII configuration from file or use defaults"""
    if config_path and os.path.exists(config_path):
        return PIIConfig.from_file(config_path)
    else:
        # Try common config file locations
        common_paths = [
            "pii_config.json",
            "config/pii_config.json",
            "pii_recognition/pii_config.json"
        ]
        
        for path in common_paths:
            if os.path.exists(path):
                logger.info(f"Found config file: {path}")
                return PIIConfig.from_file(path)
        
        logger.info("No config file found, using defaults")
        return PIIConfig()

def create_default_config_file(config_path: str = "pii_config.json", format: str = "json"):
    """Create a default configuration file"""
    config = PIIConfig()
    config.save_to_file(config_path, format)
    print(f"✅ Default configuration created: {config_path}")

def validate_config_file(config_path: str) -> bool:
    """Validate a configuration file"""
    try:
        config = PIIConfig.from_file(config_path)
        issues = config.validate()
        
        if not issues:
            print(f"✅ Configuration file is valid: {config_path}")
            return True
        else:
            print(f"❌ Configuration file has issues: {config_path}")
            for issue in issues:
                print(f"   - {issue}")
            return False
            
    except Exception as e:
        print(f"❌ Error validating config file: {e}")
        return False

def get_config_template() -> Dict[str, Any]:
    """Get a configuration template with comments"""
    return {
        "model": {
            "model_name": "bert-base-cased",  # Pre-trained model name
            "model_path": "./models/pii",     # Path to save/load model
            "max_length": 512,                # Maximum sequence length
            "device": "auto"                  # Device: 'cpu', 'cuda', or 'auto'
        },
        "training": {
            "dataset_name": "ai4privacy/pii-masking-200k",  # HuggingFace dataset
            "output_dir": "./models/pii-trained",           # Training output directory
            "epochs": 3,                                    # Number of training epochs
            "batch_size": 16,                               # Training batch size
            "learning_rate": 2e-5,                          # Learning rate
            "max_samples": None                             # Limit samples (None = all)
        },
        "inference": {
            "confidence_threshold": 0.5,     # Minimum confidence for entity detection
            "batch_size": 32,                # Inference batch size
            "mask_token": "[REDACTED]"       # Token for masking PII
        },
        "entities": {
            "entity_types": [
                "PERSON", "EMAIL", "PHONE", "ADDRESS", "SSN", 
                "CREDIT_CARD", "DATE_OF_BIRTH", "ORGANIZATION"
            ],
            "high_risk_entities": ["PERSON", "EMAIL", "PHONE", "SSN"],
            "medium_risk_entities": ["ADDRESS", "ORGANIZATION"],
            "low_risk_entities": ["LOCATION", "DATE", "TIME"]
        },
        "logging": {
            "log_level": "INFO",             # Logging level
            "log_dir": "./logs",             # Log directory
            "console_logging": True          # Enable console logging
        },
        "system": {
            "enable_gpu": True,              # Enable GPU usage
            "max_memory_gb": 8.0,            # Maximum memory usage
            "max_concurrent_requests": 10    # Max concurrent inference requests
        }
    }

# Environment variable support
def load_config_from_env() -> PIIConfig:
    """Load configuration from environment variables"""
    config = PIIConfig()
    
    # Model configuration
    if os.getenv('PII_MODEL_NAME'):
        config.model.model_name = os.getenv('PII_MODEL_NAME')
    if os.getenv('PII_MODEL_PATH'):
        config.model.model_path = os.getenv('PII_MODEL_PATH')
    if os.getenv('PII_DEVICE'):
        config.model.device = os.getenv('PII_DEVICE')
    
    # Training configuration
    if os.getenv('PII_EPOCHS'):
        config.training.epochs = int(os.getenv('PII_EPOCHS'))
    if os.getenv('PII_BATCH_SIZE'):
        config.training.batch_size = int(os.getenv('PII_BATCH_SIZE'))
    if os.getenv('PII_LEARNING_RATE'):
        config.training.learning_rate = float(os.getenv('PII_LEARNING_RATE'))
    if os.getenv('PII_OUTPUT_DIR'):
        config.training.output_dir = os.getenv('PII_OUTPUT_DIR')
    
    # Inference configuration
    if os.getenv('PII_CONFIDENCE_THRESHOLD'):
        config.inference.confidence_threshold = float(os.getenv('PII_CONFIDENCE_THRESHOLD'))
    if os.getenv('PII_MASK_TOKEN'):
        config.inference.mask_token = os.getenv('PII_MASK_TOKEN')
    
    # System configuration
    if os.getenv('PII_ENABLE_GPU'):
        config.system.enable_gpu = os.getenv('PII_ENABLE_GPU').lower() == 'true'
    if os.getenv('PII_MAX_MEMORY_GB'):
        config.system.max_memory_gb = float(os.getenv('PII_MAX_MEMORY_GB'))
    
    # Logging configuration
    if os.getenv('PII_LOG_LEVEL'):
        config.logging.log_level = os.getenv('PII_LOG_LEVEL')
    if os.getenv('PII_LOG_DIR'):
        config.logging.log_dir = os.getenv('PII_LOG_DIR')
    
    return config

# Configuration merger
def merge_configs(*configs: PIIConfig) -> PIIConfig:
    """Merge multiple configurations (later configs override earlier ones)"""
    merged = PIIConfig()
    
    for config in configs:
        merged_dict = merged.to_dict()
        config_dict = config.to_dict()
        
        # Deep merge dictionaries
        def deep_merge(dict1, dict2):
            for key, value in dict2.items():
                if key in dict1 and isinstance(dict1[key], dict) and isinstance(value, dict):
                    deep_merge(dict1[key], value)
                else:
                    dict1[key] = value
        
        deep_merge(merged_dict, config_dict)
        merged = PIIConfig.from_dict(merged_dict)
    
    return merged

# Config manager class
class PIIConfigManager:
    """Configuration manager for PII recognition"""
    
    def __init__(self, config_path: str = None):
        self.config_path = config_path
        self.config = self.load_config()
    
    def load_config(self) -> PIIConfig:
        """Load configuration from various sources"""
        configs = []
        
        # 1. Start with defaults
        configs.append(PIIConfig())
        
        # 2. Load from file if specified
        if self.config_path:
            try:
                file_config = PIIConfig.from_file(self.config_path)
                configs.append(file_config)
            except Exception as e:
                logger.warning(f"Failed to load config from file: {e}")
        
        # 3. Override with environment variables
        try:
            env_config = load_config_from_env()
            configs.append(env_config)
        except Exception as e:
            logger.warning(f"Failed to load config from environment: {e}")
        
        # Merge all configurations
        final_config = merge_configs(*configs)
        
        # Validate final configuration
        issues = final_config.validate()
        if issues:
            logger.warning(f"Configuration validation issues: {issues}")
        
        # Setup directories
        final_config.setup_directories()
        
        return final_config
    
    def reload_config(self):
        """Reload configuration"""
        self.config = self.load_config()
        logger.info("Configuration reloaded")
    
    def get_training_config(self) -> PIITrainingConfig:
        """Get training configuration"""
        return self.config.training
    
    def get_inference_config(self) -> PIIInferenceConfig:
        """Get inference configuration"""
        return self.config.inference
    
    def get_model_config(self) -> PIIModelConfig:
        """Get model configuration"""
        return self.config.model
    
    def update_config(self, updates: Dict[str, Any]):
        """Update configuration with new values"""
        current_dict = self.config.to_dict()
        
        # Deep merge updates
        def deep_merge(dict1, dict2):
            for key, value in dict2.items():
                if key in dict1 and isinstance(dict1[key], dict) and isinstance(value, dict):
                    deep_merge(dict1[key], value)
                else:
                    dict1[key] = value
        
        deep_merge(current_dict, updates)
        self.config = PIIConfig.from_dict(current_dict)
        
        # Validate after update
        issues = self.config.validate()
        if issues:
            logger.warning(f"Configuration validation issues after update: {issues}")
    
    def save_config(self, path: str = None, format: str = "json"):
        """Save current configuration to file"""
        save_path = path or self.config_path or "pii_config.json"
        self.config.save_to_file(save_path, format)

# Test and utility functions
def test_config_system():
    """Test the configuration system"""
    print("🧪 Testing PII Configuration System...")
    
    try:
        # Test 1: Default configuration
        config = PIIConfig()
        issues = config.validate()
        assert len(issues) == 0, f"Default config has issues: {issues}"
        print("✅ Default configuration is valid")
        
        # Test 2: Save and load configuration
        test_path = "test_pii_config.json"
        config.save_to_file(test_path)
        loaded_config = PIIConfig.from_file(test_path)
        assert config.to_dict() == loaded_config.to_dict(), "Save/load mismatch"
        print("✅ Save/load functionality works")
        
        # Test 3: Predefined configurations
        test_configs = [
            PIIConfigs.quick_test(),
            PIIConfigs.production(),
            PIIConfigs.high_accuracy(),
            PIIConfigs.low_resource()
        ]
        
        for test_config in test_configs:
            issues = test_config.validate()
            assert len(issues) == 0, f"Predefined config has issues: {issues}"
        print("✅ All predefined configurations are valid")
        
        # Test 4: Configuration manager
        manager = PIIConfigManager()
        assert manager.config is not None, "Config manager failed to load"
        print("✅ Configuration manager works")
        
        # Cleanup
        if os.path.exists(test_path):
            os.remove(test_path)
        
        print("🎉 All configuration tests passed!")
        return True
        
    except Exception as e:
        print(f"❌ Configuration test failed: {e}")
        return False

# Command line interface for config management
def main():
    """Command line interface for configuration management"""
    import argparse
    
    parser = argparse.ArgumentParser(description="PII Configuration Management")
    parser.add_argument('--create-default', action='store_true', help='Create default config file')
    parser.add_argument('--validate', type=str, help='Validate config file')
    parser.add_argument('--template', action='store_true', help='Show config template')
    parser.add_argument('--test', action='store_true', help='Test configuration system')
    parser.add_argument('--output', type=str, default='pii_config.json', help='Output file path')
    parser.add_argument('--format', type=str, choices=['json', 'yaml'], default='json', help='Config file format')
    
    args = parser.parse_args()
    
    if args.create_default:
        create_default_config_file(args.output, args.format)
    
    elif args.validate:
        validate_config_file(args.validate)
    
    elif args.template:
        template = get_config_template()
        print("📋 PII Configuration Template:")
        print(json.dumps(template, indent=2))
    
    elif args.test:
        test_config_system()
    
    else:
        parser.print_help()

if __name__ == "__main__":
    main()
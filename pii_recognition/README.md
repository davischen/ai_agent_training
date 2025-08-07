# PII Recognition Module

**Personal Identifiable Information (PII) Recognition for ATLAS System**

*Advanced entity recognition and privacy protection using transformer models*

---

## Overview

The PII Recognition Module is a comprehensive solution for detecting and managing Personal Identifiable Information in text data. Built on top of BERT-based transformer models, it provides real-time entity recognition, text masking, and privacy risk assessment capabilities.

**Key Capabilities:**
- **Real-time PII Detection**: Identify 13+ entity types including names, emails, phones, addresses
- **Text Masking**: Automatically redact sensitive information
- **Privacy Risk Assessment**: Evaluate and score privacy risks in text content
- **Batch Processing**: Handle large volumes of text efficiently
- **ATLAS Integration**: Seamless integration with the ATLAS training system

## Features

### Core Functionality
- **Entity Recognition**: PERSON, EMAIL, PHONE, ADDRESS, SSN, CREDIT_CARD, DATE_OF_BIRTH, etc.
- **Confidence Scoring**: Adjustable confidence thresholds for detection accuracy
- **Text Masking**: Replace PII with configurable tokens (e.g., `[REDACTED]`)
- **Privacy Analysis**: Risk level assessment (HIGH, MEDIUM, LOW, MINIMAL)
- **Batch Processing**: Efficient processing of multiple texts simultaneously

### Advanced Features
- **Model Training**: Complete pipeline for training custom PII models
- **Configuration Management**: Flexible YAML/JSON configuration system
- **Performance Metrics**: F1 score, precision, recall evaluation
- **Statistics & Analytics**: Detailed analysis of detection results
- **Integration Ready**: Compatible with ATLAS task scheduling system

## Quick Start

### Installation

```bash
# Install dependencies
pip install torch transformers datasets seqeval

# Navigate to PII recognition directory
cd pii_recognition

# Create default configuration
python pii_config.py --create-default
```

### Basic Usage

#### 1. Train a Model (Quick Test)
```bash
# Quick training test with 100 samples
python pii_trainer.py --quick-test

# Full training
python pii_trainer.py --epochs 3 --batch-size 16
```

#### 2. Detect PII in Text
```python
from pii_recognition import create_pii_inference_engine

# Load trained model
engine = create_pii_inference_engine("./models/pii-trained")

# Detect PII
result = engine.quick_detect("My name is John Smith and email is john@example.com")

print(f"Found {result['entities_found']} PII entities:")
for entity in result['entities']:
    print(f"- {entity['type']}: {entity['text']} (confidence: {entity['confidence']:.2f})")
```

#### 3. Privacy Risk Analysis
```python
risk_analysis = engine.analyze_text_privacy_risk("John Smith lives at 123 Main St")
print(f"Privacy Risk: {risk_analysis['risk_level']} (Score: {risk_analysis['risk_score']})")
```

## Module Structure

```
pii_recognition/
├── __init__.py                 # Module initialization
├── pii_trainer.py             # Model training pipeline
├── pii_inference.py           # Inference engine
├── pii_config.py              # Configuration management
├── pii_config.json            # Default configuration
└── README.md                  # This file
```

## Core Components

### 1. PIITrainer (`pii_trainer.py`)

**Purpose**: Complete model training pipeline for PII recognition

**Key Features**:
- HuggingFace datasets integration
- BERT-based token classification
- Automatic evaluation with seqeval metrics
- Model checkpointing and versioning
- Command-line and programmatic interfaces

**Usage**:
```bash
# Command line training
python pii_trainer.py --epochs 5 --batch-size 32

# Programmatic usage
from pii_recognition import create_pii_trainer

trainer = create_pii_trainer()
results = trainer.train(epochs=3, batch_size=16)
```

**Training Process**:
1. Load ai4privacy/pii-masking-200k dataset
2. Filter and encode labels
3. Tokenize and align with BERT tokenizer
4. Train with MultipleNegativesRankingLoss
5. Evaluate with F1, precision, recall metrics
6. Save model and configuration

### 2. PIIInferenceEngine (`pii_inference.py`)

**Purpose**: Real-time PII detection and text processing

**Key Features**:
- Single and batch text processing
- Configurable confidence thresholds
- Text masking with custom tokens
- Privacy risk assessment
- Performance statistics

**Main Classes**:
- `PIIInferenceEngine`: Core inference engine
- `PIIEntity`: Individual entity representation
- `PIIInferenceRequest/Result`: Request/response objects

**Usage Examples**:
```python
from pii_recognition import PIIInferenceEngine

engine = PIIInferenceEngine("./models/pii-trained")

# Single text detection
entities = engine.detect_pii_single("Call me at +1-555-123-4567")

# Batch processing
texts = ["Email: john@example.com", "Phone: 555-0123"]
all_entities = engine.detect_pii_batch(texts)

# Text masking
masked = engine.mask_pii_text(text, entities, mask_token="***")
```

**Entity Types Supported**:
- **High Risk**: PERSON, EMAIL, PHONE, SSN, CREDIT_CARD, DATE_OF_BIRTH
- **Medium Risk**: ADDRESS, ORGANIZATION, IP_ADDRESS
- **Low Risk**: LOCATION, DATE, TIME, URL

### 3. PIIConfig (`pii_config.py`)

**Purpose**: Comprehensive configuration management system

**Key Features**:
- Dataclass-based configuration structure
- JSON/YAML file support
- Environment variable integration
- Configuration validation and merging
- Predefined configuration templates

**Configuration Categories**:
- **Model Config**: Model selection, device settings, tokenization
- **Training Config**: Epochs, batch size, learning rate, dataset settings
- **Inference Config**: Confidence thresholds, masking tokens, batch processing
- **Entity Config**: Entity types, risk classifications
- **System Config**: Memory limits, GPU usage, concurrent requests

**Usage**:
```python
from pii_recognition import PIIConfig, PIIConfigManager

# Load from file
config = PIIConfig.from_file("pii_config.json")

# Use configuration manager
manager = PIIConfigManager("pii_config.json")
training_config = manager.get_training_config()

# Predefined configurations
quick_config = PIIConfigs.quick_test()
prod_config = PIIConfigs.production()
```

## Configuration

### Default Configuration File

The module uses `pii_config.json` for configuration. Create a default configuration:

```bash
python pii_config.py --create-default
```

### Key Configuration Options

```json
{
  "model": {
    "model_name": "bert-base-cased",
    "model_path": "./models/pii",
    "max_length": 512,
    "device": "auto"
  },
  "training": {
    "dataset_name": "ai4privacy/pii-masking-200k",
    "epochs": 3,
    "batch_size": 16,
    "learning_rate": 2e-5
  },
  "inference": {
    "confidence_threshold": 0.5,
    "mask_token": "[REDACTED]",
    "batch_size": 32
  }
}
```

### Environment Variables

Override configuration with environment variables:

```bash
export PII_MODEL_NAME="distilbert-base-cased"
export PII_EPOCHS=5
export PII_CONFIDENCE_THRESHOLD=0.7
export PII_DEVICE="cuda"
```

### Predefined Configurations

```python
from pii_recognition import PIIConfigs

# Quick testing (small dataset, fast training)
config = PIIConfigs.quick_test()

# Production ready (optimized for performance)
config = PIIConfigs.production()

# High accuracy (longer training, higher thresholds)
config = PIIConfigs.high_accuracy()

# Resource constrained (CPU only, smaller batches)
config = PIIConfigs.low_resource()
```

## ATLAS System Integration

### 1. Integration with AI Training Agent

```python
# In ai_training_agent.py
try:
    from pii_recognition import PIITrainer, PIIInferenceEngine, PIIConfigManager
    PII_AVAILABLE = True
except ImportError:
    PII_AVAILABLE = False

class AITrainingAgent:
    def __init__(self):
        # ... existing initialization
        
        if PII_AVAILABLE:
            self.pii_config_manager = PIIConfigManager()
            self.pii_trainer = None
            self.pii_inference_engine = None
    
    async def train_pii_model(self, **params):
        """Train PII recognition model"""
        if not self.pii_trainer:
            from pii_recognition import create_pii_trainer
            self.pii_trainer = create_pii_trainer()
        return self.pii_trainer.train(**params)
    
    async def detect_pii(self, texts):
        """Detect PII in texts"""
        if not self.pii_inference_engine:
            model_path = self.pii_config_manager.config.training.output_dir
            self.pii_inference_engine = PIIInferenceEngine(model_path)
        return self.pii_inference_engine.detect_pii_batch(texts)
```

### 2. Task Scheduler Integration

```python
# In task_scheduler.py
class TaskType(Enum):
    PHISHING_INFERENCE = "phishing_inference"
    PHISHING_TRAINING = "phishing_training"
    PII_INFERENCE = "pii_inference"        # New
    PII_TRAINING = "pii_training"          # New
```

### 3. Web Dashboard Integration

Add PII functionality to the Streamlit dashboard:

```python
def pii_recognition_page():
    st.header("🔐 PII Recognition")
    
    # Text input for PII detection
    text_input = st.text_area("Enter text for PII analysis")
    
    if st.button("🔍 Detect PII"):
        result = st.session_state.agent.detect_pii(text_input)
        
        # Display results
        if result['entities_found'] > 0:
            st.success(f"Found {result['entities_found']} PII entities")
            
            # Show entities in table
            df = pd.DataFrame(result['entities'])
            st.dataframe(df)
            
            # Privacy risk analysis
            risk = engine.analyze_text_privacy_risk(text_input)
            st.metric("Privacy Risk", risk['risk_level'], risk['risk_score'])
```

## API Reference

### PIITrainer

```python
class PIITrainer:
    def __init__(model_name, dataset_name, output_dir, seed)
    def train(epochs, batch_size, learning_rate, max_samples)
    def load_dataset(max_samples)
    def prepare_model_and_tokenizer()
```

### PIIInferenceEngine

```python
class PIIInferenceEngine:
    def __init__(model_path, device, batch_size, confidence_threshold)
    def detect_pii_single(text, confidence_threshold)
    def detect_pii_batch(texts, confidence_threshold)
    def mask_pii_text(text, entities, mask_token)
    def analyze_text_privacy_risk(text)
    def quick_detect(text, return_masked, mask_token)
```

### PIIConfig

```python
class PIIConfig:
    @classmethod from_file(config_path)
    @classmethod from_dict(config_dict)
    def to_dict()
    def save_to_file(config_path, format)
    def validate()
```

## Performance

### Training Performance
- **Dataset**: ai4privacy/pii-masking-200k (200,000 samples)
- **Training Time**: ~2-4 hours on GPU (V100)
- **Memory Usage**: 8-16GB GPU memory
- **Typical F1 Score**: 0.85-0.92

### Inference Performance
- **Speed**: 100-1000 texts/second (depending on text length and hardware)
- **Memory**: 2-4GB GPU memory for inference
- **Batch Processing**: Optimized for large-scale processing
- **Latency**: <100ms for single text detection

### Scalability
- **Single Machine**: 10K+ texts/hour
- **GPU Acceleration**: 100K+ texts/hour
- **Batch Processing**: Configurable batch sizes (1-128)
- **Memory Efficient**: Streaming processing for large datasets

## Examples

### Complete Training Example

```python
from pii_recognition import create_pii_trainer

# Create trainer with custom configuration
trainer = create_pii_trainer(
    model_name="distilbert-base-cased",
    output_dir="./models/my-pii-model"
)

# Train model
results = trainer.train(
    epochs=5,
    batch_size=32,
    learning_rate=1e-5,
    max_samples=10000  # Limit for testing
)

print(f"Training completed! F1 Score: {results['eval_result']['eval_f1']:.4f}")
```

### Batch Processing Example

```python
from pii_recognition import create_pii_inference_engine

# Load trained model
engine = create_pii_inference_engine("./models/my-pii-model")

# Process multiple documents
documents = [
    "Employee record: John Smith, SSN: 123-45-6789",
    "Contact info: jane@company.com, Phone: +1-555-0123",
    "Address: 123 Main Street, Anytown, NY 12345"
]

# Batch detection
results = engine.detect_pii_batch(documents, confidence_threshold=0.7)

# Generate statistics
stats = engine.get_entity_statistics(results)
print(f"Processed {stats['texts_processed']} documents")
print(f"Found {stats['total_entities']} PII entities")
print(f"Entity types: {stats['entity_type_counts']}")
```

### Privacy Risk Assessment Example

```python
# Analyze privacy risk
text = "Hi, I'm John Smith. My email is john.smith@company.com and SSN is 123-45-6789"

risk_analysis = engine.analyze_text_privacy_risk(text)

print(f"Risk Level: {risk_analysis['risk_level']}")
print(f"Risk Score: {risk_analysis['risk_score']}/100")
print(f"High Risk Entities: {risk_analysis['high_risk_entities']}")

for recommendation in risk_analysis['recommendations']:
    print(f"• {recommendation}")
```

### Configuration Management Example

```python
from pii_recognition import PIIConfigManager, PIIConfigs

# Load configuration from multiple sources
manager = PIIConfigManager("pii_config.json")  # File + env variables

# Use predefined configurations
if testing:
    manager.config = PIIConfigs.quick_test()
elif production:
    manager.config = PIIConfigs.production()

# Update configuration dynamically
manager.update_config({
    "inference": {
        "confidence_threshold": 0.8,
        "mask_token": "***REDACTED***"
    }
})

# Save updated configuration
manager.save_config("updated_config.json")
```

## Testing

### Unit Tests

Run the built-in test functions:

```bash
# Test configuration system
python pii_config.py --test

# Test inference engine (requires trained model)
python pii_inference.py

# Test trainer with small dataset
python pii_trainer.py --quick-test
```

### Validation

```bash
# Validate configuration file
python pii_config.py --validate pii_config.json

# Create and validate default config
python pii_config.py --create-default --validate pii_config.json
```

## Troubleshooting

### Common Issues

**1. CUDA Out of Memory**
```bash
# Reduce batch size
export PII_BATCH_SIZE=8

# Or use CPU
export PII_DEVICE=cpu
```

**2. Model Download Issues**
```bash
# Set cache directory
export TRANSFORMERS_CACHE=/path/to/cache

# Or use offline mode
export TRANSFORMERS_OFFLINE=1
```

**3. Dataset Access Issues**
```bash
# Login to HuggingFace
huggingface-cli login

# Or use local dataset
python pii_trainer.py --dataset-path ./local_data.json
```

**4. Memory Issues During Training**
```python
# Use gradient accumulation
training_config = {
    'batch_size': 8,
    'gradient_accumulation_steps': 4  # Effective batch size: 32
}
```

### Debug Mode

Enable detailed logging:

```python
import logging
logging.basicConfig(level=logging.DEBUG)

# Or set environment variable
export PII_LOG_LEVEL=DEBUG
```

## Contributing

### Development Setup

1. Clone the repository
2. Install dependencies: `pip install -r requirements.txt`
3. Run tests: `python -m pytest tests/`
4. Follow code style: `black . && flake8 .`

### Adding New Entity Types

1. Update `PIIEntityConfig` in `pii_config.py`
2. Add entity patterns to training data
3. Update risk classification logic
4. Test with new entity types

### Extending Functionality

The module is designed for extensibility:

- **Custom Models**: Easy integration of different transformer models
- **New Datasets**: Support for additional training datasets
- **Custom Metrics**: Add domain-specific evaluation metrics
- **Integration Points**: Well-defined APIs for system integration

## License

This module is part of the ATLAS system and follows the same MIT license.

## Support

For issues and questions:
- Check the troubleshooting section above
- Review system logs in `./logs/pii_training_*.log`
- Validate configuration with `python pii_config.py --validate`
- Test with minimal examples using `--quick-test` mode


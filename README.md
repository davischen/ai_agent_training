# ATLAS System Documentation

**Adaptive Training and Learning Automation System**  
*Version 1.0.0 | August 2025*

---

## About ATLAS

ATLAS is a modern AI system for phishing email detection. Built with Python, it combines machine learning with an intuitive web interface to help organizations protect themselves from email-based threats.

**Core Mission**: Automate the detection and classification of phishing emails using advanced natural language processing and machine learning techniques.

## Key Features

- **Real-time Classification**: Instant phishing detection with confidence scores
- **Automated Training**: Self-improving models with synthetic data generation  
- **Web Dashboard**: Clean, modern interface for system management
- **Task Management**: Priority-based scheduling with resource monitoring
- **Model Flexibility**: Import, export, and version control for trained models

## System Requirements

**Minimum Setup**:
- Python 3.8+
- 8GB RAM  
- 2GB storage
- Multi-core CPU

**Recommended Setup**:
- Python 3.9+
- 16GB RAM
- CUDA-compatible GPU
- SSD storage

## Installation

### Quick Setup
```bash
# Install dependencies
pip install torch sentence-transformers streamlit pandas numpy scikit-learn plotly nltk schedule

# Initialize system
python run_agent.py --mode init

# Run tests
python test_atlas.py
```

### Verify Installation
```bash
# Test core functionality
python run_agent.py --mode test

# Launch web interface
python run_streamlit.py
```

## Usage

### Command Line Interface

**Basic Operations**:
```bash
python run_agent.py --mode basic      # Demo inference
python run_agent.py --mode training   # Demo training  
python run_agent.py --mode init       # Setup system
```

**Advanced Usage**:
```bash
python run_agent.py --config custom.json --log-level DEBUG
```

### Web Dashboard

Launch with `python run_streamlit.py` and navigate to `http://localhost:8501`

**Main Sections**:
- **Overview**: System status and real-time metrics
- **Inference**: Submit emails for classification
- **Training**: Create and train custom models
- **Analytics**: Performance charts and model comparison
- **Models**: Import/export and manage trained models
- **Settings**: System configuration and maintenance

### API Usage

```python
from ai_training_agent import AITrainingAgent

# Initialize
agent = AITrainingAgent()

# Classify emails
emails = ["Urgent: Verify your account!", "Meeting at 2pm today"]
results = await agent.quick_inference(emails)

# Train custom model
data_path = agent.create_sample_training_data(1000)
task_id = await agent.train_model_from_data(data_path)

# Monitor system
status = agent.get_system_status()
```

## Configuration

**Basic Configuration** (`config.json`):
```json
{
  "system": {
    "model_dir": "./models",
    "max_concurrent_tasks": 5,
    "log_level": "INFO"
  },
  "training": {
    "epochs": 3,
    "batch_size": 16
  }
}
```

**Directory Structure**:
```
atlas/
├── data/          # Training datasets
├── models/        # AI models
├── logs/          # System logs
└── config/        # Configuration files
```

## Performance

**Typical Metrics**:
- Inference: 100-500 emails/second
- Training: 2-10 minutes per 1000 samples  
- Accuracy: 85-95% on balanced datasets
- Memory: 2-8GB depending on model size

**Scalability**:
- Single machine: 10K emails/day
- Server setup: 100K emails/day
- GPU acceleration: 1M+ emails/day

## File Overview

**Core System** (7 files):
- `ai_training_agent.py` - Main orchestrator
- `inference_engine.py` - Real-time classification
- `model_manager.py` - Model lifecycle management  
- `training_engine.py` - Training pipeline
- `task_scheduler.py` - Task and resource management
- `run_agent.py` - Command line interface
- `run_streamlit.py` - Web launcher

**Web Interface** (1 file):
- `streamlit_dashboard.py` - Complete web management

**Research Code** (2 files):
- `preprocess_phishing_data.py` - Original research pipeline
- `simple_data_preprocessor.py` - Modular data processing

**Testing** (1 file):
- `test_atlas.py` - System validation

## Troubleshooting

**Common Issues**:

*Import errors*:
```bash
pip install -r requirements.txt
```

*Memory issues*:
- Reduce batch_size in config.json
- Limit max_concurrent_tasks

*Dashboard won't start*:
```bash
pip install streamlit
streamlit run streamlit_dashboard.py --server.port 8502
```

*Model download fails*:
- Check internet connection
- Models cache after first download

**Debug Mode**:
```bash
python run_agent.py --log-level DEBUG
```

**Log Locations**:
- Console output for immediate issues
- `logs/atlas_agent.log` for detailed system logs
- Dashboard interface for task-specific logs

## Data Format

**Training CSV Structure**:
```csv
Email Text,Email Type
"Hi team, meeting at 2pm","Safe Email"
"URGENT: Verify now!","Phishing Email"
```

**Classification Results**:
```json
{
  "predictions": [0, 1],
  "confidences": [0.23, 0.87],
  "text_data": ["Email 1", "Email 2"]
}
```

## Advanced Features

**Multiple Inference Methods**:
- Zero-shot: Semantic understanding
- Few-shot: Reference-based comparison  
- Keyword: Pattern matching
- Similarity: Embedding-based classification

**Task Priorities**:
- CRITICAL: System recovery
- HIGH: Real-time inference
- NORMAL: Regular training
- LOW: Data cleanup
- BACKGROUND: Maintenance

**Resource Management**:
- Automatic CPU/memory allocation
- GPU utilization monitoring
- Task queue optimization
- Resource usage visualization

## Deployment

**Production Considerations**:
- Use dedicated server with sufficient resources
- Configure security settings appropriately
- Set up monitoring and alerting
- Enable automatic backups
- Use reverse proxy for web access

**Scaling Options**:
- Vertical: Increase server resources
- Horizontal: Distribute across multiple machines
- Cloud: Deploy to AWS/Azure/GCP
- Container: Use Docker for portability

## Contributing

**Development Setup**:
1. Clone/download system files
2. Install development dependencies
3. Run tests: `python test_atlas.py`
4. Make changes and test thoroughly
5. Update documentation as needed

**Code Standards**:
- Clear, commented code
- Error handling for edge cases
- Consistent naming conventions
- Comprehensive testing

## Support

**Getting Help**:
1. Check this documentation first
2. Review system logs for errors
3. Test with sample data
4. Use debug mode for detailed output

**Self-Diagnosis**:
- Run `python test_atlas.py` for system health
- Check `logs/atlas_agent.log` for detailed errors
- Monitor resource usage in web dashboard
- Verify configuration in `config.json`

## License

MIT License - Free to use, modify, and distribute.

Full license text available in source files.

## Technical Details

**Dependencies**:
- PyTorch: Deep learning framework
- Sentence Transformers: Text embeddings
- Streamlit: Web interface
- Scikit-learn: ML utilities
- Pandas/NumPy: Data processing
- Plotly: Visualizations

**Architecture**:
- Modular design with clear separation of concerns
- Asynchronous processing for scalability  
- Resource-aware task scheduling
- Comprehensive logging and monitoring
- RESTful internal APIs

**Security**:
- No external network calls during inference
- Local model storage and processing
- Configurable logging levels
- Resource usage limits
- Input validation and sanitization

---

*Built by the ATLAS Development Team*  
*Making email security accessible to everyone*

**Quick Links**:
- Start Here: `python run_agent.py --mode init`
- Web Interface: `python run_streamlit.py` 
- Full Test: `python test_atlas.py`
- Original Pipeline: `python preprocess_phishing_data.py`
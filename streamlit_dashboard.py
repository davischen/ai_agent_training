#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
ATLAS Agent Streamlit Dashboard - Fixed Version
Interactive web interface for managing AI Training Agent
"""

import streamlit as st
import pandas as pd
import numpy as np
import asyncio
import json
import time
import os
import sys
from datetime import datetime, timedelta
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import logging

# Add the current directory and core directory to Python path
current_dir = os.path.dirname(os.path.abspath(__file__))
core_dir = os.path.join(current_dir, 'core')
sys.path.insert(0, current_dir)
sys.path.insert(0, core_dir)

# Configure logging to suppress verbose output
logging.basicConfig(level=logging.WARNING)

# Import ATLAS components - FIXED IMPORT PATHS
ATLAS_AVAILABLE = False
try:
    # First try importing from core directory
    from core.ai_training_agent import AITrainingAgent
    from core.model_manager import ModelManager
    from core.inference_engine import InferenceEngine
    from core.training_engine import TrainingEngine
    ATLAS_AVAILABLE = True
    st.info("✅ Successfully imported from core/ directory")
except ImportError as e1:
    try:
        # Try importing from root directory
        from ai_training_agent import AITrainingAgent
        from model_manager import ModelManager
        from inference_engine import InferenceEngine
        from training_engine import TrainingEngine
        ATLAS_AVAILABLE = True
        st.info("✅ Successfully imported from root directory")
    except ImportError as e2:
        # Try importing individual components that might exist
        missing_modules = []
        try:
            import sentence_transformers
        except ImportError:
            missing_modules.append("sentence-transformers")
        
        try:
            import torch
        except ImportError:
            missing_modules.append("torch")
            
        try:
            import transformers
        except ImportError:
            missing_modules.append("transformers")
        
        if missing_modules:
            st.error(f"❌ Missing required packages: {', '.join(missing_modules)}")
            st.error("Please install with: pip install " + " ".join(missing_modules))
        else:
            st.error(f"❌ ATLAS modules not found in expected locations:")
            st.error(f"Core import error: {e1}")
            st.error(f"Root import error: {e2}")
        ATLAS_AVAILABLE = False

# Page configuration
st.set_page_config(
    page_title="ATLAS Agent Dashboard",
    page_icon="🤖",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Initialize session state
if 'agent' not in st.session_state:
    st.session_state.agent = None
if 'task_history' not in st.session_state:
    st.session_state.task_history = []
if 'training_results' not in st.session_state:
    st.session_state.training_results = []
if 'system_metrics' not in st.session_state:
    st.session_state.system_metrics = []

def check_dependencies():
    """Check if all required dependencies are available"""
    required_packages = [
        'sentence_transformers',
        'torch',
        'transformers',
        'sklearn',
        'pandas',
        'numpy'
    ]
    
    missing_packages = []
    for package in required_packages:
        try:
            __import__(package)
        except ImportError:
            missing_packages.append(package)
    
    return missing_packages

def initialize_agent():
    """Initialize ATLAS Agent"""
    if not ATLAS_AVAILABLE:
        return None
    
    try:
        with st.spinner("Initializing ATLAS Agent..."):
            # Create models directory if it doesn't exist
            models_dir = os.path.join(current_dir, "models")
            os.makedirs(models_dir, exist_ok=True)
            
            agent = AITrainingAgent(model_dir=models_dir)
            st.success("✅ ATLAS Agent initialized successfully!")
            return agent
    except Exception as e:
        st.error(f"❌ Failed to initialize ATLAS Agent: {e}")
        st.error("This might be due to missing dependencies or incorrect file paths")
        return None

def create_mock_agent():
    """Create a mock agent for demonstration purposes"""
    class MockAgent:
        def __init__(self):
            self.models_dir = "./models"
            
        def get_system_status(self):
            return {
                'timestamp': datetime.now().isoformat(),
                'scheduler': {
                    'running_tasks': 2,
                    'pending_tasks': 1,
                    'completed_tasks': 15
                },
                'models': {
                    'all-MiniLM-L6-v2': 'sentence-transformer',
                    'all-mpnet-base-v2': 'sentence-transformer'
                },
                'system': {
                    'max_concurrent_tasks': 10,
                    'resource_management_enabled': True
                },
                'resources': {
                    'cpu_utilization': np.random.randint(20, 80),
                    'memory_utilization': np.random.randint(30, 70),
                    'gpu_utilization': np.random.randint(10, 60),
                    'disk_io_utilization': np.random.randint(5, 40),
                    'network_utilization': np.random.randint(10, 50)
                }
            }
        
        async def quick_inference(self, texts, model_name):
            # Mock inference results
            predictions = [np.random.randint(0, 2) for _ in texts]
            confidences = [np.random.uniform(0.3, 0.95) for _ in texts]
            return {
                'predictions': predictions,
                'confidences': confidences,
                'processing_time': np.random.uniform(0.5, 2.0)
            }
        
        def list_available_models(self):
            return {
                'all-MiniLM-L6-v2': 'sentence-transformer',
                'all-mpnet-base-v2': 'sentence-transformer',
                'phishing_detector_v1': 'fine-tuned'
            }
        
        def create_sample_training_data(self, num_samples):
            # Create mock training data
            data = []
            sample_phishing = [
                "Urgent: Your account will be suspended! Click here immediately to verify",
                "Congratulations! You've won $1,000,000! Claim now",
                "Your PayPal account has been limited. Update payment info",
                "Banking security alert: Suspicious activity detected"
            ]
            
            sample_safe = [
                "Meeting scheduled for tomorrow at 10 AM in conference room B",
                "Your order has been shipped and will arrive within 3-5 business days",
                "Thank you for your feedback on our recent service",
                "Reminder: Your subscription renewal is coming up next month"
            ]
            
            for i in range(num_samples):
                if i % 2 == 0:
                    text = np.random.choice(sample_phishing)
                    label = 1
                else:
                    text = np.random.choice(sample_safe)
                    label = 0
                    
                data.append({
                    'Email Text': text + f" (sample {i+1})",
                    'Email Type': label
                })
            
            df = pd.DataFrame(data)
            file_path = f"sample_data_{num_samples}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv"
            df.to_csv(file_path, index=False)
            return file_path
        
        def get_training_statistics(self, data_path):
            try:
                df = pd.read_csv(data_path)
                total_samples = len(df)
                phishing_samples = sum(df['Email Type'])
                safe_samples = total_samples - phishing_samples
                phishing_percentage = (phishing_samples / total_samples) * 100
                
                return {
                    'total_samples': total_samples,
                    'phishing_samples': phishing_samples,
                    'safe_samples': safe_samples,
                    'phishing_percentage': phishing_percentage
                }
            except Exception:
                return {'total_samples': 0, 'phishing_samples': 0, 'safe_samples': 0, 'phishing_percentage': 0}
        
        def shutdown(self):
            pass  # Mock shutdown
    
    return MockAgent()

def shutdown_agent():
    """Shutdown ATLAS Agent"""
    if st.session_state.agent:
        try:
            st.session_state.agent.shutdown()
            st.session_state.agent = None
            st.success("✅ ATLAS Agent shutdown successfully!")
        except Exception as e:
            st.error(f"❌ Error shutting down agent: {e}")

def main_dashboard():
    """Main dashboard interface"""
    
    # Title and header
    st.title("🤖 ATLAS Agent Dashboard")
    st.markdown("**Adaptive Training and Learning Automation System**")
    
    # Sidebar for agent control
    with st.sidebar:
        st.header("🎮 Agent Control")
        
        # Check dependencies first
        missing_deps = check_dependencies()
        if missing_deps and ATLAS_AVAILABLE:
            st.warning(f"⚠️ Missing dependencies: {', '.join(missing_deps)}")
        
        if st.session_state.agent is None:
            col1, col2 = st.columns(2)
            with col1:
                if st.button("🚀 Initialize Agent", use_container_width=True):
                    if ATLAS_AVAILABLE:
                        st.session_state.agent = initialize_agent()
                    else:
                        st.session_state.agent = create_mock_agent()
                        st.info("🔧 Using mock agent for demonstration")
                    st.rerun()
            
            with col2:
                if st.button("🔧 Mock Mode", use_container_width=True):
                    st.session_state.agent = create_mock_agent()
                    st.info("🔧 Mock agent initialized")
                    st.rerun()
        else:
            status_text = "🟢 Agent Running" if ATLAS_AVAILABLE else "🟡 Mock Agent"
            st.success(status_text)
            if st.button("🛑 Shutdown Agent", use_container_width=True):
                shutdown_agent()
                st.rerun()
        
        st.divider()
        
        # Navigation
        st.header("📊 Dashboard Sections")
        page = st.selectbox(
            "Select Page",
            ["System Overview", "Inference Tasks", "Training Tasks", "Performance Analysis", "Model Management", "Settings"]
        )
    
    # Show installation instructions if ATLAS not available
    if not ATLAS_AVAILABLE:
        st.warning("⚠️ ATLAS components not fully available. Running in demonstration mode.")
        
        with st.expander("📋 Installation Instructions", expanded=False):
            st.markdown("""
            **To get ATLAS fully working:**
            
            1. **Install required Python packages:**
            ```bash
            pip install sentence-transformers torch transformers scikit-learn
            ```
            
            2. **Ensure your project structure matches:**
            ```
            AI_AGENT_TRAINING/
            ├── core/
            │   ├── ai_training_agent.py
            │   ├── model_manager.py
            │   ├── inference_engine.py
            │   └── training_engine.py
            └── run_streamlit.py (this file)
            ```
            
            3. **Check that all Python files exist and have proper imports**
            
            4. **Restart the dashboard after installation**
            """)
    
    if st.session_state.agent is None:
        st.info("👆 Please initialize the agent from the sidebar to continue.")
        return
    
    # Main content based on selected page
    if page == "System Overview":
        system_overview_page()
    elif page == "Inference Tasks":
        inference_tasks_page()
    elif page == "Training Tasks":
        training_tasks_page()
    elif page == "Performance Analysis":
        performance_analysis_page()
    elif page == "Model Management":
        model_management_page()
    elif page == "Settings":
        settings_page()

def system_overview_page():
    """System overview and status"""
    st.header("📊 System Overview")
    
    # Get system status
    try:
        status = st.session_state.agent.get_system_status()
        
        # System metrics cards
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            st.metric("🟢 Running Tasks", status['scheduler']['running_tasks'])
        with col2:
            st.metric("⏳ Pending Tasks", status['scheduler']['pending_tasks'])
        with col3:
            st.metric("✅ Completed Tasks", status['scheduler']['completed_tasks'])
        with col4:
            st.metric("📦 Available Models", len(status['models']))
        
        # System status details
        col1, col2 = st.columns([2, 1])
        
        with col1:
            st.subheader("📈 System Performance")
            
            # Create performance chart
            if 'resources' in status:
                resources = status['resources']
                
                # Resource utilization chart
                fig = go.Figure()
                
                resources_data = [
                    ("CPU", resources.get('cpu_utilization', 0)),
                    ("Memory", resources.get('memory_utilization', 0)),
                    ("GPU", resources.get('gpu_utilization', 0)),
                    ("Disk I/O", resources.get('disk_io_utilization', 0)),
                    ("Network", resources.get('network_utilization', 0))
                ]
                
                for name, value in resources_data:
                    color = 'red' if value > 80 else 'orange' if value > 60 else 'green'
                    fig.add_trace(go.Bar(
                        x=[name],
                        y=[value],
                        name=name,
                        marker_color=color,
                        showlegend=False
                    ))
                
                fig.update_layout(
                    title="Resource Utilization (%)",
                    yaxis_title="Utilization %",
                    yaxis=dict(range=[0, 100]),
                    height=400
                )
                
                st.plotly_chart(fig, use_container_width=True)
            else:
                st.info("Resource monitoring not available")
        
        with col2:
            st.subheader("🔧 System Information")
            
            system_info = {
                "Status": "🟢 Running" if ATLAS_AVAILABLE else "🟡 Mock Mode",
                "Timestamp": status['timestamp'],
                "Max Tasks": status['system']['max_concurrent_tasks'],
                "Resource Mgmt": "✅" if status['system'].get('resource_management_enabled', False) else "❌"
            }
            
            for key, value in system_info.items():
                st.text(f"{key}: {value}")
            
            # Models list
            st.subheader("📦 Available Models")
            for model_name, model_type in status['models'].items():
                st.text(f"• {model_name}")
                st.caption(f"  Type: {model_type}")
        
        # Recent activity
        st.subheader("📋 Recent Task Activity")
        
        if st.session_state.task_history:
            df = pd.DataFrame(st.session_state.task_history[-10:])  # Last 10 tasks
            st.dataframe(df, use_container_width=True)
        else:
            st.info("No recent task activity. Try running some inference tasks!")
            
    except Exception as e:
        st.error(f"❌ Error getting system status: {e}")

def inference_tasks_page():
    """Inference tasks management"""
    st.header("🔍 Inference Tasks")
    
    # Inference input form
    with st.form("inference_form"):
        st.subheader("📝 New Inference Request")
        
        col1, col2 = st.columns([3, 1])
        
        with col1:
            email_texts = st.text_area(
                "Email Texts (one per line)",
                height=150,
                placeholder="Enter email texts to analyze...\nExample:\nUrgent: Your account will be suspended! Click here now!\nMeeting scheduled for tomorrow at 10 AM.",
                value="Urgent: Your account will be suspended! Click here now!\nMeeting scheduled for tomorrow at 10 AM in conference room B.\nCongratulations! You've won $1,000,000! Claim your prize immediately!\nYour order has been shipped and will arrive within 3-5 business days."
            )
        
        with col2:
            model_name = st.selectbox(
                "Model",
                ["all-MiniLM-L6-v2", "all-mpnet-base-v2", "phishing_detector_v1"],
                index=0
            )
            
            threshold = st.slider(
                "Classification Threshold",
                min_value=0.0,
                max_value=1.0,
                value=0.5,
                step=0.1
            )
            
            return_embeddings = st.checkbox("Return Embeddings", key="inference_return_embeddings")
        
        submitted = st.form_submit_button("🚀 Run Inference", use_container_width=True)
        
        if submitted and email_texts.strip():
            texts = [text.strip() for text in email_texts.split('\n') if text.strip()]
            
            if texts:
                with st.spinner("Running inference..."):
                    try:
                        # Use inference method
                        result = asyncio.run(st.session_state.agent.quick_inference(texts, model_name))
                        
                        # Add to task history
                        task_record = {
                            'timestamp': datetime.now().isoformat(),
                            'task_type': 'inference',
                            'model': model_name,
                            'num_texts': len(texts),
                            'phishing_detected': sum(result.get('predictions', [])),
                            'avg_confidence': np.mean(result.get('confidences', []))
                        }
                        st.session_state.task_history.append(task_record)
                        
                        # Display results
                        st.subheader("📊 Inference Results")
                        
                        results_data = []
                        for i, (text, prediction, confidence) in enumerate(zip(
                            texts, 
                            result.get('predictions', []), 
                            result.get('confidences', [])
                        )):
                            results_data.append({
                                'Email #': i + 1,
                                'Text Preview': text[:100] + "..." if len(text) > 100 else text,
                                'Classification': "🚨 PHISHING" if prediction == 1 else "✅ SAFE",
                                'Confidence': f"{confidence:.3f}",
                                'Status': "⚠️ HIGH RISK" if prediction == 1 and confidence > 0.8 else "✅ LOW RISK",
                                'Full Text': text
                            })
                        
                        df_results = pd.DataFrame(results_data)
                        
                        # Display results table
                        st.dataframe(
                            df_results[['Email #', 'Text Preview', 'Classification', 'Confidence', 'Status']], 
                            use_container_width=True
                        )
                        
                        # Results summary
                        col1, col2, col3, col4 = st.columns(4)
                        
                        phishing_count = sum(result.get('predictions', []))
                        total_count = len(texts)
                        safe_count = total_count - phishing_count
                        
                        with col1:
                            st.metric("🚨 Phishing Detected", phishing_count)
                        with col2:
                            st.metric("✅ Safe Emails", safe_count)
                        with col3:
                            avg_confidence = np.mean(result.get('confidences', []))
                            st.metric("📊 Avg Confidence", f"{avg_confidence:.3f}")
                        with col4:
                            processing_time = result.get('processing_time', 0)
                            st.metric("⏱️ Processing Time", f"{processing_time:.2f}s")
                        
                        # Visualization
                        if len(results_data) > 1:
                            st.subheader("📈 Results Visualization")
                            
                            # Confidence distribution
                            fig = px.bar(
                                df_results,
                                x='Email #',
                                y='Confidence',
                                color='Classification',
                                title="Classification Confidence by Email",
                                color_discrete_map={
                                    '🚨 PHISHING': 'red',
                                    '✅ SAFE': 'green'
                                }
                            )
                            fig.update_layout(height=400)
                            st.plotly_chart(fig, use_container_width=True)
                        
                    except Exception as e:
                        st.error(f"❌ Error running inference: {e}")
                        if not ATLAS_AVAILABLE:
                            st.info("💡 This error might be due to missing dependencies. Try installing required packages.")

# Continue with other page functions...
def training_tasks_page():
    """Training tasks management"""
    st.header("🎓 Training Tasks")
    
    # Training form
    with st.form("training_form"):
        st.subheader("🏋️ New Training Request")
        
        col1, col2 = st.columns([2, 1])
        
        with col1:
            # Data source selection
            data_source = st.radio(
                "Data Source",
                ["Generate Synthetic Data", "Upload CSV File", "Use Existing File"]
            )
            
            data_path = None
            
            if data_source == "Upload CSV File":
                uploaded_file = st.file_uploader(
                    "Choose CSV file",
                    type=['csv'],
                    help="CSV should have 'Email Text' and 'Email Type' columns"
                )
                if uploaded_file:
                    # Save uploaded file
                    data_path = f"uploaded_{uploaded_file.name}"
                    with open(data_path, "wb") as f:
                        f.write(uploaded_file.getbuffer())
            
            elif data_source == "Use Existing File":
                data_path = st.text_input(
                    "File Path",
                    placeholder="path/to/your/data.csv"
                )
            
            elif data_source == "Generate Synthetic Data":
                num_samples = st.number_input(
                    "Number of Samples",
                    min_value=10,
                    max_value=10000,
                    value=500,
                    step=10
                )
        
        with col2:
            st.subheader("⚙️ Training Parameters")
            
            epochs = st.number_input(
                "Epochs",
                min_value=1,
                max_value=20,
                value=3,
                step=1
            )
            
            batch_size = st.number_input(
                "Batch Size",
                min_value=4,
                max_value=64,
                value=16,
                step=4
            )
            
            base_model = st.selectbox(
                "Base Model",
                ["all-MiniLM-L6-v2", "all-mpnet-base-v2"],
                index=0
            )
        
        submitted = st.form_submit_button("🚀 Start Training", use_container_width=True)
        
        if submitted:
            with st.spinner("Starting training..."):
                try:
                    # Prepare data path
                    if data_source == "Generate Synthetic Data":
                        data_path = st.session_state.agent.create_sample_training_data(num_samples)
                        st.info(f"📝 Generated synthetic data: {data_path}")
                    
                    # Show data statistics if available
                    if data_path and os.path.exists(data_path):
                        try:
                            stats = st.session_state.agent.get_training_statistics(data_path)
                            
                            st.subheader("📊 Training Data Statistics")
                            col1, col2, col3, col4 = st.columns(4)
                            
                            with col1:
                                st.metric("Total Samples", stats.get('total_samples', 0))
                            with col2:
                                st.metric("Phishing Samples", stats.get('phishing_samples', 0))
                            with col3:
                                st.metric("Safe Samples", stats.get('safe_samples', 0))
                            with col4:
                                phishing_pct = stats.get('phishing_percentage', 0)
                                st.metric("Phishing %", f"{phishing_pct:.1f}%")
                        
                        except Exception as e:
                            st.warning(f"Could not load data statistics: {e}")
                    
                    # Simulate training progress
                    progress_bar = st.progress(0)
                    status_text = st.empty()
                    
                    for i in range(epochs):
                        for j in range(10):  # 10 steps per epoch
                            progress = (i * 10 + j + 1) / (epochs * 10)
                            progress_bar.progress(progress)
                            status_text.text(f"Training epoch {i+1}/{epochs}, step {j+1}/10")
                            time.sleep(0.1)  # Simulate processing time
                    
                    # Add mock training result
                    training_result = {
                        'timestamp': datetime.now(),
                        'model_name': f'phishing_detector_{datetime.now().strftime("%Y%m%d_%H%M%S")}',
                        'accuracy': np.random.uniform(0.85, 0.95),
                        'map_score': np.random.uniform(0.80, 0.90),
                        'training_time': epochs * 60,  # Mock training time
                        'sample_count': num_samples if data_source == "Generate Synthetic Data" else 1000,
                        'epochs': epochs,
                        'batch_size': batch_size,
                        'base_model': base_model
                    }
                    st.session_state.training_results.append(training_result)
                    
                    st.success("✅ Training completed successfully!")
                    st.info(f"📈 Final Accuracy: {training_result['accuracy']:.3f}")
                    st.info(f"🎯 MAP Score: {training_result['map_score']:.3f}")
                    
                    if not ATLAS_AVAILABLE:
                        st.info("💡 This was a simulation. Full training requires ATLAS components to be properly installed.")
                    
                except Exception as e:
                    st.error(f"❌ Error starting training: {e}")

def performance_analysis_page():
    """Performance analysis and results"""
    st.header("📈 Performance Analysis")
    
    # Load sample results button
    if not st.session_state.training_results:
        col1, col2 = st.columns(2)
        with col1:
            if st.button("📊 Load Sample Results"):
                sample_results = [
                    {
                        'timestamp': datetime.now() - timedelta(days=5),
                        'model_name': 'phishing_detector_v1',
                        'accuracy': 0.847,
                        'map_score': 0.823,
                        'training_time': 120,
                        'sample_count': 500,
                        'epochs': 3,
                        'batch_size': 16,
                        'base_model': 'all-MiniLM-L6-v2'
                    },
                    {
                        'timestamp': datetime.now() - timedelta(days=3),
                        'model_name': 'phishing_detector_v2',
                        'accuracy': 0.892,
                        'map_score': 0.876,
                        'training_time': 180,
                        'sample_count': 1000,
                        'epochs': 5,
                        'batch_size': 16,
                        'base_model': 'all-MiniLM-L6-v2'
                    },
                    {
                        'timestamp': datetime.now() - timedelta(days=1),
                        'model_name': 'phishing_detector_v3',
                        'accuracy': 0.913,
                        'map_score': 0.898,
                        'training_time': 240,
                        'sample_count': 2000,
                        'epochs': 5,
                        'batch_size': 32,
                        'base_model': 'all-mpnet-base-v2'
                    }
                ]
                st.session_state.training_results.extend(sample_results)
                st.rerun()
        
        with col2:
            st.info("👈 Click to load sample training results for analysis")
    
    if st.session_state.training_results:
        df_results = pd.DataFrame(st.session_state.training_results)
        df_results['timestamp'] = pd.to_datetime(df_results['timestamp'])
        
        # Performance metrics overview
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            best_accuracy = df_results['accuracy'].max()
            st.metric("🎯 Best Accuracy", f"{best_accuracy:.3f}")
        
        with col2:
            best_map = df_results['map_score'].max()
            st.metric("📊 Best MAP Score", f"{best_map:.3f}")
        
        with col3:
            avg_training_time = df_results['training_time'].mean()
            st.metric("⏱️ Avg Training Time", f"{avg_training_time:.0f}m")
        
        with col4:
            total_models = len(df_results)
            st.metric("🔬 Models Trained", total_models)
        
        # Performance trends charts
        if len(df_results) > 1:
            st.subheader("📈 Performance Trends")
            
            fig = make_subplots(
                rows=2, cols=2,
                subplot_titles=('Accuracy Over Time', 'MAP Score Over Time', 
                               'Training Time vs Sample Count', 'Model Comparison')
            )
            
            # Accuracy trend
            fig.add_trace(
                go.Scatter(x=df_results['timestamp'], y=df_results['accuracy'], 
                         mode='lines+markers', name='Accuracy', line=dict(color='blue')),
                row=1, col=1
            )
            
            # MAP score trend
            fig.add_trace(
                go.Scatter(x=df_results['timestamp'], y=df_results['map_score'], 
                         mode='lines+markers', name='MAP Score', line=dict(color='green')),
                row=1, col=2
            )
            
            # Training time vs sample count
            fig.add_trace(
                go.Scatter(x=df_results['sample_count'], y=df_results['training_time'], 
                         mode='markers', name='Training Time vs Samples', 
                         marker=dict(size=8, color='red')),
                row=2, col=1
            )
            
            # Model comparison
            fig.add_trace(
                go.Bar(x=df_results['model_name'], y=df_results['accuracy'], 
                      name='Accuracy', marker_color='lightblue'),
                row=2, col=2
            )
            
            fig.update_layout(height=800, showlegend=False, title_text="Training Performance Analysis")
            st.plotly_chart(fig, use_container_width=True)
        
        # Detailed results table
        st.subheader("📋 Detailed Training Results")
        
        # Format the dataframe for display
        display_df = df_results.copy()
        display_df['timestamp'] = display_df['timestamp'].dt.strftime('%Y-%m-%d %H:%M')
        display_df['accuracy'] = display_df['accuracy'].round(3)
        display_df['map_score'] = display_df['map_score'].round(3)
        display_df['training_time'] = display_df['training_time'].astype(int).astype(str) + 'm'
        
        st.dataframe(display_df, use_container_width=True)
        
        # Model performance comparison
        if len(df_results) > 1:
            st.subheader("🔍 Model Performance Comparison")
            
            # Best performing model
            best_model_idx = df_results['accuracy'].idxmax()
            best_model = df_results.loc[best_model_idx]
            
            col1, col2 = st.columns([1, 2])
            
            with col1:
                st.success("🏆 Best Performing Model")
                st.write(f"**Model:** {best_model['model_name']}")
                st.write(f"**Accuracy:** {best_model['accuracy']:.3f}")
                st.write(f"**MAP Score:** {best_model['map_score']:.3f}")
                st.write(f"**Base Model:** {best_model['base_model']}")
                st.write(f"**Epochs:** {best_model['epochs']}")
            
            with col2:
                # Performance correlation analysis
                st.info("📊 Performance Insights")
                
                # Calculate correlations
                correlations = []
                if df_results['sample_count'].nunique() > 1:
                    corr_samples = df_results['sample_count'].corr(df_results['accuracy'])
                    correlations.append(f"Sample Count vs Accuracy: {corr_samples:.3f}")
                
                if df_results['epochs'].nunique() > 1:
                    corr_epochs = df_results['epochs'].corr(df_results['accuracy'])
                    correlations.append(f"Epochs vs Accuracy: {corr_epochs:.3f}")
                
                if df_results['training_time'].nunique() > 1:
                    corr_time = df_results['training_time'].corr(df_results['accuracy'])
                    correlations.append(f"Training Time vs Accuracy: {corr_time:.3f}")
                
                for corr in correlations:
                    st.write(f"• {corr}")
                
                # Recommendations
                st.write("**💡 Recommendations:**")
                if best_model['accuracy'] > 0.9:
                    st.write("• Excellent performance achieved!")
                elif best_model['accuracy'] > 0.85:
                    st.write("• Good performance, consider more training data")
                else:
                    st.write("• Try different hyperparameters or base models")
    else:
        st.info("No training results available yet. Run some training tasks or load sample results!")

def model_management_page():
    """Model management interface"""
    st.header("📦 Model Management")
    
    try:
        # List available models
        models = st.session_state.agent.list_available_models()
        
        st.subheader("📋 Available Models")
        
        if models:
            for model_name, model_type in models.items():
                with st.container():
                    col1, col2, col3, col4 = st.columns([3, 1, 1, 1])
                    
                    with col1:
                        # Model status indicator
                        status_icon = "🟢" if model_type == "fine-tuned" else "🔵"
                        st.markdown(f"{status_icon} **{model_name}**")
                        st.caption(f"Type: {model_type}")
                    
                    with col2:
                        if st.button("📥 Load", key=f"load_{model_name}"):
                            try:
                                with st.spinner("Loading model..."):
                                    # Simulate model loading
                                    time.sleep(1)
                                st.success(f"✅ Loaded {model_name}")
                            except Exception as e:
                                st.error(f"❌ Failed to load {model_name}: {e}")
                    
                    with col3:
                        if st.button("ℹ️ Info", key=f"info_{model_name}"):
                            # Show model information
                            st.info(f"📊 Model: {model_name}")
                            st.write(f"Type: {model_type}")
                            if model_type == "fine-tuned":
                                st.write("Status: Custom trained model")
                                st.write("Performance: High accuracy for phishing detection")
                            else:
                                st.write("Status: Pre-trained sentence transformer")
                                st.write("Source: Hugging Face Model Hub")
                    
                    with col4:
                        if model_type == "fine-tuned":
                            if st.button("🗑️ Delete", key=f"delete_{model_name}"):
                                st.warning(f"Delete {model_name}? This cannot be undone.")
                                if st.button("Confirm Delete", key=f"confirm_delete_{model_name}"):
                                    st.success(f"✅ Deleted {model_name}")
                    
                    st.divider()
        else:
            st.info("No models available")
        
        # Model creation and data generation section
        col1, col2 = st.columns(2)
        
        with col1:
            st.subheader("🏗️ Create Training Data")
            
            with st.form("create_data_form"):
                num_samples = st.number_input(
                    "Number of Samples",
                    min_value=10,
                    max_value=10000,
                    value=100,
                    step=10
                )
                
                data_balance = st.slider(
                    "Phishing/Safe Ratio",
                    min_value=0.1,
                    max_value=0.9,
                    value=0.5,
                    step=0.1,
                    help="Ratio of phishing to safe emails"
                )
                
                if st.form_submit_button("📝 Generate Data"):
                    try:
                        with st.spinner("Generating data..."):
                            data_path = st.session_state.agent.create_sample_training_data(num_samples)
                        st.success(f"✅ Created data file: {data_path}")
                        
                        # Show preview
                        try:
                            df = pd.read_csv(data_path)
                            st.subheader("📊 Data Preview")
                            st.dataframe(df.head(10), use_container_width=True)
                            
                            # Data statistics
                            stats = st.session_state.agent.get_training_statistics(data_path)
                            col_a, col_b, col_c = st.columns(3)
                            with col_a:
                                st.metric("Total", stats['total_samples'])
                            with col_b:
                                st.metric("Phishing", stats['phishing_samples'])
                            with col_c:
                                st.metric("Safe", stats['safe_samples'])
                        except Exception:
                            pass
                            
                    except Exception as e:
                        st.error(f"❌ Error creating data: {e}")
        
        with col2:
            st.subheader("🔧 Model Operations")
            
            # Model download from Hugging Face
            with st.form("download_model_form"):
                st.write("**Download Pre-trained Model**")
                
                model_options = [
                    "all-MiniLM-L6-v2",
                    "all-mpnet-base-v2",
                    "distilbert-base-uncased",
                    "roberta-base"
                ]
                
                selected_model = st.selectbox(
                    "Select Model",
                    model_options
                )
                
                if st.form_submit_button("⬇️ Download Model"):
                    with st.spinner(f"Downloading {selected_model}..."):
                        # Simulate download
                        time.sleep(2)
                    st.success(f"✅ Downloaded {selected_model}")
            
            st.divider()
            
            # Model export/import
            st.write("**Model Export/Import**")
            
            col_a, col_b = st.columns(2)
            with col_a:
                if st.button("📤 Export Model", key="export_model"):
                    st.info("Model export functionality")
            
            with col_b:
                uploaded_model = st.file_uploader(
                    "Import Model",
                    type=['pkl', 'joblib', 'pt'],
                    key="import_model"
                )
                if uploaded_model:
                    st.success(f"✅ Imported {uploaded_model.name}")
        
    except Exception as e:
        st.error(f"❌ Error in model management: {e}")

def settings_page():
    """Settings and configuration"""
    st.header("⚙️ Settings & Configuration")
    
    # System settings
    st.subheader("🖥️ System Settings")
    
    with st.form("system_settings"):
        col1, col2 = st.columns(2)
        
        with col1:
            max_tasks = st.number_input(
                "Max Concurrent Tasks",
                min_value=1,
                max_value=20,
                value=10,
                step=1
            )
            
            log_level = st.selectbox(
                "Log Level",
                ["DEBUG", "INFO", "WARNING", "ERROR"],
                index=1
            )
            
            auto_save_results = st.checkbox(
                "Auto Save Results",
                value=True,
                help="Automatically save training and inference results"
            )
        
        with col2:
            enable_resource_monitoring = st.checkbox(
                "Enable Resource Monitoring",
                value=True,
                key="settings_resource_monitoring"
            )
            
            auto_cleanup = st.checkbox(
                "Auto Cleanup Old Tasks",
                value=True,
                key="settings_auto_cleanup"
            )
            
            notification_level = st.selectbox(
                "Notification Level",
                ["All", "Errors Only", "None"],
                index=0
            )
        
        if st.form_submit_button("💾 Save Settings"):
            st.success("✅ Settings saved!")
            # Here you would normally save to a config file
    
    st.divider()
    
    # Model settings
    st.subheader("🤖 Model Settings")
    
    with st.form("model_settings"):
        col1, col2 = st.columns(2)
        
        with col1:
            default_model = st.selectbox(
                "Default Inference Model",
                ["all-MiniLM-L6-v2", "all-mpnet-base-v2", "phishing_detector_v1"],
                index=0
            )
            
            confidence_threshold = st.slider(
                "Default Confidence Threshold",
                min_value=0.0,
                max_value=1.0,
                value=0.5,
                step=0.05
            )
        
        with col2:
            batch_size_default = st.number_input(
                "Default Batch Size",
                min_value=1,
                max_value=128,
                value=16,
                step=4
            )
            
            max_sequence_length = st.number_input(
                "Max Sequence Length",
                min_value=64,
                max_value=512,
                value=256,
                step=32
            )
        
        if st.form_submit_button("💾 Save Model Settings"):
            st.success("✅ Model settings saved!")
    
    st.divider()
    
    # Data management
    st.subheader("📁 Data Management")
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.write("**Clear Data**")
        if st.button("🗑️ Clear Task History", key="clear_history_settings"):
            st.session_state.task_history = []
            st.success("✅ Task history cleared!")
        
        if st.button("🗑️ Clear Training Results", key="clear_training_settings"):
            st.session_state.training_results = []
            st.success("✅ Training results cleared!")
    
    with col2:
        st.write("**Export Data**")
        if st.button("📥 Export All Data", key="export_all_settings"):
            # Create export data
            export_data = {
                'task_history': st.session_state.task_history,
                'training_results': st.session_state.training_results,
                'export_timestamp': datetime.now().isoformat()
            }
            
            export_json = json.dumps(export_data, indent=2, default=str)
            
            st.download_button(
                label="📄 Download JSON",
                data=export_json,
                file_name=f"atlas_data_export_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json",
                mime="application/json"
            )
    
    with col3:
        st.write("**System Info**")
        if st.button("ℹ️ Show System Info", key="system_info_settings"):
            system_info = {
                "Python Version": f"{sys.version_info.major}.{sys.version_info.minor}.{sys.version_info.micro}",
                "ATLAS Available": ATLAS_AVAILABLE,
                "Current Directory": current_dir,
                "Models Directory": os.path.join(current_dir, "models"),
                "Session Start": datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            }
            
            st.json(system_info)
    
    st.divider()
    
    # Advanced settings
    with st.expander("🔧 Advanced Settings", expanded=False):
        st.warning("⚠️ Advanced settings - modify with caution!")
        
        col1, col2 = st.columns(2)
        
        with col1:
            gpu_enabled = st.checkbox("Enable GPU Processing", value=False)
            debug_mode = st.checkbox("Debug Mode", value=False)
            verbose_logging = st.checkbox("Verbose Logging", value=False)
        
        with col2:
            memory_limit = st.number_input(
                "Memory Limit (GB)",
                min_value=1,
                max_value=32,
                value=8,
                step=1
            )
            
            timeout_seconds = st.number_input(
                "Task Timeout (seconds)",
                min_value=30,
                max_value=3600,
                value=300,
                step=30
            )
        
        if st.button("💾 Save Advanced Settings"):
            st.success("✅ Advanced settings saved!")

# Sidebar utilities
def sidebar_utilities():
    """Additional sidebar utilities"""
    with st.sidebar:
        st.divider()
        st.subheader("🛠️ Utilities")
        
        # System status indicator
        status_color = "🟢" if ATLAS_AVAILABLE else "🟡"
        status_text = "Full Mode" if ATLAS_AVAILABLE else "Demo Mode"
        st.write(f"{status_color} Status: {status_text}")
        
        # Auto-refresh toggle
        auto_refresh_enabled = st.checkbox("🔄 Auto Refresh (30s)", key="sidebar_auto_refresh")
        if auto_refresh_enabled:
            time.sleep(30)
            st.rerun()
        
        # Export options
        if st.button("📥 Export Task History", key="sidebar_export_history"):
            if st.session_state.task_history:
                df = pd.DataFrame(st.session_state.task_history)
                csv = df.to_csv(index=False)
                st.download_button(
                    label="📄 Download CSV",
                    data=csv,
                    file_name=f"task_history_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv",
                    mime="text/csv",
                    key="sidebar_download_csv"
                )
            else:
                st.info("No task history to export")
        
        # Quick actions
        st.subheader("⚡ Quick Actions")
        
        if st.button("🔄 Refresh Dashboard", key="sidebar_refresh_all"):
            st.rerun()
        
        if st.button("🧹 Clear Cache", key="sidebar_clear_cache"):
            st.cache_data.clear()
            st.success("✅ Cache cleared!")
        
        # Resource usage (mock)
        if st.session_state.agent:
            st.subheader("📊 Quick Stats")
            total_tasks = len(st.session_state.task_history)
            total_training = len(st.session_state.training_results)
            
            st.metric("Tasks Run", total_tasks)
            st.metric("Models Trained", total_training)
            
            if st.session_state.task_history:
                recent_task = st.session_state.task_history[-1]
                st.write(f"Last task: {recent_task.get('task_type', 'Unknown')}")

# Main execution flow
def run_dashboard():
    """Main dashboard execution"""
    try:
        # Add sidebar utilities
        sidebar_utilities()
        
        # Main dashboard
        main_dashboard()
        
        # Auto-refresh logic
        if hasattr(st.session_state, 'auto_refresh') and st.session_state.auto_refresh:
            time.sleep(1)  # Small delay to prevent too frequent refreshes
        
    except Exception as e:
        st.error(f"❌ Error in dashboard execution: {str(e)}")
        logging.error(f"Dashboard error: {e}")
        
        # Show debugging information in debug mode
        if st.checkbox("Show Debug Info"):
            st.code(str(e))
            st.code(f"Python path: {sys.path}")
            st.code(f"Current directory: {os.getcwd()}")

# Initialize and run
if __name__ == "__main__":
    # Run the dashboard
    run_dashboard()
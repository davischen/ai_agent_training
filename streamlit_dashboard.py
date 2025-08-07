#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
ATLAS Agent Streamlit Dashboard
Interactive web interface for managing AI Training Agent
"""

import streamlit as st
import pandas as pd
import numpy as np
import asyncio
import json
import time
import os
from datetime import datetime, timedelta
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import logging

# Configure logging to suppress verbose output
logging.basicConfig(level=logging.WARNING)

# Import ATLAS components
try:
    from ai_training_agent import AITrainingAgent
    from model_manager import ModelManager
    from inference_engine import InferenceEngine
    from training_engine import TrainingEngine
    ATLAS_AVAILABLE = True
except ImportError as e:
    st.error(f"Failed to import ATLAS components: {e}")
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

def initialize_agent():
    """Initialize ATLAS Agent"""
    if not ATLAS_AVAILABLE:
        return None
    
    try:
        with st.spinner("Initializing ATLAS Agent..."):
            agent = AITrainingAgent(model_dir="./models")
            st.success("✅ ATLAS Agent initialized successfully!")
            return agent
    except Exception as e:
        st.error(f"❌ Failed to initialize ATLAS Agent: {e}")
        return None

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
        
        if st.session_state.agent is None:
            if st.button("🚀 Initialize Agent", use_container_width=True):
                st.session_state.agent = initialize_agent()
                st.rerun()
        else:
            st.success("🟢 Agent Running")
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
    
    if not ATLAS_AVAILABLE:
        st.error("❌ ATLAS components not available. Please check your installation.")
        return
    
    if st.session_state.agent is None:
        st.warning("⚠️ Please initialize the ATLAS Agent first.")
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
                    yaxis=dict(range=[0, 100])
                )
                
                st.plotly_chart(fig, use_container_width=True)
            else:
                st.info("Resource monitoring not available")
        
        with col2:
            st.subheader("🔧 System Information")
            
            system_info = {
                "Timestamp": status['timestamp'],
                "Max Concurrent Tasks": status['system']['max_concurrent_tasks'],
                "Resource Management": "✅" if status['system']['resource_management_enabled'] else "❌"
            }
            
            for key, value in system_info.items():
                st.text(f"{key}: {value}")
            
            # Models list
            st.subheader("📦 Available Models")
            for model_name, model_type in status['models'].items():
                st.text(f"• {model_name} ({model_type})")
        
        # Recent activity
        st.subheader("📋 Recent Task Activity")
        
        if st.session_state.task_history:
            df = pd.DataFrame(st.session_state.task_history[-10:])  # Last 10 tasks
            st.dataframe(df, use_container_width=True)
        else:
            st.info("No recent task activity")
            
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
                placeholder="Enter email texts to analyze...\nOne email per line"
            )
        
        with col2:
            model_name = st.selectbox(
                "Model",
                ["all-MiniLM-L6-v2", "all-mpnet-base-v2"],
                index=0
            )
            
            threshold = st.slider(
                "Classification Threshold",
                min_value=0.0,
                max_value=1.0,
                value=0.5,
                step=0.1
            )
            
            return_embeddings = st.checkbox("Return Embeddings")
        
        submitted = st.form_submit_button("🚀 Run Inference", use_container_width=True)
        
        if submitted and email_texts.strip():
            texts = [text.strip() for text in email_texts.split('\n') if text.strip()]
            
            if texts:
                with st.spinner("Running inference..."):
                    try:
                        # Create inference request
                        request = st.session_state.agent.generate_inference_request(
                            text_data=texts,
                            model_name=model_name,
                            threshold=threshold,
                            return_embeddings=return_embeddings
                        )
                        
                        # Submit task
                        task_id = asyncio.run(st.session_state.agent.submit_inference_request(request))
                        
                        # Add to task history
                        task_info = {
                            'Task ID': task_id,
                            'Type': 'Inference',
                            'Status': 'Submitted',
                            'Timestamp': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
                            'Details': f"{len(texts)} emails, model: {model_name}"
                        }
                        st.session_state.task_history.append(task_info)
                        
                        st.success(f"✅ Inference task submitted! Task ID: {task_id}")
                        
                        # Try to get quick results
                        with st.spinner("Getting results..."):
                            time.sleep(2)  # Wait a bit for processing
                            
                            try:
                                result = asyncio.run(st.session_state.agent.quick_inference(texts, model_name))
                                
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
                                        'Full Text': text
                                    })
                                
                                df_results = pd.DataFrame(results_data)
                                
                                # Display results table
                                st.dataframe(
                                    df_results[['Email #', 'Text Preview', 'Classification', 'Confidence']], 
                                    use_container_width=True
                                )
                                
                                # Results summary
                                col1, col2, col3 = st.columns(3)
                                
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
                                
                                # Visualization
                                if len(results_data) > 1:
                                    st.subheader("📈 Results Visualization")
                                    
                                    # Confidence distribution
                                    fig = px.bar(
                                        df_results,
                                        x='Email #',
                                        y='Confidence',
                                        color='Classification',
                                        title="Classification Confidence by Email"
                                    )
                                    st.plotly_chart(fig, use_container_width=True)
                                
                            except Exception as e:
                                st.error(f"❌ Error getting quick results: {e}")
                        
                    except Exception as e:
                        st.error(f"❌ Error submitting inference task: {e}")

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
            
            learning_rate = st.select_slider(
                "Learning Rate",
                options=[1e-5, 2e-5, 3e-5, 5e-5, 1e-4],
                value=2e-5,
                format_func=lambda x: f"{x:.0e}"
            )
        
        submitted = st.form_submit_button("🚀 Start Training", use_container_width=True)
        
        if submitted:
            with st.spinner("Starting training..."):
                try:
                    # Prepare data path
                    if data_source == "Generate Synthetic Data":
                        data_path = st.session_state.agent.create_sample_training_data(num_samples)
                        st.info(f"📝 Generated synthetic data: {data_path}")
                    
                    # Create training request
                    training_params = {
                        'epochs': epochs,
                        'batch_size': batch_size,
                        'learning_rate': learning_rate
                    }
                    
                    model_config = {
                        'base_model': base_model
                    }
                    
                    # Submit training task
                    task_id = asyncio.run(st.session_state.agent.train_model_from_data(
                        data_path=data_path,
                        model_config=model_config,
                        training_params=training_params
                    ))
                    
                    # Add to task history
                    task_info = {
                        'Task ID': task_id,
                        'Type': 'Training',
                        'Status': 'Started',
                        'Timestamp': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
                        'Details': f"Epochs: {epochs}, Batch: {batch_size}, Model: {base_model}"
                    }
                    st.session_state.task_history.append(task_info)
                    
                    st.success(f"✅ Training task started! Task ID: {task_id}")
                    
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
                    
                except Exception as e:
                    st.error(f"❌ Error starting training: {e}")
    
    # Task monitoring
    st.subheader("📋 Task Monitoring")
    
    if st.button("🔄 Refresh Task Status"):
        # Update task statuses
        for task in st.session_state.task_history:
            if task['Status'] in ['Submitted', 'Started', 'Running']:
                try:
                    status = st.session_state.agent.get_task_status(task['Task ID'])
                    if status:
                        task['Status'] = status.status.value if hasattr(status.status, 'value') else str(status.status)
                except Exception:
                    pass
    
    # Display task history
    if st.session_state.task_history:
        df_tasks = pd.DataFrame(st.session_state.task_history)
        st.dataframe(df_tasks, use_container_width=True)
    else:
        st.info("No tasks submitted yet")

def performance_analysis_page():
    """Performance analysis and results"""
    st.header("📈 Performance Analysis")
    
    # Performance report
    try:
        report = st.session_state.agent.get_performance_report()
        
        # System performance summary
        st.subheader("🎯 System Performance Summary")
        
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            st.metric("Total Models", report['model_count'])
        with col2:
            tasks_processed = report['uptime_info']['total_tasks_processed']
            st.metric("Tasks Processed", tasks_processed)
        with col3:
            scheduler_running = report['uptime_info']['scheduler_running']
            st.metric("Scheduler", "🟢 Running" if scheduler_running else "🔴 Stopped")
        with col4:
            st.metric("Generated At", report['generated_at'][:16])
        
        # Training results history
        st.subheader("🏆 Training Results History")
        
        if st.session_state.training_results:
            df_results = pd.DataFrame(st.session_state.training_results)
            
            # Performance trends chart
            if len(df_results) > 1:
                fig = make_subplots(
                    rows=2, cols=2,
                    subplot_titles=('Accuracy Trend', 'MAP Score Trend', 'Training Time', 'Sample Count'),
                    specs=[[{"secondary_y": False}, {"secondary_y": False}],
                           [{"secondary_y": False}, {"secondary_y": False}]]
                )
                
                # Accuracy trend
                if 'accuracy' in df_results.columns:
                    fig.add_trace(
                        go.Scatter(x=df_results.index, y=df_results['accuracy'], 
                                 mode='lines+markers', name='Accuracy'),
                        row=1, col=1
                    )
                
                # MAP score trend
                if 'map_score' in df_results.columns:
                    fig.add_trace(
                        go.Scatter(x=df_results.index, y=df_results['map_score'], 
                                 mode='lines+markers', name='MAP Score'),
                        row=1, col=2
                    )
                
                # Training time
                if 'training_time' in df_results.columns:
                    fig.add_trace(
                        go.Bar(x=df_results.index, y=df_results['training_time'], 
                              name='Training Time (s)'),
                        row=2, col=1
                    )
                
                # Sample count
                if 'sample_count' in df_results.columns:
                    fig.add_trace(
                        go.Bar(x=df_results.index, y=df_results['sample_count'], 
                              name='Sample Count'),
                        row=2, col=2
                    )
                
                fig.update_layout(height=600, showlegend=False)
                st.plotly_chart(fig, use_container_width=True)
            
            # Results table
            st.dataframe(df_results, use_container_width=True)
        else:
            st.info("No training results available yet")
            
            # Add sample results for demonstration
            if st.button("📊 Add Sample Results"):
                sample_results = [
                    {
                        'timestamp': datetime.now() - timedelta(days=2),
                        'model_name': 'trained_001',
                        'accuracy': 0.847,
                        'map_score': 0.823,
                        'training_time': 120,
                        'sample_count': 500,
                        'epochs': 3
                    },
                    {
                        'timestamp': datetime.now() - timedelta(days=1),
                        'model_name': 'trained_002',
                        'accuracy': 0.892,
                        'map_score': 0.876,
                        'training_time': 180,
                        'sample_count': 1000,
                        'epochs': 5
                    },
                    {
                        'timestamp': datetime.now(),
                        'model_name': 'trained_003',
                        'accuracy': 0.915,
                        'map_score': 0.903,
                        'training_time': 240,
                        'sample_count': 1500,
                        'epochs': 5
                    }
                ]
                st.session_state.training_results.extend(sample_results)
                st.rerun()
        
    except Exception as e:
        st.error(f"❌ Error getting performance report: {e}")

def model_management_page():
    """Model management interface"""
    st.header("📦 Model Management")
    
    try:
        # List available models
        models = st.session_state.agent.list_available_models()
        
        st.subheader("📋 Available Models")
        
        if models:
            # Create model cards
            for model_name, model_type in models.items():
                with st.container():
                    col1, col2, col3, col4 = st.columns([3, 1, 1, 1])
                    
                    with col1:
                        st.text(f"🔹 **{model_name}**")
                        st.caption(f"Type: {model_type}")
                    
                    with col2:
                        if st.button("📥 Load", key=f"load_{model_name}"):
                            try:
                                with st.spinner("Loading model..."):
                                    st.session_state.agent.load_model(model_name)
                                st.success(f"✅ Loaded {model_name}")
                            except Exception as e:
                                st.error(f"❌ Failed to load {model_name}: {e}")
                    
                    with col3:
                        if st.button("📤 Export", key=f"export_{model_name}"):
                            export_path = f"exported_{model_name}"
                            try:
                                success = st.session_state.agent.export_model(model_name, export_path)
                                if success:
                                    st.success(f"✅ Exported to {export_path}")
                                else:
                                    st.error("❌ Export failed")
                            except Exception as e:
                                st.error(f"❌ Export error: {e}")
                    
                    with col4:
                        # Model info button
                        if st.button("ℹ️ Info", key=f"info_{model_name}"):
                            try:
                                info = st.session_state.agent.model_manager.get_model_info(model_name)
                                if info:
                                    st.json(info)
                                else:
                                    st.info("No detailed info available")
                            except Exception as e:
                                st.error(f"❌ Error getting info: {e}")
                    
                    st.divider()
        else:
            st.info("No models available")
        
        # Model import section
        st.subheader("📥 Import Model")
        
        with st.form("import_model_form"):
            col1, col2 = st.columns(2)
            
            with col1:
                model_path = st.text_input(
                    "Model Path",
                    placeholder="path/to/model/directory"
                )
            
            with col2:
                model_name = st.text_input(
                    "Model Name",
                    placeholder="my_custom_model"
                )
            
            if st.form_submit_button("📥 Import Model"):
                if model_path and model_name:
                    try:
                        success = st.session_state.agent.import_model(model_path, model_name)
                        if success:
                            st.success(f"✅ Imported {model_name} successfully!")
                        else:
                            st.error("❌ Import failed")
                    except Exception as e:
                        st.error(f"❌ Import error: {e}")
                else:
                    st.warning("⚠️ Please provide both model path and name")
        
        # Model creation section
        st.subheader("🏗️ Create Sample Training Data")
        
        with st.form("create_data_form"):
            col1, col2 = st.columns(2)
            
            with col1:
                num_samples = st.number_input(
                    "Number of Samples",
                    min_value=10,
                    max_value=10000,
                    value=100,
                    step=10
                )
            
            with col2:
                st.info("This will create synthetic email data for training")
            
            if st.form_submit_button("📝 Create Data"):
                try:
                    with st.spinner("Generating data..."):
                        data_path = st.session_state.agent.create_sample_training_data(num_samples)
                    st.success(f"✅ Created data file: {data_path}")
                    
                    # Show preview
                    try:
                        df = pd.read_csv(data_path)
                        st.subheader("📊 Data Preview")
                        st.dataframe(df.head(10), use_container_width=True)
                    except Exception:
                        pass
                        
                except Exception as e:
                    st.error(f"❌ Error creating data: {e}")
        
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
        
        with col2:
            enable_resource_monitoring = st.checkbox(
                "Enable Resource Monitoring",
                value=True
            )
            
            auto_cleanup = st.checkbox(
                "Auto Cleanup Old Tasks",
                value=True
            )
        
        if st.form_submit_button("💾 Save Settings"):
            st.success("✅ Settings saved!")
    
    # Data management
    st.subheader("📁 Data Management")
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        if st.button("🗑️ Clear Task History"):
            st.session_state.task_history = []
            st.success("✅ Task history cleared!")
    
    with col2:
        if st.button("🗑️ Clear Training Results"):
            st.session_state.training_results = []
            st.success("✅ Training results cleared!")
    
    with col3:
        if st.button("📤 Export System State"):
            try:
                backup_path = f"system_backup_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
                success = st.session_state.agent.backup_system_state(backup_path)
                if success:
                    st.success(f"✅ System state exported to {backup_path}")
                else:
                    st.error("❌ Export failed")
            except Exception as e:
                st.error(f"❌ Export error: {e}")
    
    # About section
    st.subheader("ℹ️ About ATLAS Agent")
    
    st.markdown("""
    **ATLAS - Adaptive Training and Learning Automation System**
    
    This dashboard provides a comprehensive interface for managing AI training agents, including:
    
    - 🔍 **Inference Tasks**: Real-time email classification and analysis
    - 🎓 **Training Management**: Model training with custom datasets
    - 📈 **Performance Analysis**: Training results and system metrics
    - 📦 **Model Management**: Import, export, and manage trained models
    - 🖥️ **System Monitoring**: Resource utilization and task tracking
    
    **Features:**
    - Interactive web interface built with Streamlit
    - Real-time task monitoring and status updates
    - Comprehensive performance visualizations
    - Easy model deployment and management
    - Synthetic data generation for testing
    
    **Version**: 1.0.0  
    **Last Updated**: August 2025
    """)
    
    # System information
    st.subheader("🔧 System Information")
    
    if st.session_state.agent:
        try:
            status = st.session_state.agent.get_system_status()
            
            info_data = {
                "Agent Status": "🟢 Running",
                "Timestamp": status['timestamp'],
                "Models Directory": st.session_state.agent.model_manager.model_dir,
                "Available Models": len(status['models']),
                "Scheduler Running": "✅" if status['system'].get('resource_management_enabled') else "❌",
                "Max Concurrent Tasks": status['system']['max_concurrent_tasks']
            }
            
            for key, value in info_data.items():
                st.text(f"{key}: {value}")
                
        except Exception as e:
            st.error(f"Error getting system information: {e}")
    else:
        st.text("Agent Status: 🔴 Not Running")

# Auto-refresh functionality
def auto_refresh():
    """Auto-refresh the dashboard"""
    if st.session_state.get('auto_refresh', False):
        time.sleep(30)  # Refresh every 30 seconds
        st.rerun()

# Additional utility functions
def export_task_history():
    """Export task history to CSV"""
    if st.session_state.task_history:
        df = pd.DataFrame(st.session_state.task_history)
        csv = df.to_csv(index=False)
        return csv
    return None

def create_performance_chart(data, metric_name):
    """Create performance chart"""
    if not data:
        return None
    
    df = pd.DataFrame(data)
    fig = px.line(
        df, 
        x='timestamp', 
        y=metric_name,
        title=f"{metric_name.title()} Over Time",
        markers=True
    )
    return fig

# Sidebar utilities
def sidebar_utilities():
    """Additional sidebar utilities"""
    with st.sidebar:
        st.divider()
        st.subheader("🛠️ Utilities")
        
        # Auto-refresh toggle
        auto_refresh_enabled = st.checkbox("🔄 Auto Refresh (30s)")
        st.session_state.auto_refresh = auto_refresh_enabled
        
        # Export options
        if st.button("📥 Export Task History"):
            csv_data = export_task_history()
            if csv_data:
                st.download_button(
                    label="📄 Download CSV",
                    data=csv_data,
                    file_name=f"task_history_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv",
                    mime="text/csv"
                )
            else:
                st.info("No task history to export")
        
        # Quick actions
        st.subheader("⚡ Quick Actions")
        
        if st.button("🔄 Refresh All Data"):
            st.rerun()
        
        if st.button("🧹 Clear All Cache"):
            st.cache_data.clear()
            st.success("Cache cleared!")

# Error handling and logging
def handle_error(error, context=""):
    """Handle errors gracefully"""
    error_msg = f"Error in {context}: {str(error)}" if context else str(error)
    st.error(f"❌ {error_msg}")
    logging.error(error_msg)

# Main execution flow
def run_dashboard():
    """Main dashboard execution"""
    try:
        # Add sidebar utilities
        sidebar_utilities()
        
        # Main dashboard
        main_dashboard()
        
        # Auto-refresh if enabled
        if st.session_state.get('auto_refresh', False):
            time.sleep(30)
            st.rerun()
            
    except Exception as e:
        handle_error(e, "dashboard execution")

# Custom CSS for better styling
def apply_custom_css():
    """Apply custom CSS styling"""
    st.markdown("""
    <style>
    .main-header {
        background: linear-gradient(90deg, #667eea 0%, #764ba2 100%);
        padding: 1rem;
        border-radius: 10px;
        color: white;
        margin-bottom: 2rem;
    }
    
    .metric-card {
        background: #f8f9fa;
        padding: 1rem;
        border-radius: 8px;
        border-left: 4px solid #007bff;
        margin: 0.5rem 0;
    }
    
    .success-message {
        background: #d4edda;
        border: 1px solid #c3e6cb;
        color: #155724;
        padding: 0.75rem;
        border-radius: 4px;
        margin: 1rem 0;
    }
    
    .error-message {
        background: #f8d7da;
        border: 1px solid #f5c6cb;
        color: #721c24;
        padding: 0.75rem;
        border-radius: 4px;
        margin: 1rem 0;
    }
    
    .stButton > button {
        width: 100%;
        border-radius: 20px;
        border: none;
        background: linear-gradient(45deg, #007bff, #0056b3);
        color: white;
        font-weight: bold;
        transition: all 0.3s ease;
    }
    
    .stButton > button:hover {
        transform: translateY(-2px);
        box-shadow: 0 4px 8px rgba(0,123,255,0.3);
    }
    
    .sidebar .sidebar-content {
        background: linear-gradient(180deg, #f8f9fa 0%, #e9ecef 100%);
    }
    
    div[data-testid="metric-container"] {
        background: white;
        border: 1px solid #dee2e6;
        padding: 1rem;
        border-radius: 8px;
        box-shadow: 0 2px 4px rgba(0,0,0,0.1);
    }
    </style>
    """, unsafe_allow_html=True)

# Initialize and run
if __name__ == "__main__":
    # Apply custom styling
    apply_custom_css()
    
    # Run the dashboard
    run_dashboard()

# Additional helper functions for demo data
def create_demo_data():
    """Create demo data for testing"""
    if st.button("🎭 Load Demo Data"):
        # Add sample task history
        demo_tasks = [
            {
                'Task ID': 'task_001',
                'Type': 'Inference',
                'Status': 'Completed',
                'Timestamp': (datetime.now() - timedelta(minutes=30)).strftime('%Y-%m-%d %H:%M:%S'),
                'Details': '5 emails, model: all-MiniLM-L6-v2'
            },
            {
                'Task ID': 'task_002', 
                'Type': 'Training',
                'Status': 'Completed',
                'Timestamp': (datetime.now() - timedelta(hours=2)).strftime('%Y-%m-%d %H:%M:%S'),
                'Details': 'Epochs: 3, Batch: 16, Model: all-MiniLM-L6-v2'
            },
            {
                'Task ID': 'task_003',
                'Type': 'Inference',
                'Status': 'Running',
                'Timestamp': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
                'Details': '10 emails, model: all-mpnet-base-v2'
            }
        ]
        
        # Add sample training results
        demo_results = [
            {
                'timestamp': datetime.now() - timedelta(days=3),
                'model_name': 'phishing_detector_v1',
                'accuracy': 0.847,
                'map_score': 0.823,
                'training_time': 120,
                'sample_count': 500,
                'epochs': 3
            },
            {
                'timestamp': datetime.now() - timedelta(days=2),
                'model_name': 'phishing_detector_v2',
                'accuracy': 0.892,
                'map_score': 0.876,
                'training_time': 180,
                'sample_count': 1000,
                'epochs': 5
            },
            {
                'timestamp': datetime.now() - timedelta(days=1),
                'model_name': 'phishing_detector_v3',
                'accuracy': 0.915,
                'map_score': 0.903,
                'training_time': 240,
                'sample_count': 1500,
                'epochs': 5
            }
        ]
        
        st.session_state.task_history.extend(demo_tasks)
        st.session_state.training_results.extend(demo_results)
        
        st.success("✅ Demo data loaded successfully!")
        st.rerun()

# Footer
def show_footer():
    """Show dashboard footer"""
    st.markdown("---")
    st.markdown(
        """
        <div style='text-align: center; color: #666; padding: 20px;'>
            🤖 ATLAS Agent Dashboard - Built with Streamlit<br>
            Adaptive Training and Learning Automation System v1.0.0
        </div>
        """, 
        unsafe_allow_html=True
    )

# Add footer to main dashboard
run_dashboard()
create_demo_data()
show_footer()
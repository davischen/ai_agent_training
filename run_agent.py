#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
ATLAS Agent Main Execution Script
Run the AI Training Agent with automatic execution capabilities
"""

import os
import sys
import asyncio
import logging
import json
import argparse
from pathlib import Path

# Add project root to Python path
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))

from atlas.core.ai_training_agent import AITrainingAgent
from atlas.core.auto_execution import AutomaticTrainingOrchestrator

def setup_logging(log_level: str = "INFO"):
    """Setup logging configuration"""
    log_dir = project_root / "atlas" / "logs"
    log_dir.mkdir(exist_ok=True)
    
    logging.basicConfig(
        level=getattr(logging, log_level.upper()),
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler(log_dir / "atlas_agent.log"),
            logging.StreamHandler(sys.stdout)
        ]
    )

def load_config(config_path: str) -> dict:
    """Load configuration from JSON file"""
    with open(config_path, 'r', encoding='utf-8') as f:
        return json.load(f)

async def run_basic_agent():
    """Run basic AI Training Agent"""
    print("🚀 Starting ATLAS Agent in Basic Mode...")
    
    config_path = project_root / "atlas" / "config" / "default_config.json"
    config = load_config(config_path)
    
    # Initialize agent
    agent = AITrainingAgent(
        model_dir=config['system']['model_dir']
    )
    
    print("✅ ATLAS Agent initialized successfully!")
    
    # Example inference
    sample_emails = [
        "Urgent: Your account will be suspended unless you verify immediately!",
        "Hi team, please find the quarterly report attached for your review.",
        "WINNER! You've won $1,000,000! Click here to claim your prize now!"
    ]
    
    print("\n📧 Processing sample emails...")
    
    # Create inference request
    inference_request = agent.generate_inference_request(
        text_data=sample_emails,
        return_embeddings=False,
        threshold=0.5
    )
    
    # Submit and wait for results
    task_id = await agent.submit_inference_request(inference_request)
    print(f"📋 Inference task submitted: {task_id}")
    
    # Monitor task
    for i in range(30):  # Wait up to 30 seconds
        await asyncio.sleep(1)
        status = agent.get_task_status(task_id)
        if status and status.status.value in ['completed', 'failed']:
            break
    
    final_status = agent.get_task_status(task_id)
    if final_status and final_status.result:
        print("\n📊 Inference Results:")
        for i, (email, prediction) in enumerate(zip(sample_emails, final_status.result.get('predictions', []))):
            label = "🚨 PHISHING" if prediction == 1 else "✅ SAFE"
            print(f"  {i+1}. {label} - {email[:50]}...")
    
    # Graceful shutdown
    agent.shutdown()
    print("\n🏁 ATLAS Agent stopped.")

async def run_auto_training_agent():
    """Run automatic training agent"""
    print("🤖 Starting ATLAS Agent in Automatic Training Mode...")
    
    config_path = project_root / "atlas" / "config" / "default_config.json"
    config = load_config(config_path)
    
    # Initialize automatic training orchestrator
    orchestrator = AutomaticTrainingOrchestrator(config)
    
    print("✅ Automatic Training Orchestrator initialized!")
    print("📡 Starting monitoring for automatic training triggers...")
    print("⏰ Monitoring schedule:")
    print("  - Data collection: Every 10 minutes")
    print("  - Training triggers: Every 30 minutes")
    print("  - Daily training: 02:00 AM")
    print("\n📝 Press Ctrl+C to stop...\n")
    
    try:
        # Start monitoring
        await orchestrator.start_monitoring()
    except KeyboardInterrupt:
        print("\n🛑 Stopping automatic training...")
        orchestrator.stop_monitoring()
        print("🏁 ATLAS Agent stopped.")

def initialize_data_structure():
    """Initialize data directory structure"""
    print("🏗️  Initializing ATLAS Agent data structure...")
    
    directories = [
        "atlas/data/raw/emails",
        "atlas/data/raw/threat_intel", 
        "atlas/data/raw/user_feedback",
        "atlas/data/processed",
        "atlas/models/base_models",
        "atlas/models/trained_models",
        "atlas/logs",
        "atlas/config"
    ]
    
    for directory in directories:
        path = project_root / directory
        path.mkdir(parents=True, exist_ok=True)
        print(f"  ✅ Created: {directory}")
    
    # Create sample config if not exists
    config_path = project_root / "atlas" / "config" / "default_config.json"
    if not config_path.exists():
        print("  📝 Creating default configuration...")
        # Config creation logic would go here
    
    print("🎉 Data structure initialized successfully!")

def load_initial_data():
    """Load initial training data"""
    print("📥 Loading initial training data...")
    
    # Check for initial data file
    initial_data_path = project_root / "atlas" / "data" / "raw" / "initial_training_data.csv"
    
    if not initial_data_path.exists():
        print("⚠️  No initial training data found.")
        print("💡 Please create 'atlas/data/raw/initial_training_data.csv' with your training data.")
        print("   Format: Email Text, Email Type")
        return False
    
    try:
        import pandas as pd
        df = pd.read_csv(initial_data_path)
        print(f"  ✅ Loaded {len(df)} initial training samples")
        
        # Process and store data (simplified)
        # In real implementation, this would use DataSourceManager
        
        return True
    except Exception as e:
        print(f"❌ Error loading initial data: {e}")
        return False

def main():
    """Main execution function"""
    parser = argparse.ArgumentParser(description="ATLAS AI Training Agent")
    parser.add_argument(
        '--mode', 
        choices=['basic', 'auto', 'init'], 
        default='basic',
        help='Execution mode: basic (manual), auto (automatic), init (initialize)'
    )
    parser.add_argument(
        '--config', 
        default='atlas/config/default_config.json',
        help='Configuration file path'
    )
    parser.add_argument(
        '--log-level',
        choices=['DEBUG', 'INFO', 'WARNING', 'ERROR'],
        default='INFO',
        help='Logging level'
    )
    
    args = parser.parse_args()
    
    # Setup logging
    setup_logging(args.log_level)
    
    print("🌟 ATLAS - Adaptive Training and Learning Automation System")
    print("=" * 60)
    
    if args.mode == 'init':
        initialize_data_structure()
        load_initial_data()
        
    elif args.mode == 'basic':
        asyncio.run(run_basic_agent())
        
    elif args.mode == 'auto':
        asyncio.run(run_auto_training_agent())

if __name__ == "__main__":
    main()

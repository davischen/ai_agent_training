#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
ATLAS Agent Main Execution Script (Adapted for same-level directory structure)
Run the AI Training Agent with automatic execution capabilities
"""

import os
import sys
import asyncio
import logging
import json
import argparse
from pathlib import Path

# 修改：同一層目錄導入
from core.ai_training_agent import AITrainingAgent

def setup_logging(log_level: str = "INFO"):
    """Setup logging configuration"""
    log_dir = Path("logs")
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
    try:
        with open(config_path, 'r', encoding='utf-8') as f:
            return json.load(f)
    except FileNotFoundError:
        # 返回默認配置
        return {
            'system': {
                'model_dir': './models',
                'max_concurrent_tasks': 5,
                'log_level': 'INFO'
            },
            'training': {
                'epochs': 3,
                'batch_size': 16
            }
        }

async def run_basic_agent():
    """Run basic AI Training Agent"""
    print("🚀 Starting ATLAS Agent in Basic Mode...")
    
    # 使用默認配置或從文件載入
    config = load_config('config.json')
    
    # Initialize agent
    agent = AITrainingAgent(
        model_dir=config['system']['model_dir']
    )
    
    print("✅ ATLAS Agent initialized successfully!")
    
    try:
        # Example inference
        sample_emails = [
            "Urgent: Your account will be suspended unless you verify immediately!",
            "Hi team, please find the quarterly report attached for your review.",
            "WINNER! You've won $1,000,000! Click here to claim your prize now!"
        ]
        
        print("\n📧 Processing sample emails...")
        
        # Quick inference test
        result = await agent.quick_inference(sample_emails)
        
        print("\n📊 Inference Results:")
        for i, (email, prediction) in enumerate(zip(sample_emails, result.get('predictions', []))):
            label = "🚨 PHISHING" if prediction == 1 else "✅ SAFE"
            confidence = result.get('confidences', [0])[i] if i < len(result.get('confidences', [])) else 0
            print(f"  {i+1}. {label} (confidence: {confidence:.2f}) - {email[:50]}...")
        
        # Test system status
        print("\n📋 System Status:")
        status = agent.get_system_status()
        print(f"Available models: {list(status.get('models', {}).keys())}")
        
        # Test sample data creation
        print("\n📝 Creating sample training data...")
        data_path = agent.create_sample_training_data(num_samples=20)
        print(f"Sample data created: {data_path}")
        
        # Test training data statistics
        stats = agent.get_training_statistics(data_path)
        print(f"Training data stats: {stats}")
        
    except Exception as e:
        print(f"❌ Error during demo: {e}")
        import traceback
        traceback.print_exc()
    
    finally:
        # Graceful shutdown
        agent.shutdown()
        print("\n🏁 ATLAS Agent stopped.")

async def run_training_demo():
    """Run training demonstration"""
    print("🎓 Starting ATLAS Agent Training Demo...")
    
    config = load_config('config.json')
    
    # Initialize agent
    agent = AITrainingAgent(model_dir=config['system']['model_dir'])
    
    print("✅ ATLAS Agent initialized for training!")
    
    try:
        # Create sample training data
        print("\n📝 Creating sample training data...")
        data_path = agent.create_sample_training_data(num_samples=100)
        
        # Start training
        print(f"\n🎓 Starting model training with data from: {data_path}")
        training_task_id = await agent.train_model_from_data(
            data_path=data_path,
            training_params={'epochs': 2, 'batch_size': 8}  # Small for demo
        )
        
        print(f"📋 Training task submitted: {training_task_id}")
        
        # Monitor training progress
        print("\n⏳ Monitoring training progress...")
        for i in range(60):  # Wait up to 60 seconds
            await asyncio.sleep(1)
            status = agent.get_task_status(training_task_id)
            if status and hasattr(status, 'status'):
                print(f"  Status: {status.status}", end='\r')
                if status.status.value in ['completed', 'failed']:
                    break
        
        # Get final results
        final_status = agent.get_task_status(training_task_id)
        if final_status:
            print(f"\n📊 Training completed with status: {final_status.status}")
            if final_status.result:
                print(f"📈 Training results: {final_status.result}")
        
    except Exception as e:
        print(f"❌ Training demo failed: {e}")
        import traceback
        traceback.print_exc()
    
    finally:
        agent.shutdown()
        print("\n🏁 Training demo completed.")

def run_simple_test():
    """Run simple synchronous test"""
    print("🧪 Running Simple ATLAS Agent Test...")
    
    try:
        # Test basic imports
        from core.model_manager import ModelManager
        from core.inference_engine import InferenceEngine
        from core.training_engine import TrainingEngine
        print("✅ All modules imported successfully")
        
        # Test model manager
        mm = ModelManager("./test_models")
        print("✅ ModelManager created")
        
        # Test inference engine
        ie = InferenceEngine(mm)
        print("✅ InferenceEngine created")
        
        # Test training engine
        te = TrainingEngine(mm)
        print("✅ TrainingEngine created")
        
        # Test sample data generation
        sample_data = te.data_generator.generate_synthetic_emails(10)
        print(f"✅ Generated {len(sample_data)} sample emails")
        
        print("\n🎉 All basic tests passed!")
        return True
        
    except Exception as e:
        print(f"❌ Test failed: {e}")
        import traceback
        traceback.print_exc()
        return False

def initialize_data_structure():
    """Initialize data directory structure"""
    print("🏗️  Initializing ATLAS Agent data structure...")
    
    directories = [
        "data/raw",
        "data/processed",
        "models/base_models", 
        "models/trained_models",
        "logs",
        "config"
    ]
    
    for directory in directories:
        path = Path(directory)
        path.mkdir(parents=True, exist_ok=True)
        print(f"  ✅ Created: {directory}")
    
    # Create sample config
    config_path = Path("config.json")
    if not config_path.exists():
        print("  📝 Creating default configuration...")
        default_config = {
            "system": {
                "model_dir": "./models",
                "max_concurrent_tasks": 5,
                "log_level": "INFO"
            },
            "training": {
                "epochs": 3,
                "batch_size": 16,
                "learning_rate": 2e-5
            },
            "inference": {
                "batch_size": 32,
                "threshold": 0.5
            }
        }
        
        with open(config_path, 'w', encoding='utf-8') as f:
            json.dump(default_config, f, indent=2)
        
        print(f"  ✅ Created configuration file: {config_path}")
    
    print("🎉 Data structure initialized successfully!")

def create_sample_data():
    """Create sample training data"""
    print("📝 Creating sample training data...")
    
    sample_data = {
        'Email Text': [
            "URGENT: Your account will be suspended unless you verify immediately!",
            "Hi team, please find the quarterly financial report attached for your review.",
            "WINNER! You've won $1,000,000! Click here to claim your prize now!",
            "Meeting reminder: Our weekly team sync is scheduled for tomorrow at 2 PM.",
            "Your PayPal account has been limited. Verify your identity to restore access.",
            "Thank you for your purchase. Your order #12345 has been shipped.",
            "SECURITY ALERT: Suspicious login detected. Confirm your identity immediately.",
            "Project update: The new feature development is on track for next week's release."
        ],
        'Email Type': [
            'Phishing Email', 'Safe Email', 'Phishing Email', 'Safe Email',
            'Phishing Email', 'Safe Email', 'Phishing Email', 'Safe Email'
        ]
    }
    
    import pandas as pd
    df = pd.DataFrame(sample_data)
    
    # Save to data directory
    data_dir = Path("data/raw")
    data_dir.mkdir(parents=True, exist_ok=True)
    
    sample_file = data_dir / "sample_training_data.csv"
    df.to_csv(sample_file, index=False)
    
    print(f"✅ Sample data created: {sample_file}")
    print(f"📊 Created {len(df)} samples ({sum(df['Email Type'] == 'Phishing Email')} phishing, {sum(df['Email Type'] == 'Safe Email')} safe)")

def main():
    """Main execution function"""
    parser = argparse.ArgumentParser(description="ATLAS AI Training Agent")
    parser.add_argument(
        '--mode', 
        choices=['basic', 'training', 'test', 'init'], 
        default='basic',
        help='Execution mode: basic (inference demo), training (training demo), test (simple test), init (initialize)'
    )
    parser.add_argument(
        '--config', 
        default='config.json',
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
        create_sample_data()
        
    elif args.mode == 'test':
        success = run_simple_test()
        if success:
            print("\n✅ Ready to run ATLAS Agent!")
            print("Try: python run_agent.py --mode basic")
        
    elif args.mode == 'basic':
        asyncio.run(run_basic_agent())
        
    elif args.mode == 'training':
        asyncio.run(run_training_demo())

if __name__ == "__main__":
    main()

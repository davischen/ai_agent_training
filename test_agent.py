"""
Simple test script for ATLAS Agent
"""

import asyncio
import pandas as pd
import json
import os
from pathlib import Path

def test_basic_setup():
    """Test basic environment setup"""
    print("🧪 Testing ATLAS Agent Setup...")
    
    # Test 1: Check required directories
    required_dirs = [
        "atlas/data",
        "atlas/models", 
        "atlas/logs",
        "atlas/config"
    ]
    
    for directory in required_dirs:
        path = Path(directory)
        if path.exists():
            print(f"  ✅ Directory exists: {directory}")
        else:
            print(f"  ❌ Missing directory: {directory}")
            path.mkdir(parents=True, exist_ok=True)
            print(f"  🔧 Created directory: {directory}")
    
    # Test 2: Check dependencies
    try:
        import torch
        import sentence_transformers
        import sklearn
        import pandas
        import numpy
        import nltk
        import schedule
        print("  ✅ All required packages installed")
    except ImportError as e:
        print(f"  ❌ Missing package: {e}")
        return False
    
    # Test 3: Create sample config
    config_path = Path("atlas/config/test_config.json")
    if not config_path.exists():
        sample_config = {
            "data_sources": {
                "synthetic_data": {"enabled": True},
                "storage_path": "atlas/data/test_data.db"
            },
            "system": {
                "max_concurrent_tasks": 2,
                "model_dir": "atlas/models",
                "log_level": "INFO"
            }
        }
        
        with open(config_path, 'w') as f:
            json.dump(sample_config, f, indent=2)
        print(f"  ✅ Created test config: {config_path}")
    
    return True

def create_sample_data():
    """Create sample training data"""
    print("📝 Creating sample training data...")
    
    sample_data = {
        'Email Text': [
            "Urgent: Your account will be suspended unless you verify immediately!",
            "WINNER! You've won $1,000,000! Click here to claim your prize now!",
            "Your bank account has been compromised. Update your details now.",
            "Hi team, our weekly meeting is scheduled for tomorrow at 2 PM.",
            "Please find attached the quarterly financial report for review.",
            "System maintenance notification: Server downtime this weekend.",
            "Confirm your identity to prevent account closure immediately!",
            "Meeting room booking confirmation for Conference Room A.",
            "Your package delivery failed. Reschedule at suspicious-link.com",
            "Thank you for your purchase. Your receipt is attached."
        ],
        'Email Type': [
            'Phishing Email', 'Phishing Email', 'Phishing Email',
            'Safe Email', 'Safe Email', 'Safe Email',
            'Phishing Email', 'Safe Email', 'Phishing Email', 'Safe Email'
        ]
    }
    
    df = pd.DataFrame(sample_data)
    
    # Save to CSV
    output_path = Path("atlas/data/sample_training_data.csv")
    df.to_csv(output_path, index=False)
    print(f"  ✅ Sample data saved to: {output_path}")
    print(f"  📊 Created {len(df)} samples ({sum(df['Email Type'] == 'Phishing Email')} phishing, {sum(df['Email Type'] == 'Safe Email')} safe)")
    
    return output_path

async def test_basic_inference():
    """Test basic inference functionality"""
    print("🔍 Testing basic inference...")
    
    try:
        # Import our modules (simplified test)
        from sentence_transformers import SentenceTransformer
        
        # Test model loading
        model = SentenceTransformer('all-MiniLM-L6-v2')
        print("  ✅ Model loaded successfully")
        
        # Test inference
        test_emails = [
            "Urgent: Verify your account now!",
            "Meeting scheduled for tomorrow"
        ]
        
        embeddings = model.encode(test_emails)
        print(f"  ✅ Generated embeddings: {embeddings.shape}")
        
        # Simple classification test
        import numpy as np
        from sklearn.metrics.pairwise import cosine_similarity
        
        # Create dummy classification
        similarity_matrix = cosine_similarity(embeddings)
        print(f"  ✅ Similarity calculation successful")
        
        return True
        
    except Exception as e:
        print(f"  ❌ Inference test failed: {e}")
        return False

def test_data_processing():
    """Test data processing functionality"""
    print("📊 Testing data processing...")
    
    try:
        # Test CSV reading
        sample_path = Path("atlas/data/sample_training_data.csv")
        if not sample_path.exists():
            create_sample_data()
        
        df = pd.read_csv(sample_path)
        print(f"  ✅ Loaded {len(df)} samples from CSV")
        
        # Test data cleaning
        df['Label'] = df['Email Type'].map({'Safe Email': 0, 'Phishing Email': 1})
        print(f"  ✅ Label mapping: {df['Label'].value_counts().to_dict()}")
        
        # Test text preprocessing
        df['Clean_Text'] = df['Email Text'].str.lower().str.strip()
        print(f"  ✅ Text preprocessing completed")
        
        return True
        
    except Exception as e:
        print(f"  ❌ Data processing test failed: {e}")
        return False

async def run_all_tests():
    """Run all tests"""
    print("🚀 Starting ATLAS Agent Tests")
    print("=" * 50)
    
    # Test 1: Basic setup
    setup_ok = test_basic_setup()
    
    # Test 2: Create sample data
    if setup_ok:
        create_sample_data()
    
    # Test 3: Data processing
    data_ok = test_data_processing()
    
    # Test 4: Basic inference
    inference_ok = await test_basic_inference()
    
    # Summary
    print("\n📋 Test Summary:")
    print(f"  Setup: {'✅ PASS' if setup_ok else '❌ FAIL'}")
    print(f"  Data Processing: {'✅ PASS' if data_ok else '❌ FAIL'}")
    print(f"  Basic Inference: {'✅ PASS' if inference_ok else '❌ FAIL'}")
    
    if all([setup_ok, data_ok, inference_ok]):
        print("\n🎉 All tests passed! ATLAS Agent is ready to run.")
        print("\nNext steps:")
        print("  1. Run: python run_agent.py --mode basic")
        print("  2. Check logs in: atlas/logs/")
        print("  3. Monitor data in: atlas/data/")
    else:
        print("\n⚠️  Some tests failed. Please check the errors above.")
        print("Make sure all dependencies are installed:")
        print("  pip install -r requirements.txt")

if __name__ == "__main__":
    asyncio.run(run_all_tests())

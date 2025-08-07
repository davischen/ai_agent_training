#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
ATLAS Agent Streamlit Dashboard Launcher
Quick launcher for the Streamlit dashboard
"""

import subprocess
import sys
import os
from pathlib import Path

def check_dependencies():
    """Check if required dependencies are installed"""
    required_packages = [
        'streamlit',
        'plotly',
        'pandas',
        'numpy'
    ]
    
    missing_packages = []
    
    for package in required_packages:
        try:
            __import__(package)
        except ImportError:
            missing_packages.append(package)
    
    if missing_packages:
        print("❌ Missing required packages:")
        for package in missing_packages:
            print(f"   - {package}")
        print("\n📦 Install missing packages with:")
        print(f"   pip install {' '.join(missing_packages)}")
        return False
    
    return True

def check_atlas_components():
    """Check if ATLAS components are available"""
    try:
        from ai_training_agent import AITrainingAgent
        from model_manager import ModelManager
        print("✅ ATLAS components found")
        return True
    except ImportError as e:
        print(f"⚠️  ATLAS components not found: {e}")
        print("   Dashboard will run in demo mode")
        return False

def create_requirements_file():
    """Create requirements.txt for Streamlit dashboard"""
    requirements = [
        "streamlit>=1.28.0",
        "plotly>=5.15.0", 
        "pandas>=1.5.0",
        "numpy>=1.21.0",
        "torch>=1.9.0",
        "sentence-transformers>=2.2.0",
        "scikit-learn>=1.0.0",
        "nltk>=3.6.0",
        "schedule>=1.1.0"
    ]
    
    requirements_path = Path("requirements_streamlit.txt")
    
    if not requirements_path.exists():
        with open(requirements_path, 'w') as f:
            for req in requirements:
                f.write(f"{req}\n")
        print(f"📝 Created {requirements_path}")
    
    return requirements_path

def run_streamlit():
    """Run the Streamlit dashboard"""
    dashboard_file = "streamlit_dashboard.py"
    
    if not os.path.exists(dashboard_file):
        print(f"❌ Dashboard file {dashboard_file} not found!")
        print("   Please ensure streamlit_dashboard.py is in the current directory")
        return False
    
    print("🚀 Starting ATLAS Agent Dashboard...")
    print("📱 Dashboard will open in your default browser")
    print("🛑 Press Ctrl+C to stop the dashboard")
    
    try:
        # Run Streamlit
        cmd = [
            sys.executable, "-m", "streamlit", "run", dashboard_file,
            "--server.port", "8501",
            "--server.address", "localhost",
            "--browser.gatherUsageStats", "false"
        ]
        
        subprocess.run(cmd)
        
    except KeyboardInterrupt:
        print("\n🛑 Dashboard stopped by user")
    except Exception as e:
        print(f"❌ Error running dashboard: {e}")
        return False
    
    return True

def main():
    """Main launcher function"""
    print("🌟 ATLAS Agent Dashboard Launcher")
    print("=" * 50)
    
    # Check dependencies
    print("🔍 Checking dependencies...")
    if not check_dependencies():
        print("\n❌ Please install missing dependencies and try again")
        
        # Offer to create requirements file
        create_req = input("\n📝 Create requirements file? (y/n): ").lower().strip()
        if create_req in ['y', 'yes']:
            req_file = create_requirements_file()
            print(f"\n📦 Install with: pip install -r {req_file}")
        
        return False
    
    # Check ATLAS components
    print("🔍 Checking ATLAS components...")
    atlas_available = check_atlas_components()
    
    if not atlas_available:
        print("⚠️  Some features may be limited without ATLAS components")
        
        continue_anyway = input("Continue anyway? (y/n): ").lower().strip()
        if continue_anyway not in ['y', 'yes']:
            return False
    
    print("✅ All checks passed!")
    print()
    
    # Run dashboard
    success = run_streamlit()
    
    if success:
        print("✅ Dashboard session completed")
    else:
        print("❌ Dashboard failed to start")
    
    return success

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
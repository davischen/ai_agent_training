#!/usr/bin/env python3
"""
ATLAS Setup Script
Helps set up and verify the ATLAS Agent environment
"""

import os
import sys
import subprocess
import importlib

def check_python_version():
    """Check if Python version is compatible"""
    version = sys.version_info
    if version.major == 3 and version.minor >= 8:
        print(f"✅ Python {version.major}.{version.minor}.{version.micro} is compatible")
        return True
    else:
        print(f"❌ Python {version.major}.{version.minor}.{version.micro} is not compatible. Need Python 3.8+")
        return False

def install_package(package_name):
    """Install a Python package using pip"""
    try:
        subprocess.check_call([sys.executable, "-m", "pip", "install", package_name])
        print(f"✅ Installed {package_name}")
        return True
    except subprocess.CalledProcessError:
        print(f"❌ Failed to install {package_name}")
        return False

def check_and_install_dependencies():
    """Check and install required dependencies"""
    required_packages = [
        'streamlit',
        'pandas',
        'numpy',
        'plotly',
        'sentence-transformers',
        'torch',
        'transformers',
        'scikit-learn'
    ]
    
    missing_packages = []
    
    print("🔍 Checking dependencies...")
    
    for package in required_packages:
        try:
            importlib.import_module(package.replace('-', '_'))
            print(f"✅ {package} is available")
        except ImportError:
            print(f"❌ {package} is missing")
            missing_packages.append(package)
    
    if missing_packages:
        print(f"\n📦 Installing missing packages: {', '.join(missing_packages)}")
        
        for package in missing_packages:
            if not install_package(package):
                return False
        
        print("✅ All packages installed successfully!")
    else:
        print("✅ All dependencies are available!")
    
    return True

def check_project_structure():
    """Check if project structure is correct"""
    print("🔍 Checking project structure...")
    
    current_dir = os.getcwd()
    print(f"Current directory: {current_dir}")
    
    # Expected files and directories
    expected_structure = {
        'core/': 'directory',
        'core/ai_training_agent.py': 'file',
        'core/model_manager.py': 'file',
        'core/inference_engine.py': 'file',
        'core/training_engine.py': 'file',
        'run_streamlit.py': 'file'
    }
    
    all_good = True
    
    for path, item_type in expected_structure.items():
        full_path = os.path.join(current_dir, path)
        
        if item_type == 'directory':
            if os.path.isdir(full_path):
                print(f"✅ Directory found: {path}")
            else:
                print(f"❌ Directory missing: {path}")
                all_good = False
        
        elif item_type == 'file':
            if os.path.isfile(full_path):
                print(f"✅ File found: {path}")
            else:
                print(f"❌ File missing: {path}")
                all_good = False
    
    # Create models directory if it doesn't exist
    models_dir = os.path.join(current_dir, 'models')
    if not os.path.exists(models_dir):
        os.makedirs(models_dir)
        print(f"✅ Created models directory: {models_dir}")
    
    return all_good

def test_imports():
    """Test importing ATLAS components"""
    print("🔍 Testing ATLAS component imports...")
    
    # Add current directory and core to path
    current_dir = os.getcwd()
    core_dir = os.path.join(current_dir, 'core')
    
    if current_dir not in sys.path:
        sys.path.insert(0, current_dir)
    if core_dir not in sys.path:
        sys.path.insert(0, core_dir)
    
    try:
        from core.ai_training_agent import AITrainingAgent
        print("✅ Successfully imported AITrainingAgent")
    except ImportError as e:
        print(f"❌ Failed to import AITrainingAgent: {e}")
        return False
    
    try:
        from core.model_manager import ModelManager
        print("✅ Successfully imported ModelManager")
    except ImportError as e:
        print(f"❌ Failed to import ModelManager: {e}")
        return False
    
    try:
        from core.inference_engine import InferenceEngine
        print("✅ Successfully imported InferenceEngine")
    except ImportError as e:
        print(f"❌ Failed to import InferenceEngine: {e}")
        return False
    
    try:
        from core.training_engine import TrainingEngine
        print("✅ Successfully imported TrainingEngine")
    except ImportError as e:
        print(f"❌ Failed to import TrainingEngine: {e}")
        return False
    
    print("✅ All ATLAS components imported successfully!")
    return True

def create_sample_requirements_file():
    """Create a requirements.txt file"""
    requirements_content = """streamlit>=1.28.0
pandas>=1.5.0
numpy>=1.24.0
plotly>=5.15.0
sentence-transformers>=2.2.0
torch>=2.0.0
scikit-learn>=1.3.0
transformers>=4.30.0
asyncio-mqtt>=0.11.0
"""
    
    with open('requirements.txt', 'w') as f:
        f.write(requirements_content)
    
    print("✅ Created requirements.txt file")

def main():
    """Main setup function"""
    print("🚀 ATLAS Agent Setup Script")
    print("=" * 40)
    
    # Check Python version
    if not check_python_version():
        return False
    
    # Check project structure
    structure_ok = check_project_structure()
    
    # Install dependencies
    if not check_and_install_dependencies():
        print("❌ Dependency installation failed")
        return False
    
    # Create requirements file
    create_sample_requirements_file()
    
    # Test imports (only if structure is OK)
    if structure_ok:
        imports_ok = test_imports()
    else:
        print("⚠️ Skipping import test due to missing files")
        imports_ok = False
    
    print("\n" + "=" * 40)
    
    if structure_ok and imports_ok:
        print("🎉 Setup completed successfully!")
        print("\nTo run the dashboard:")
        print("  streamlit run run_streamlit.py")
    else:
        print("⚠️ Setup completed with issues.")
        print("\nThe dashboard will run in demo mode.")
        print("To run anyway:")
        print("  streamlit run run_streamlit.py")
        
        if not structure_ok:
            print("\n📝 Missing files need to be created in the core/ directory:")
            print("  - ai_training_agent.py")
            print("  - model_manager.py") 
            print("  - inference_engine.py")
            print("  - training_engine.py")
    
    return True

if __name__ == "__main__":
    main()
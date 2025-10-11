#!/usr/bin/env python3
"""
Setup script for Policy Intelligence Assistant
"""
import subprocess
import sys
import os
from pathlib import Path

def run_command(command, description):
    """Run a command and handle errors"""
    print(f"🔄 {description}...")
    try:
        result = subprocess.run(command, shell=True, check=True, capture_output=True, text=True)
        print(f"✅ {description} completed")
        return True
    except subprocess.CalledProcessError as e:
        print(f"❌ {description} failed: {e}")
        if e.stdout:
            print(f"Output: {e.stdout}")
        if e.stderr:
            print(f"Error: {e.stderr}")
        return False

def check_python_version():
    """Check if Python version is compatible"""
    print("🐍 Checking Python version...")
    if sys.version_info < (3, 8):
        print("❌ Python 3.8 or higher is required")
        return False
    print(f"✅ Python {sys.version_info.major}.{sys.version_info.minor} detected")
    return True

def install_dependencies():
    """Install required dependencies"""
    print("📦 Installing dependencies...")
    
    # Install basic requirements
    if not run_command("pip install -r requirements.txt", "Installing basic requirements"):
        return False
    
    # Install spaCy model
    if not run_command("python -m spacy download en_core_web_sm", "Installing spaCy English model"):
        print("⚠️  spaCy model installation failed, but continuing...")
    
    return True

def create_directories():
    """Create necessary directories"""
    print("📁 Creating directories...")
    
    directories = [
        "data/raw",
        "data/processed", 
        "data/sample",
        "logs",
        "outputs"
    ]
    
    for dir_path in directories:
        Path(dir_path).mkdir(parents=True, exist_ok=True)
        print(f"✅ Created directory: {dir_path}")
    
    return True

def run_tests():
    """Run system tests"""
    print("🧪 Running system tests...")
    
    if not run_command("python test_pipeline.py", "Running pipeline tests"):
        print("⚠️  Some tests failed, but setup continues...")
        return False
    
    return True

def create_sample_data():
    """Create sample data for testing"""
    print("📄 Creating sample data...")
    
    try:
        from policy_intelligence.data.sample_documents import create_sample_documents
        sample_dir = create_sample_documents()
        print(f"✅ Sample data created in {sample_dir}")
        return True
    except Exception as e:
        print(f"❌ Failed to create sample data: {e}")
        return False

def main():
    """Main setup function"""
    print("🚀 Policy Intelligence Assistant Setup")
    print("=" * 50)
    
    # Check Python version
    if not check_python_version():
        return 1
    
    # Install dependencies
    if not install_dependencies():
        print("❌ Dependency installation failed")
        return 1
    
    # Create directories
    if not create_directories():
        print("❌ Directory creation failed")
        return 1
    
    # Create sample data
    if not create_sample_data():
        print("❌ Sample data creation failed")
        return 1
    
    # Run tests
    if not run_tests():
        print("⚠️  Some tests failed, but setup completed")
    
    print("\n" + "=" * 50)
    print("🎉 Setup completed successfully!")
    print("\nNext steps:")
    print("1. Run the pipeline: python cli.py run")
    print("2. Try interactive Q&A: python cli.py qa")
    print("3. Check status: python cli.py status")
    print("4. Open demo notebook: jupyter notebook notebooks/demo.ipynb")
    print("\nFor more information, see README.md")
    
    return 0

if __name__ == "__main__":
    sys.exit(main())


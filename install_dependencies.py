#!/usr/bin/env python3
"""
Quick Dependency Installation Script

This script helps install all required dependencies for the 
Statistics & ML Learning App.
"""

import subprocess
import sys
import os

def run_command(command):
    """Run a command and return success status."""
    try:
        result = subprocess.run(command, shell=True, check=True, 
                              capture_output=True, text=True)
        return True, result.stdout
    except subprocess.CalledProcessError as e:
        return False, e.stderr

def check_python_version():
    """Check if Python version is compatible."""
    version = sys.version_info
    if version.major < 3 or (version.major == 3 and version.minor < 8):
        print("❌ Python 3.8+ required. Current version:", sys.version)
        return False
    print(f"✅ Python {version.major}.{version.minor}.{version.micro} - Compatible")
    return True

def install_dependencies():
    """Install all required dependencies."""
    print("📦 Installing dependencies...")
    
    # Check if requirements.txt exists
    if not os.path.exists('requirements.txt'):
        print("❌ requirements.txt not found!")
        return False
    
    # Install dependencies
    print("Installing packages from requirements.txt...")
    success, output = run_command("pip install -r requirements.txt")
    
    if success:
        print("✅ Dependencies installed successfully!")
        return True
    else:
        print(f"❌ Installation failed: {output}")
        return False

def test_imports():
    """Test if all critical imports work."""
    print("\n🧪 Testing imports...")
    
    packages = [
        'streamlit',
        'numpy', 
        'pandas',
        'matplotlib',
        'seaborn',
        'plotly',
        'scipy',
        'sklearn'
    ]
    
    all_good = True
    for package in packages:
        try:
            __import__(package)
            print(f"✅ {package}")
        except ImportError:
            print(f"❌ {package} - Import failed")
            all_good = False
    
    return all_good

def main():
    """Main installation process."""
    print("🚀 Statistics & ML Learning App - Dependency Setup")
    print("="*50)
    
    # Check Python version
    if not check_python_version():
        return False
    
    # Install dependencies
    if not install_dependencies():
        return False
    
    # Test imports
    if not test_imports():
        print("\n⚠️  Some packages failed to import. You may need to:")
        print("   - Restart your terminal")
        print("   - Use a virtual environment")
        print("   - Install packages manually")
        return False
    
    print("\n🎉 Setup Complete!")
    print("="*50)
    print("✅ All dependencies installed successfully!")
    print("\n🚀 To start the application:")
    print("   streamlit run main_app.py")
    print("\n💡 Or run the test structure:")
    print("   python test_structure.py")
    
    return True

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
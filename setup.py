#!/usr/bin/env python3
"""
Quick setup script for PEDS Walkability App
Run this to install dependencies and set up the environment
"""

import subprocess
import sys
import os

def run_command(command, description):
    """Run a command and handle errors"""
    print(f"\n🔧 {description}...")
    try:
        result = subprocess.run(command, shell=True, check=True, capture_output=True, text=True)
        print(f"✅ {description} completed successfully")
        return True
    except subprocess.CalledProcessError as e:
        print(f"❌ {description} failed:")
        print(f"Error: {e.stderr}")
        return False

def main():
    print("🚶 PEDS Walkability App Setup")
    print("=" * 40)
    
    # Check Python version
    if sys.version_info < (3, 8):
        print("❌ Python 3.8+ is required")
        return False
    
    print(f"✅ Python {sys.version_info.major}.{sys.version_info.minor} detected")
    
    # Install dependencies
    if not run_command("pip install -r requirements.txt", "Installing dependencies"):
        print("\n💡 If installation fails, try:")
        print("   pip install --upgrade pip")
        print("   pip install -r requirements.txt --no-cache-dir")
        return False
    
    # Create models directory
    models_dir = "models"
    if not os.path.exists(models_dir):
        os.makedirs(models_dir)
        print(f"✅ Created {models_dir} directory for SAM checkpoints")
    
    # Create .env.example if it doesn't exist
    env_example = ".env.example"
    if not os.path.exists(env_example):
        with open(env_example, 'w') as f:
            f.write("# Google Maps Platform API Key\n")
            f.write("GOOGLE_MAPS_API_KEY=your_api_key_here\n\n")
            f.write("# SAM Models Directory (optional)\n")
            f.write("SAM_MODELS_DIR=models\n")
        print(f"✅ Created {env_example}")
    
    print("\n🎉 Setup completed successfully!")
    print("\n📋 Next steps:")
    print("1. Get a Google Maps API key from: https://console.cloud.google.com/apis/credentials")
    print("2. Enable Street View Static API")
    print("3. Set your API key: set GOOGLE_MAPS_API_KEY=your_key_here")
    print("4. Download a SAM model (.pth) to the models/ folder")
    print("5. Run the app: python -m streamlit run app/streamlit_app.py")
    
    return True

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
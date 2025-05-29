"""
Simple runner script for the dataset visualizer
Run this to start the Streamlit app
"""
import subprocess
import sys
import os

def install_requirements():
    """Install required packages"""
    try:
        subprocess.check_call([sys.executable, "-m", "pip", "install", "-r", "requirements.txt"])
        print("✅ Requirements installed successfully!")
    except subprocess.CalledProcessError as e:
        print(f"❌ Error installing requirements: {e}")
        return False
    return True

def run_visualizer():
    """Run the Streamlit visualizer"""
    try:
        subprocess.run([sys.executable, "-m", "streamlit", "run", "dataset_visualizer.py"])
    except FileNotFoundError:
        print("❌ Streamlit not found. Installing requirements...")
        if install_requirements():
            subprocess.run([sys.executable, "-m", "streamlit", "run", "dataset_visualizer.py"])

if __name__ == "__main__":
    print("🚀 Starting Multilingual Dataset Visualizer...")
    
    # Check if dataset exists
    if not os.path.exists("dataset.json"):
        print("❌ dataset.json not found in current directory!")
        sys.exit(1)
    
    # Check if requirements are installed
    try:
        import streamlit
        import plotly
        import pandas
    except ImportError:
        print("📦 Installing required packages...")
        if not install_requirements():
            sys.exit(1)
    
    print("🎯 Launching visualizer...")
    run_visualizer()

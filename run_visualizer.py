"""
Simple runner script for the dataset visualizer
Run this to start the Streamlit app
"""
import subprocess
import sys
import os
import json

def validate_and_fix_dataset():
    """Validate and fix common dataset issues"""
    try:
        with open("dataset.json", "r", encoding="utf-8") as f:
            data = json.load(f)
        
        print("🔍 Validating dataset structure...")
        
        # Check if translations exist
        if "translations" not in data:
            print("❌ No 'translations' key found in dataset!")
            return False
        
        # Fix common issues in translations
        fixed_count = 0
        invalid_entries = []
        
        for i, entry in enumerate(data["translations"]):
            # Check if entry has all required fields
            required_fields = ["sq", "en", "de"]
            missing_fields = [field for field in required_fields if field not in entry]
            
            if missing_fields:
                print(f"⚠️  Entry {i} missing fields: {missing_fields}")
                # Add missing fields with empty strings
                for field in missing_fields:
                    entry[field] = ""
                    fixed_count += 1
            
            # Ensure all required fields exist and are strings
            for field in required_fields:
                if field in entry:
                    # Handle None values
                    if entry[field] is None:
                        entry[field] = ""
                        fixed_count += 1
                    # Convert numeric values to strings
                    elif isinstance(entry[field], (int, float)):
                        entry[field] = str(entry[field])
                        fixed_count += 1
                    # Ensure it's a string
                    elif not isinstance(entry[field], str):
                        entry[field] = str(entry[field])
                        fixed_count += 1
            
            # Ensure id is numeric
            if "id" in entry:
                if isinstance(entry["id"], str):
                    try:
                        entry["id"] = int(entry["id"])
                    except ValueError:
                        # If ID can't be converted, use the index
                        entry["id"] = i + 1
                        fixed_count += 1
                elif not isinstance(entry["id"], int):
                    entry["id"] = i + 1
                    fixed_count += 1
            else:
                # Add missing ID
                entry["id"] = i + 1
                fixed_count += 1
            
            # Ensure other fields are strings if they exist
            for field in ["category", "difficulty"]:
                if field in entry and entry[field] is not None:
                    if not isinstance(entry[field], str):
                        entry[field] = str(entry[field])
                        fixed_count += 1
        
        # Remove any completely invalid entries
        if invalid_entries:
            print(f"🗑️  Removing {len(invalid_entries)} invalid entries")
            for idx in sorted(invalid_entries, reverse=True):
                del data["translations"][idx]
        
        if fixed_count > 0:
            print(f"🔧 Fixed {fixed_count} data issues")
            # Save the fixed dataset
            with open("dataset.json", "w", encoding="utf-8") as f:
                json.dump(data, f, ensure_ascii=False, indent=2)
            print("💾 Saved corrected dataset")
        
        print(f"✅ Dataset validated: {len(data['translations'])} entries found")
        return True
        
    except json.JSONDecodeError as e:
        print(f"❌ Invalid JSON in dataset.json: {e}")
        return False
    except Exception as e:
        print(f"❌ Error validating dataset: {e}")
        print(f"📍 Error details: {type(e).__name__}: {str(e)}")
        return False

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
    
    # Validate and fix dataset
    if not validate_and_fix_dataset():
        print("❌ Dataset validation failed!")
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

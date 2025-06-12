import os
import shutil
import glob
import random

def move_simulation_files_to_datasets():
    """Move existing simulation files to appropriate dataset folders"""
    
    # Create directories
    os.makedirs("train_dataset", exist_ok=True)
    os.makedirs("test_dataset", exist_ok=True)
    os.makedirs("pred_dataset", exist_ok=True)
    
    # Find all simulation files in current directory and dataset folder
    json_files = glob.glob("fire_simulation_*.json") + glob.glob("dataset/fire_simulation_*.json")
    
    if not json_files:
        print("No simulation files found to organize.")
        return
    
    print(f"Found {len(json_files)} simulation files")
    
    # Shuffle for random distribution
    random.shuffle(json_files)
    
    # Split: 80% train, 20% test
    split_point = int(0.8 * len(json_files))
    train_files = json_files[:split_point]
    test_files = json_files[split_point:]
    
    print(f"Moving {len(train_files)} files to train_dataset/")
    print(f"Moving {len(test_files)} files to test_dataset/")
    
    # Move train files
    for json_file in train_files:
        bin_file = json_file.replace('.json', '.bin')
        
        # Move JSON file
        dest_json = os.path.join("train_dataset", os.path.basename(json_file))
        if json_file != dest_json:
            shutil.move(json_file, dest_json)
            print(f"  Moved: {json_file} -> {dest_json}")
        
        # Move corresponding BIN file if exists
        if os.path.exists(bin_file):
            dest_bin = os.path.join("train_dataset", os.path.basename(bin_file))
            if bin_file != dest_bin:
                shutil.move(bin_file, dest_bin)
                print(f"  Moved: {bin_file} -> {dest_bin}")
    
    # Move test files
    for json_file in test_files:
        bin_file = json_file.replace('.json', '.bin')
        
        # Move JSON file
        dest_json = os.path.join("test_dataset", os.path.basename(json_file))
        if json_file != dest_json:
            shutil.move(json_file, dest_json)
            print(f"  Moved: {json_file} -> {dest_json}")
        
        # Move corresponding BIN file if exists
        if os.path.exists(bin_file):
            dest_bin = os.path.join("test_dataset", os.path.basename(bin_file))
            if bin_file != dest_bin:
                shutil.move(bin_file, dest_bin)
                print(f"  Moved: {bin_file} -> {dest_bin}")
    
    print("\nDataset organization completed!")
    print(f"Train dataset: {len(glob.glob('train_dataset/*.json'))} files")
    print(f"Test dataset: {len(glob.glob('test_dataset/*.json'))} files")

def check_dataset_status():
    """Check the current status of datasets"""
    print("Dataset Status:")
    print("=" * 40)
    
    train_files = glob.glob("train_dataset/*.json")
    test_files = glob.glob("test_dataset/*.json")
    pred_files = glob.glob("pred_dataset/*.json")
    
    print(f"Train dataset: {len(train_files)} files")
    print(f"Test dataset:  {len(test_files)} files")
    print(f"Pred dataset:  {len(pred_files)} files")
    
    if train_files:
        print("\nTrain files:")
        for f in train_files[:5]:  # Show first 5
            print(f"  {os.path.basename(f)}")
        if len(train_files) > 5:
            print(f"  ... and {len(train_files) - 5} more")
    
    if test_files:
        print("\nTest files:")
        for f in test_files[:5]:  # Show first 5
            print(f"  {os.path.basename(f)}")
        if len(test_files) > 5:
            print(f"  ... and {len(test_files) - 5} more")

if __name__ == "__main__":
    print("Fire Simulation Dataset Preparation")
    print("=" * 40)
    
    # Check current status
    check_dataset_status()
    
    # Ask user if they want to organize files
    if len(glob.glob("fire_simulation_*.json")) + len(glob.glob("dataset/fire_simulation_*.json")) > 0:
        response = input("\nOrganize simulation files into train/test datasets? (y/n): ")
        if response.lower() == 'y':
            move_simulation_files_to_datasets()
            print("\n" + "=" * 40)
            check_dataset_status()
        else:
            print("No files moved.")
    else:
        print("\nNo simulation files found to organize.")
        print("Run fire_simulation.py to generate simulation data first.")
import numpy as np
import json
import glob
import os

def check_prediction_shapes():
    """Check the shapes of prediction data"""
    print("=== PREDICTION SHAPE ANALYSIS ===")
    
    # Check test data
    test_files = glob.glob("test_dataset/fire_simulation_*.json")
    pred_files = glob.glob("pred_dataset/fire_simulation_*.json")
    
    if not test_files:
        print("No test files found")
        return
    if not pred_files:
        print("No prediction files found")
        return
    
    test_file = test_files[0]
    pred_file = pred_files[0]
    
    print(f"Checking files:")
    print(f"  Test: {os.path.basename(test_file)}")
    print(f"  Pred: {os.path.basename(pred_file)}")
    
    # Load test data
    with open(test_file, 'r') as f:
        test_meta = json.load(f)
    
    test_bin = test_file.replace('.json', '.bin')
    with open(test_bin, 'rb') as f:
        test_bytes = f.read()
    test_shape = tuple(test_meta['data_shape'])
    test_data = np.frombuffer(test_bytes, dtype=np.float32).reshape(test_shape)
    
    # Load prediction data
    with open(pred_file, 'r') as f:
        pred_meta = json.load(f)
    
    pred_bin = pred_file.replace('.json', '.bin')
    with open(pred_bin, 'rb') as f:
        pred_bytes = f.read()
    pred_shape = tuple(pred_meta['data_shape'])
    pred_data = np.frombuffer(pred_bytes, dtype=np.float32).reshape(pred_shape)
    
    print(f"\nData shapes:")
    print(f"  Test data:       {test_data.shape}")
    print(f"  Prediction data: {pred_data.shape}")
    
    print(f"\nShape breakdown:")
    print(f"  Test: {test_shape[0]} timesteps × {test_shape[1]} height × {test_shape[2]} width × {test_shape[3]} variables")
    print(f"  Pred: {pred_shape[0]} timesteps × {pred_shape[1]} height × {pred_shape[2]} width × {pred_shape[3]} variables")
    
    if test_shape == pred_shape:
        print("  ✓ Shapes match perfectly")
    else:
        print("  ✗ Shapes don't match!")
    
    # Check grid dimensions specifically
    print(f"\nGrid dimensions:")
    print(f"  Test grid: {test_shape[1]} × {test_shape[2]}")
    print(f"  Pred grid: {pred_shape[1]} × {pred_shape[2]}")
    
    if test_shape[1] == 20 and test_shape[2] == 20:
        print("  ✓ Test data is 20×20 grid")
    else:
        print(f"  ✗ Test data is {test_shape[1]}×{test_shape[2]} grid (expected 20×20)")
    
    if pred_shape[1] == 20 and pred_shape[2] == 20:
        print("  ✓ Prediction data is 20×20 grid")
    else:
        print(f"  ✗ Prediction data is {pred_shape[1]}×{pred_shape[2]} grid (expected 20×20)")
    
    # Check variables
    print(f"\nVariable dimensions:")
    print(f"  Test variables: {test_shape[3]}")
    print(f"  Pred variables: {pred_shape[3]}")
    
    if test_shape[3] == 9 and pred_shape[3] == 9:
        print("  ✓ Both have 9 variables")
    else:
        print("  ✗ Variable count mismatch")
    
    # Check prediction start
    pred_start = pred_meta.get('prediction_info', {}).get('prediction_start_timestep', 25)
    print(f"\nPrediction timeline:")
    print(f"  Predictions start at timestep: {pred_start}")
    
    # Check if prediction data is non-zero after start timestep
    non_zero_count = 0
    zero_count = 0
    
    for t in range(pred_data.shape[0]):
        if np.any(pred_data[t] != 0):
            non_zero_count += 1
        else:
            zero_count += 1
    
    print(f"  Non-zero timesteps: {non_zero_count}")
    print(f"  Zero timesteps: {zero_count}")
    
    if non_zero_count > 0:
        print("  ✓ Prediction data contains non-zero values")
    else:
        print("  ✗ All prediction data is zero!")
    
    # Sample a specific location and timestep
    if pred_start < pred_data.shape[0]:
        sample_t = pred_start + 10
        x, y = 18, 11  # Ignition point
        
        if sample_t < pred_data.shape[0]:
            test_sample = test_data[sample_t, y, x, :]
            pred_sample = pred_data[sample_t, y, x, :]
            
            print(f"\nSample at timestep {sample_t}, location ({x}, {y}):")
            print(f"  Test data: {test_sample}")
            print(f"  Pred data: {pred_sample}")

if __name__ == "__main__":
    check_prediction_shapes()
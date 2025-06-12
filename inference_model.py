import numpy as np
import json
import glob
import os
import torch
import torch.nn as nn
from train_model import FirePredictionNet
import time

class FireInferenceEngine:
    def __init__(self, model_path="fire_prediction_model.pth"):
        """
        Initialize fire prediction inference engine for Ground Truth based prediction
        
        Args:
            model_path: Path to trained model weights
        """
        self.device = torch.device('mps' if torch.backends.mps.is_available() else 
                                 'cuda' if torch.cuda.is_available() else 'cpu')
        print(f"Loading model on device: {self.device}")
        
        # Load model with CORRECT parameters matching training
        self.model = FirePredictionNet(input_channels=9, hidden_dim=128, num_layers=3)
        
        if os.path.exists(model_path):
            checkpoint = torch.load(model_path, map_location=self.device)
            self.model.load_state_dict(checkpoint['model_state_dict'])
            print(f"Model loaded from {model_path}")
            print(f"Training loss: {checkpoint.get('train_loss', 'N/A'):.6f}")
            print(f"Validation loss: {checkpoint.get('val_loss', 'N/A'):.6f}")
        else:
            raise FileNotFoundError(f"Model file not found: {model_path}")
        
        self.model = self.model.to(self.device)
        self.model.eval()
        
        self.sequence_length = 20
        self.prediction_offset = 5
    
    def _normalize_data(self, data):
        """Apply same normalization as training"""
        normalized_data = data.copy()
        
        # Fire state (0-5) -> normalize to 0-1
        normalized_data[..., 0] = normalized_data[..., 0] / 5.0
        
        # Temperature (20-1200°C) -> normalize to 0-1
        normalized_data[..., 1] = (normalized_data[..., 1] - 20) / 1180
        
        # Smoke density (0-6) -> already reasonable range
        normalized_data[..., 2] = np.clip(normalized_data[..., 2] / 6.0, 0, 1)
        
        # Visibility (0-30m) -> normalize to 0-1
        normalized_data[..., 3] = normalized_data[..., 3] / 30.0
        
        # CO concentration (0-40000ppm) -> normalize to 0-1
        normalized_data[..., 4] = np.clip(normalized_data[..., 4] / 40000, 0, 1)
        
        # HCN concentration (0-6000ppm) -> normalize to 0-1
        normalized_data[..., 5] = np.clip(normalized_data[..., 5] / 6000, 0, 1)
        
        # Air velocity (0-6m/s) -> normalize to 0-1
        normalized_data[..., 6] = np.clip(normalized_data[..., 6] / 6.0, 0, 1)
        
        # Thermal radiation (0-100kW/m²) -> normalize to 0-1
        normalized_data[..., 7] = np.clip(normalized_data[..., 7] / 100, 0, 1)
        
        # Pressure (101000-102500Pa) -> normalize to 0-1
        normalized_data[..., 8] = (normalized_data[..., 8] - 101000) / 1500
        
        return normalized_data
    
    def _denormalize_data(self, normalized_data):
        """Reverse normalization to original scale with proper clipping"""
        denormalized_data = normalized_data.copy()
        
        # Fire state (0-1) -> (0-5) with clipping
        denormalized_data[..., 0] = np.clip(denormalized_data[..., 0] * 5.0, 0, 5)
        
        # Temperature (0-1) -> (20-1200°C) with clipping
        denormalized_data[..., 1] = np.clip(denormalized_data[..., 1] * 1180 + 20, 20, 1200)
        
        # Smoke density (0-1) -> (0-6) with clipping
        denormalized_data[..., 2] = np.clip(denormalized_data[..., 2] * 6.0, 0, 6)
        
        # Visibility (0-1) -> (0-30m) with clipping
        denormalized_data[..., 3] = np.clip(denormalized_data[..., 3] * 30.0, 0, 30)
        
        # CO concentration (0-1) -> (0-40000ppm) with clipping
        denormalized_data[..., 4] = np.clip(denormalized_data[..., 4] * 40000, 0, 40000)
        
        # HCN concentration (0-1) -> (0-6000ppm) with clipping
        denormalized_data[..., 5] = np.clip(denormalized_data[..., 5] * 6000, 0, 6000)
        
        # Air velocity (0-1) -> (0-6m/s) with clipping
        denormalized_data[..., 6] = np.clip(denormalized_data[..., 6] * 6.0, 0, 6)
        
        # Thermal radiation (0-1) -> (0-100kW/m²) with clipping
        denormalized_data[..., 7] = np.clip(denormalized_data[..., 7] * 100, 0, 100)
        
        # Pressure (0-1) -> (101000-102500Pa) with clipping
        denormalized_data[..., 8] = np.clip(denormalized_data[..., 8] * 1500 + 101000, 101000, 102500)
        
        return denormalized_data
    
    def load_test_data(self, json_file):
        """Load test simulation data"""
        try:
            with open(json_file, 'r') as f:
                metadata = json.load(f)
            
            bin_file = json_file.replace('.json', '.bin')
            if not os.path.exists(bin_file):
                raise FileNotFoundError(f"Binary file not found: {bin_file}")
            
            with open(bin_file, 'rb') as f:
                data_bytes = f.read()
            
            data_shape = tuple(metadata['data_shape'])
            data = np.frombuffer(data_bytes, dtype=np.float32).reshape(data_shape)
            
            return data, metadata
            
        except Exception as e:
            print(f"Error loading {json_file}: {e}")
            return None, None
    
    def predict_sequence(self, input_data):
        """
        Predict fire progression using Ground Truth input sequence
        
        Args:
            input_data: Input sequence [sequence_length, height, width, variables]
        
        Returns:
            Predicted frame [height, width, variables] - DENORMALIZED
        """
        # IMPORTANT: Normalize input data
        normalized_input = self._normalize_data(input_data)
        
        # Prepare input tensor
        # [sequence_length, height, width, variables] -> [1, sequence_length, variables, height, width]
        input_tensor = torch.FloatTensor(normalized_input).permute(0, 3, 1, 2).unsqueeze(0)
        input_tensor = input_tensor.to(self.device, non_blocking=True)
        
        with torch.no_grad():
            # Get prediction (normalized output)
            output = self.model(input_tensor)
            
            # Convert back to numpy
            # [1, variables, height, width] -> [height, width, variables]
            normalized_prediction = output.squeeze(0).permute(1, 2, 0).cpu().numpy()
        
        # IMPORTANT: Denormalize the prediction
        denormalized_prediction = self._denormalize_data(normalized_prediction)
        
        return denormalized_prediction
    
    def process_test_file(self, test_file, output_dir="pred_dataset"):
        """
        Process test file using Ground Truth based prediction (실제 관측 기반 예측)
        
        Args:
            test_file: Path to test data file
            output_dir: Directory to save predictions
        """
        print(f"Processing: {test_file}")
        
        # Load test data
        data, metadata = self.load_test_data(test_file)
        if data is None:
            print(f"Skipping {test_file} due to loading error")
            return
        
        timesteps, height, width, variables = data.shape
        
        # Create prediction array (same shape as input)
        pred_data = np.zeros_like(data)
        
        # Calculate prediction timeline
        # We need 20 frames as input to predict 5 frames later
        start_pred_timestep = self.sequence_length + self.prediction_offset
        
        print(f"Input shape: {data.shape}")
        print(f"Starting Ground Truth based predictions from timestep {start_pred_timestep}")
        print(f"Using 20 observed frames -> predict 5 steps ahead (실제 관측 기반)")
        
        # Generate predictions using Ground Truth input sequences (관측 기반 예측)
        prediction_count = 0
        for t in range(start_pred_timestep, timesteps):
            # Use Ground Truth frames as input: frames [t-25, t-5) to predict frame t
            input_start = t - self.prediction_offset - self.sequence_length  # t-25
            input_end = t - self.prediction_offset                           # t-5
            
            if input_start >= 0:
                # IMPORTANT: Always use Ground Truth data as input (관측된 프레임들)
                input_sequence = data[input_start:input_end]  # Ground Truth input
                
                # Make prediction for target frame (t)
                prediction = self.predict_sequence(input_sequence)
                
                # Store prediction at target timestep
                pred_data[t] = prediction
                prediction_count += 1
                
                if prediction_count % 50 == 0:
                    print(f"  Generated {prediction_count} GT-based predictions...")
                    
                # Debug: Print first few predictions to verify
                if prediction_count <= 3:
                    fire_state_sample = prediction[11, 18, 0]  # Fire state at ignition point
                    temp_sample = prediction[11, 18, 1]  # Temperature at ignition point
                    print(f"    T={t}: GT Input[{input_start}:{input_end}] -> Pred Fire={fire_state_sample:.2f}, Temp={temp_sample:.1f}°C")
        
        print(f"Total Ground Truth based predictions generated: {prediction_count}")
        
        # Save prediction data
        self._save_prediction_data(pred_data, metadata, test_file, output_dir)
    
    def _save_prediction_data(self, pred_data, metadata, original_file, output_dir):
        """Save prediction data in the same format as input"""
        os.makedirs(output_dir, exist_ok=True)
        
        # Get original filename
        base_name = os.path.basename(original_file)
        pred_bin_file = os.path.join(output_dir, base_name.replace('.json', '.bin'))
        pred_json_file = os.path.join(output_dir, base_name)
        
        # Convert to float32 and save binary data
        pred_data_f32 = pred_data.astype(np.float32)
        with open(pred_bin_file, 'wb') as f:
            f.write(pred_data_f32.tobytes())
        
        # Update metadata
        pred_metadata = metadata.copy()
        pred_metadata['prediction_info'] = {
            'model_type': 'ConvLSTM Fire Prediction - Ground Truth Based',
            'input_sequence_length': self.sequence_length,
            'prediction_offset': self.prediction_offset,
            'prediction_start_timestep': self.sequence_length + self.prediction_offset,
            'normalization_applied': True,
            'prediction_method': 'Ground Truth based observation prediction',
            'generated_at': time.strftime('%Y-%m-%d %H:%M:%S')
        }
        pred_metadata['data_shape'] = list(pred_data.shape)
        pred_metadata['file_size_bytes'] = int(pred_data_f32.nbytes)
        
        # Save metadata
        with open(pred_json_file, 'w') as f:
            json.dump(pred_metadata, f, indent=2)
        
        print(f"Prediction saved:")
        print(f"  Binary: {pred_bin_file}")
        print(f"  Metadata: {pred_json_file}")
        print(f"  Size: {pred_data_f32.nbytes / 1024 / 1024:.1f} MB")
    
    def process_all_test_files(self, test_dir="test_dataset", output_dir="pred_dataset"):
        """Process all test files in the test directory"""
        test_files = glob.glob(os.path.join(test_dir, "fire_simulation_*.json"))
        
        if not test_files:
            print(f"No test files found in {test_dir}")
            return
        
        print(f"Found {len(test_files)} test files to process")
        print("=" * 60)
        
        for i, test_file in enumerate(test_files, 1):
            print(f"\nProcessing file {i}/{len(test_files)}")
            try:
                self.process_test_file(test_file, output_dir)
                print("✓ Successfully processed")
            except Exception as e:
                print(f"✗ Error processing {test_file}: {e}")
            
            print("-" * 40)
        
        print(f"\nProcessing completed!")
        print(f"Predictions saved to: {output_dir}")


def main():
    print("Fire Simulation Prediction Inference - Ground Truth Based")
    print("=" * 50)
    
    # Check if model exists
    model_path = "fire_prediction_model.pth"
    if not os.path.exists(model_path):
        print(f"Error: Model file not found: {model_path}")
        print("Please run train_model.py first to train the model.")
        return
    
    try:
        # Initialize inference engine
        inference_engine = FireInferenceEngine(model_path)
        
        # Process all test files
        inference_engine.process_all_test_files()
        
        print("Ground Truth based inference completed successfully!")
        
    except Exception as e:
        print(f"Error during inference: {e}")


if __name__ == "__main__":
    main()
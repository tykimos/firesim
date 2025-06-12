import numpy as np
import json
import glob
import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import train_test_split
import time
import math

class FireSimulationDataset(Dataset):
    def __init__(self, data_files, sequence_length=20, prediction_offset=5, augment_data=True):
        """
        Advanced dataset for fire simulation prediction with spatial-temporal features
        """
        self.sequence_length = sequence_length
        self.prediction_offset = prediction_offset
        self.augment_data = augment_data
        self.samples = []
        
        print(f"Loading {len(data_files)} simulation files...")
        for file_path in data_files:
            self._load_simulation_file(file_path)
        
        print(f"Total samples created: {len(self.samples)}")
        if augment_data:
            print("Data augmentation enabled")
    
    def _load_simulation_file(self, json_file):
        """Load simulation file and create training samples"""
        try:
            with open(json_file, 'r') as f:
                metadata = json.load(f)
            
            bin_file = json_file.replace('.json', '.bin')
            if not os.path.exists(bin_file):
                print(f"Warning: Binary file not found: {bin_file}")
                return
            
            with open(bin_file, 'rb') as f:
                data_bytes = f.read()
            
            data_shape = tuple(metadata['data_shape'])
            data = np.frombuffer(data_bytes, dtype=np.float32).reshape(data_shape)
            
            total_timesteps = data_shape[0]
            
            # Create overlapping sequences with sliding window
            for start_idx in range(0, total_timesteps - self.sequence_length - self.prediction_offset + 1, 1):
                input_sequence = data[start_idx:start_idx + self.sequence_length]
                target_frame = data[start_idx + self.sequence_length + self.prediction_offset - 1]
                
                self.samples.append({
                    'input': input_sequence,
                    'target': target_frame,
                    'metadata': metadata
                })
                
        except Exception as e:
            print(f"Error loading {json_file}: {e}")
    
    def __len__(self):
        return len(self.samples)
    
    def __getitem__(self, idx):
        sample = self.samples[idx]
        
        input_data = sample['input'].copy()
        target_data = sample['target'].copy()
        
        # Apply data augmentation
        if self.augment_data:
            input_data, target_data = self._apply_augmentation(input_data, target_data)
        
        # Normalize data
        input_data = self._normalize_data(input_data)
        target_data = self._normalize_data(target_data)
        
        # Convert to tensors: [sequence_length, height, width, variables] -> [sequence_length, variables, height, width]
        input_tensor = torch.FloatTensor(input_data).permute(0, 3, 1, 2)
        target_tensor = torch.FloatTensor(target_data).permute(2, 0, 1)
        
        return input_tensor, target_tensor
    
    def _normalize_data(self, data):
        """Normalize data for better training"""
        # Variable-specific normalization
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
    
    def _apply_augmentation(self, input_seq, target_frame):
        """Apply spatial augmentations"""
        import random
        
        # Random horizontal flip
        if random.random() < 0.5:
            input_seq = np.flip(input_seq, axis=2).copy()
            target_frame = np.flip(target_frame, axis=1).copy()
        
        # Random vertical flip
        if random.random() < 0.5:
            input_seq = np.flip(input_seq, axis=1).copy()
            target_frame = np.flip(target_frame, axis=0).copy()
        
        # Random 90-degree rotations
        rot_choice = random.randint(0, 3)
        if rot_choice > 0:
            input_seq = np.rot90(input_seq, k=rot_choice, axes=(1, 2)).copy()
            target_frame = np.rot90(target_frame, k=rot_choice, axes=(0, 1)).copy()
        
        return input_seq, target_frame


class SpatialTemporalAttention(nn.Module):
    def __init__(self, channels, spatial_size):
        super(SpatialTemporalAttention, self).__init__()
        self.channels = channels
        self.spatial_size = spatial_size
        
        # Spatial attention
        self.spatial_attention = nn.Sequential(
            nn.Conv2d(channels, channels // 4, 1),
            nn.ReLU(),
            nn.Conv2d(channels // 4, 1, 1),
            nn.Sigmoid()
        )
        
        # Channel attention
        self.channel_attention = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(channels, channels // 4, 1),
            nn.ReLU(),
            nn.Conv2d(channels // 4, channels, 1),
            nn.Sigmoid()
        )
    
    def forward(self, x):
        # x: [batch, channels, height, width]
        
        # Spatial attention
        spatial_att = self.spatial_attention(x)
        x_spatial = x * spatial_att
        
        # Channel attention
        channel_att = self.channel_attention(x_spatial)
        x_attended = x_spatial * channel_att
        
        return x_attended


class AdvancedConvLSTMCell(nn.Module):
    def __init__(self, input_dim, hidden_dim, kernel_size, dropout=0.1):
        super(AdvancedConvLSTMCell, self).__init__()
        
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.kernel_size = kernel_size
        self.padding = kernel_size // 2
        
        # ConvLSTM gates
        self.conv_gates = nn.Conv2d(
            in_channels=self.input_dim + self.hidden_dim,
            out_channels=4 * self.hidden_dim,
            kernel_size=self.kernel_size,
            padding=self.padding,
            bias=True
        )
        
        # Dropout for regularization
        self.dropout = nn.Dropout2d(dropout)
        
        # Layer normalization
        self.layer_norm = nn.GroupNorm(4, 4 * self.hidden_dim)
    
    def forward(self, input_tensor, cur_state):
        h_cur, c_cur = cur_state
        
        # Concatenate input and hidden state
        combined = torch.cat([input_tensor, h_cur], dim=1)
        
        # Apply convolution and normalization
        combined_conv = self.conv_gates(combined)
        combined_conv = self.layer_norm(combined_conv)
        
        # Split into gates
        cc_i, cc_f, cc_o, cc_g = torch.split(combined_conv, self.hidden_dim, dim=1)
        
        # Apply gates with improved activations
        i = torch.sigmoid(cc_i)
        f = torch.sigmoid(cc_f)
        o = torch.sigmoid(cc_o)
        g = torch.tanh(cc_g)
        
        # Update cell state
        c_next = f * c_cur + i * g
        h_next = o * torch.tanh(c_next)
        
        # Apply dropout
        h_next = self.dropout(h_next)
        
        return h_next, c_next


class FirePredictionNet(nn.Module):
    def __init__(self, input_channels=9, hidden_dim=128, num_layers=3):
        super(FirePredictionNet, self).__init__()
        
        self.input_channels = input_channels
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        
        # Enhanced feature extraction with residual connections
        self.feature_extractor = nn.Sequential(
            nn.Conv2d(input_channels, 64, 3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.Conv2d(64, 64, 3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.Conv2d(64, hidden_dim, 3, padding=1),
            nn.BatchNorm2d(hidden_dim),
            nn.ReLU()
        )
        
        # Multi-scale feature extraction
        self.multi_scale = nn.ModuleList([
            nn.Conv2d(hidden_dim, hidden_dim // 4, 1),  # 1x1
            nn.Conv2d(hidden_dim, hidden_dim // 4, 3, padding=1),  # 3x3
            nn.Conv2d(hidden_dim, hidden_dim // 4, 5, padding=2),  # 5x5
            nn.Conv2d(hidden_dim, hidden_dim // 4, 7, padding=3),  # 7x7
        ])
        
        # Advanced ConvLSTM layers
        self.conv_lstm_layers = nn.ModuleList([
            AdvancedConvLSTMCell(hidden_dim, hidden_dim, 3, dropout=0.1)
            for _ in range(num_layers)
        ])
        
        # Spatial-temporal attention
        self.attention = SpatialTemporalAttention(hidden_dim, 20)
        
        # Enhanced decoder with skip connections
        self.decoder = nn.Sequential(
            nn.Conv2d(hidden_dim, 128, 3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(),
            nn.Conv2d(128, 64, 3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.Conv2d(64, 32, 3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.Conv2d(32, input_channels, 1)
        )
        
        # Variable-specific heads for better prediction
        self.variable_heads = nn.ModuleList([
            nn.Sequential(
                nn.Conv2d(input_channels, 16, 1),
                nn.ReLU(),
                nn.Conv2d(16, 1, 1)
            ) for _ in range(input_channels)
        ])
        
        # Initialize weights
        self._initialize_weights()
    
    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
    
    def forward(self, x):
        batch_size, seq_len, channels, height, width = x.size()
        
        # Initialize hidden states for all layers
        h_states = []
        c_states = []
        for _ in range(self.num_layers):
            h = torch.zeros(batch_size, self.hidden_dim, height, width).to(x.device)
            c = torch.zeros(batch_size, self.hidden_dim, height, width).to(x.device)
            h_states.append(h)
            c_states.append(c)
        
        # Process sequence
        for t in range(seq_len):
            current_input = x[:, t]
            
            # Extract features
            features = self.feature_extractor(current_input)
            
            # Multi-scale feature fusion
            multi_scale_features = []
            for conv in self.multi_scale:
                multi_scale_features.append(conv(features))
            features = torch.cat(multi_scale_features, dim=1)
            
            # Pass through ConvLSTM layers
            layer_input = features
            for i in range(self.num_layers):
                h_states[i], c_states[i] = self.conv_lstm_layers[i](
                    layer_input, (h_states[i], c_states[i])
                )
                layer_input = h_states[i]
        
        # Apply attention to final hidden state
        attended_features = self.attention(h_states[-1])
        
        # Decode to prediction
        decoded = self.decoder(attended_features)
        
        # Apply variable-specific heads
        variable_outputs = []
        for i, head in enumerate(self.variable_heads):
            var_output = head(decoded)
            variable_outputs.append(var_output)
        
        # Concatenate variable outputs
        output = torch.cat(variable_outputs, dim=1)
        
        return output


class WeightedMSELoss(nn.Module):
    def __init__(self):
        super(WeightedMSELoss, self).__init__()
        
        # Variable-specific weights (higher for more important variables)
        self.weights = torch.tensor([
            3.0,  # Fire state (most important)
            2.0,  # Temperature
            1.5,  # Smoke density
            1.5,  # Visibility
            1.0,  # CO concentration
            1.0,  # HCN concentration
            1.0,  # Air velocity
            1.5,  # Thermal radiation
            0.5   # Pressure (least important)
        ])
    
    def forward(self, predictions, targets):
        device = predictions.device
        weights = self.weights.to(device)
        
        # Calculate MSE for each variable
        mse_per_var = torch.mean((predictions - targets) ** 2, dim=(0, 2, 3))
        
        # Apply weights
        weighted_mse = torch.sum(mse_per_var * weights) / torch.sum(weights)
        
        return weighted_mse


def load_training_data(train_dir="train_dataset"):
    """Load all training data files"""
    json_files = glob.glob(os.path.join(train_dir, "fire_simulation_*.json"))
    
    if not json_files:
        raise ValueError(f"No training data found in {train_dir}")
    
    print(f"Found {len(json_files)} training files")
    return json_files


def train_model(model, train_loader, val_loader, num_epochs=100, learning_rate=0.001):
    """Train the advanced fire prediction model with M3 Mac acceleration"""
    # Check for M3 Mac MPS support
    if torch.backends.mps.is_available():
        device = torch.device('mps')
        print(f"Training on Apple M3 Mac GPU (MPS): {device}")
    elif torch.cuda.is_available():
        device = torch.device('cuda')
        print(f"Training on CUDA GPU: {device}")
    else:
        device = torch.device('cpu')
        print(f"Training on CPU: {device}")
    
    print(f"PyTorch version: {torch.__version__}")
    if device.type == 'mps':
        print("MPS backend enabled for Apple Silicon acceleration")
    
    model = model.to(device)
    
    # Use weighted loss
    criterion = WeightedMSELoss()
    
    # Advanced optimizer with scheduler - optimized for M3
    optimizer = optim.AdamW(
        model.parameters(), 
        lr=learning_rate, 
        weight_decay=1e-4,
        eps=1e-7 if device.type == 'mps' else 1e-8  # Better numerical stability on MPS
    )
    scheduler = optim.lr_scheduler.CosineAnnealingWarmRestarts(optimizer, T_0=10, T_mult=2)
    
    # Enable MPS optimizations
    if device.type == 'mps':
        # Enable MPS high memory efficiency
        torch.mps.empty_cache()
        print("MPS cache cleared and optimizations enabled")
    
    best_val_loss = float('inf')
    patience = 15
    patience_counter = 0
    
    print("Starting advanced training...")
    
    for epoch in range(num_epochs):
        # Training phase
        model.train()
        train_loss = 0.0
        
        for batch_idx, (inputs, targets) in enumerate(train_loader):
            # Move data to device with non_blocking for MPS
            non_blocking = device.type == 'mps'
            inputs = inputs.to(device, non_blocking=non_blocking)
            targets = targets.to(device, non_blocking=non_blocking)
            
            optimizer.zero_grad()
            
            # Forward pass with MPS optimizations
            outputs = model(inputs)
            loss = criterion(outputs, targets)
            
            # Backward pass
            loss.backward()
            
            # Gradient clipping
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            
            optimizer.step()
            train_loss += loss.item()
            
            # Clear MPS cache periodically to prevent memory issues
            if device.type == 'mps' and batch_idx % 20 == 0:
                torch.mps.empty_cache()
            
            if batch_idx % 50 == 0:
                print(f'Epoch {epoch+1}/{num_epochs}, Batch {batch_idx}/{len(train_loader)}, Loss: {loss.item():.6f}')
                if device.type == 'mps':
                    print(f'  MPS Memory Usage: {torch.mps.current_allocated_memory() / 1024**2:.1f} MB')
        
        # Validation phase
        model.eval()
        val_loss = 0.0
        
        with torch.no_grad():
            for inputs, targets in val_loader:
                # Move data to device with non_blocking for MPS
                non_blocking = device.type == 'mps'
                inputs = inputs.to(device, non_blocking=non_blocking)
                targets = targets.to(device, non_blocking=non_blocking)
                outputs = model(inputs)
                val_loss += criterion(outputs, targets).item()
            
            # Clear MPS cache after validation
            if device.type == 'mps':
                torch.mps.empty_cache()
        
        # Calculate average losses
        train_loss = train_loss / len(train_loader)
        val_loss = val_loss / len(val_loader)
        
        print(f'Epoch {epoch+1}/{num_epochs}:')
        print(f'  Train Loss: {train_loss:.6f}')
        print(f'  Val Loss: {val_loss:.6f}')
        print(f'  Learning Rate: {scheduler.get_last_lr()[0]:.6f}')
        
        # Learning rate scheduling
        scheduler.step()
        
        # Early stopping and model saving
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            patience_counter = 0
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'train_loss': train_loss,
                'val_loss': val_loss,
                'scheduler_state_dict': scheduler.state_dict(),
            }, 'fire_prediction_model.pth')
            print(f'  New best model saved! Val Loss: {val_loss:.6f}')
        else:
            patience_counter += 1
            
        if patience_counter >= patience:
            print(f'Early stopping triggered after {patience} epochs without improvement')
            break
        
        print('-' * 60)
    
    return best_val_loss


def main():
    print("Advanced Fire Simulation Prediction Model Training")
    print("=" * 60)
    
    # Load training data
    try:
        data_files = load_training_data()
    except ValueError as e:
        print(f"Error: {e}")
        return
    
    # Create datasets
    print("Creating advanced training dataset...")
    train_dataset_full = FireSimulationDataset(data_files, sequence_length=20, prediction_offset=5, augment_data=True)
    
    print("Creating validation dataset...")
    val_dataset_full = FireSimulationDataset(data_files, sequence_length=20, prediction_offset=5, augment_data=False)
    
    if len(train_dataset_full) == 0:
        print("Error: No valid training samples found.")
        return
    
    # Split datasets
    train_size = int(0.8 * len(train_dataset_full))
    val_size = len(val_dataset_full) - train_size
    
    indices = list(range(len(train_dataset_full)))
    train_indices = indices[:train_size]
    val_indices = indices[train_size:train_size + val_size]
    
    train_dataset = torch.utils.data.Subset(train_dataset_full, train_indices)
    val_dataset = torch.utils.data.Subset(val_dataset_full, val_indices)
    
    print(f"Training samples: {len(train_dataset)}")
    print(f"Validation samples: {len(val_dataset)}")
    
    # Create data loaders - optimized for M3 Mac
    # Increase batch size for better GPU utilization on M3
    batch_size = 4 if torch.backends.mps.is_available() else 2
    # Use pin_memory for faster GPU transfer on MPS
    pin_memory = torch.backends.mps.is_available()
    
    train_loader = DataLoader(
        train_dataset, 
        batch_size=batch_size, 
        shuffle=True, 
        num_workers=0,  # Keep 0 for MPS compatibility
        pin_memory=pin_memory,
        persistent_workers=False
    )
    val_loader = DataLoader(
        val_dataset, 
        batch_size=batch_size, 
        shuffle=False, 
        num_workers=0,
        pin_memory=pin_memory,
        persistent_workers=False
    )
    
    # Create advanced model
    model = FirePredictionNet(input_channels=9, hidden_dim=128, num_layers=3)
    total_params = sum(p.numel() for p in model.parameters())
    print(f"Advanced model created with {total_params:,} parameters")
    
    # Train model
    best_loss = train_model(model, train_loader, val_loader, num_epochs=5)
    
    print("Advanced training completed!")
    print(f"Best validation loss: {best_loss:.6f}")
    print("Model saved as 'fire_prediction_model.pth'")


if __name__ == "__main__":
    main()
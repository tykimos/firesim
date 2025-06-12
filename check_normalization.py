import numpy as np
import json
import glob
from train_model import FireSimulationDataset

def check_normalization_consistency():
    print("=== 정규화 일치성 확인 ===")
    
    # 1. 훈련 데이터의 정규화 확인
    print("1. 훈련 데이터 정규화 확인")
    train_files = glob.glob("train_dataset/fire_simulation_*.json")
    if train_files:
        train_dataset = FireSimulationDataset(train_files[:1], sequence_length=20, prediction_offset=5, augment_data=False)
        
        if len(train_dataset) > 0:
            # 첫 번째 샘플 가져오기
            input_tensor, target_tensor = train_dataset[0]
            print(f"훈련 입력 shape: {input_tensor.shape}")
            print(f"훈련 타겟 shape: {target_tensor.shape}")
            
            # 정규화된 값 범위 확인
            for var_idx in range(9):
                input_var = input_tensor[:, var_idx, :, :].numpy()
                target_var = target_tensor[var_idx, :, :].numpy()
                
                print(f"  Var {var_idx}: Input range=[{np.min(input_var):.4f}, {np.max(input_var):.4f}], Target range=[{np.min(target_var):.4f}, {np.max(target_var):.4f}]")
    
    # 2. 테스트 데이터의 원본값 확인
    print("\n2. 테스트 데이터 원본값 확인")
    with open('test_dataset/fire_simulation_20250611_220929.json', 'r') as f:
        metadata = json.load(f)
    
    with open('test_dataset/fire_simulation_20250611_220929.bin', 'rb') as f:
        data_bytes = f.read()
    
    data_shape = tuple(metadata['data_shape'])
    test_data = np.frombuffer(data_bytes, dtype=np.float32).reshape(data_shape)
    
    print(f"테스트 데이터 shape: {test_data.shape}")
    
    # 각 변수별 원본 값 범위
    for var_idx in range(9):
        var_data = test_data[:, :, :, var_idx]
        print(f"  Var {var_idx}: 원본 range=[{np.min(var_data):.2f}, {np.max(var_data):.2f}], mean={np.mean(var_data):.2f}")
    
    # 3. 점화 지점의 특정 시점 값들 확인
    print("\n3. 점화 지점 특정 시점 정규화 확인")
    ignition_x, ignition_y = 18, 11
    
    for t in [0, 10, 20]:
        print(f"\nT={t}:")
        original_values = test_data[t, ignition_y, ignition_x, :]
        
        # 수동 정규화 적용
        normalized_manual = original_values.copy()
        normalized_manual[0] = normalized_manual[0] / 5.0  # Fire state
        normalized_manual[1] = (normalized_manual[1] - 20) / 1180  # Temperature
        normalized_manual[2] = np.clip(normalized_manual[2] / 6.0, 0, 1)  # Smoke
        normalized_manual[3] = normalized_manual[3] / 30.0  # Visibility
        normalized_manual[4] = np.clip(normalized_manual[4] / 40000, 0, 1)  # CO
        normalized_manual[5] = np.clip(normalized_manual[5] / 6000, 0, 1)  # HCN
        normalized_manual[6] = np.clip(normalized_manual[6] / 6.0, 0, 1)  # Air velocity
        normalized_manual[7] = np.clip(normalized_manual[7] / 100, 0, 1)  # Thermal radiation
        normalized_manual[8] = (normalized_manual[8] - 101000) / 1500  # Pressure
        
        print(f"  원본: Fire={original_values[0]:.2f}, Temp={original_values[1]:.1f}°C")
        print(f"  정규화: Fire={normalized_manual[0]:.4f}, Temp={normalized_manual[1]:.4f}")
    
    # 4. 훈련 데이터 샘플과 비교
    print("\n4. 훈련 데이터 샘플의 정규화된 값 확인")
    if len(train_dataset) > 0:
        # 첫 번째 샘플의 첫 번째 프레임
        input_tensor, target_tensor = train_dataset[0]
        
        # 점화 지점 (18, 11)에서의 값
        train_fire_input = input_tensor[0, 0, ignition_y, ignition_x].item()
        train_temp_input = input_tensor[0, 1, ignition_y, ignition_x].item()
        train_fire_target = target_tensor[0, ignition_y, ignition_x].item()
        train_temp_target = target_tensor[1, ignition_y, ignition_x].item()
        
        print(f"  훈련 입력: Fire={train_fire_input:.4f}, Temp={train_temp_input:.4f}")
        print(f"  훈련 타겟: Fire={train_fire_target:.4f}, Temp={train_temp_target:.4f}")
        
        # 정규화 범위 확인
        if train_fire_input < 0 or train_fire_input > 1:
            print("  ⚠️ 훈련 데이터의 Fire State가 정규화 범위 [0,1]을 벗어남!")
        if train_temp_input < 0 or train_temp_input > 1:
            print("  ⚠️ 훈련 데이터의 Temperature가 정규화 범위 [0,1]을 벗어남!")

if __name__ == "__main__":
    check_normalization_consistency() 
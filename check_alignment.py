import numpy as np
import json

def check_temporal_alignment():
    print("=== 시간적 정렬 문제 확인 ===")
    
    # 예측 데이터 로드
    with open('pred_dataset/fire_simulation_20250611_220929.json', 'r') as f:
        pred_meta = json.load(f)

    with open('pred_dataset/fire_simulation_20250611_220929.bin', 'rb') as f:
        pred_bytes = f.read()

    pred_shape = tuple(pred_meta['data_shape'])
    pred_data = np.frombuffer(pred_bytes, dtype=np.float32).reshape(pred_shape)

    # Ground Truth 데이터 로드
    with open('test_dataset/fire_simulation_20250611_220929.json', 'r') as f:
        gt_meta = json.load(f)

    with open('test_dataset/fire_simulation_20250611_220929.bin', 'rb') as f:
        gt_bytes = f.read()

    gt_shape = tuple(gt_meta['data_shape'])
    gt_data = np.frombuffer(gt_bytes, dtype=np.float32).reshape(gt_shape)

    print(f'GT shape: {gt_data.shape}')
    print(f'Pred shape: {pred_data.shape}')
    
    # 예측 시작 시점 확인
    pred_start = pred_meta['prediction_info']['prediction_start_timestep']
    print(f'Prediction starts from timestep: {pred_start}')

    print('\n=== 점화 지점 (18, 11) Fire State 비교 ===')
    ignition_x, ignition_y = 18, 11
    
    for t in [0, 5, 10, 15, 20, 25, 26, 27, 30, 35, 40, 50]:
        if t < gt_data.shape[0]:
            gt_fire = gt_data[t, ignition_y, ignition_x, 0]
            pred_fire = pred_data[t, ignition_y, ignition_x, 0]
            gt_temp = gt_data[t, ignition_y, ignition_x, 1]
            pred_temp = pred_data[t, ignition_y, ignition_x, 1]
            
            status = "✓" if pred_fire != 0.0 else "✗"
            print(f'T={t:2d}: GT Fire={gt_fire:.2f} Temp={gt_temp:4.0f}°C | Pred Fire={pred_fire:.2f} Temp={pred_temp:4.0f}°C {status}')

    print('\n=== 처음 25개 프레임 예측값 확인 ===')
    non_zero_count = 0
    for t in range(25):
        pred_fire = pred_data[t, ignition_y, ignition_x, 0]
        pred_temp = pred_data[t, ignition_y, ignition_x, 1]
        if pred_fire != 0 or pred_temp != 0:
            print(f'T={t}: Pred Fire={pred_fire:.2f}, Temp={pred_temp:.2f} (Non-zero!)')
            non_zero_count += 1
    
    if non_zero_count == 0:
        print("처음 25개 프레임이 모두 0 -> 올바름")
    else:
        print(f"처음 25개 프레임 중 {non_zero_count}개가 non-zero -> 문제 있음")
    
    print('\n=== 실제 화재 진행 시점 비교 ===')
    # Ground Truth에서 화재가 시작되는 시점 찾기
    gt_ignition_time = None
    for t in range(gt_data.shape[0]):
        if gt_data[t, ignition_y, ignition_x, 0] > 0.5:  # Fire state > 0.5
            gt_ignition_time = t
            break
    
    # Prediction에서 화재가 시작되는 시점 찾기
    pred_ignition_time = None
    for t in range(pred_data.shape[0]):
        if pred_data[t, ignition_y, ignition_x, 0] > 0.5:
            pred_ignition_time = t
            break
    
    print(f'GT 점화 시점: T={gt_ignition_time}')
    print(f'Pred 점화 시점: T={pred_ignition_time}')
    
    if pred_ignition_time and gt_ignition_time:
        delay = pred_ignition_time - gt_ignition_time
        print(f'예측 지연: {delay} timesteps ({delay} seconds)')
        
        if delay > 5:
            print("⚠️  예측이 크게 지연되고 있습니다!")
        else:
            print("✓ 예측 지연이 허용 범위 내입니다.")

if __name__ == "__main__":
    check_temporal_alignment() 
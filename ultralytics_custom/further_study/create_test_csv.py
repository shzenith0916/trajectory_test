#!/usr/bin/env python3
"""
테스트용 CSV 파일을 생성하는 스크립트
실제 궤적 데이터가 없을 때 스파이크 제거 스크립트를 테스트하기 위해 사용
"""

import pandas as pd
import numpy as np
from pathlib import Path


def create_sample_trajectory_with_spikes():
    """
    스파이크가 포함된 샘플 궤적 데이터 생성
    """
    # 기본 궤적 (타원형)
    t = np.linspace(0, 2*np.pi, 100)
    center_x, center_y = 500, 300
    radius_x, radius_y = 200, 100

    # 기본 타원형 궤적
    x_base = center_x + radius_x * np.cos(t)
    y_base = center_y + radius_y * np.sin(t)

    # 약간의 노이즈 추가
    noise_x = np.random.normal(0, 5, len(x_base))
    noise_y = np.random.normal(0, 5, len(y_base))

    x_noisy = x_base + noise_x
    y_noisy = y_base + noise_y

    # 스파이크 추가 (이상치)
    spike_indices = np.random.choice(len(x_base), size=15, replace=False)

    for idx in spike_indices:
        # 랜덤한 방향으로 큰 스파이크 추가
        spike_x = np.random.normal(0, 100)
        spike_y = np.random.normal(0, 100)

        x_noisy[idx] += spike_x
        y_noisy[idx] += spike_y

    return x_noisy, y_noisy


def create_test_csv_files():
    """
    테스트용 CSV 파일들을 생성
    """
    # 출력 디렉토리 생성
    test_dir = Path("test_csv_data")
    test_dir.mkdir(exist_ok=True)

    # 여러 개의 테스트 파일 생성
    for i in range(5):
        x_coords, y_coords = create_sample_trajectory_with_spikes()

        df = pd.DataFrame({
            'red_dot_x': np.random.normal(300, 50, len(x_coords)),
            'red_dot_y': np.random.normal(200, 30, len(x_coords)),
            'blue_dot_x': x_coords,
            'blue_dot_y': y_coords,
            'frame_w': 1000,
            'frame_h': 600,
            'red_dot_x_norm': np.random.normal(0.3, 0.05, len(x_coords)),
            'red_dot_y_norm': np.random.normal(0.2, 0.03, len(x_coords)),
            'blue_dot_x_norm': x_coords / 1000,
            'blue_dot_y_norm': y_coords / 600,
        })

        csv_path = test_dir / f"test_video_{i+1}_trajectories.csv"
        df.to_csv(csv_path, index=False, encoding='utf-8-sig')
        print(f"Created test CSV: {csv_path}")

    print(f"\nTest CSV files created in: {test_dir}")
    return test_dir


if __name__ == '__main__':
    test_dir = create_test_csv_files()
    print(f"\nYou can now test the spike removal script with:")
    print(f"python spike_removal_comparison.py --root {test_dir}")

#!/usr/bin/env python3

# 개별 CSV 파일의 궤적에서 스파이크를 제거하고 원본과 비교하는 스크립트

# 이 스크립트는:
# 1. 각 CSV 파일의 blue_dot 궤적을 읽어옴
# 2. 다양한 방법으로 스파이크(이상치) 제거
# 3. 원본과 스파이크 제거된 결과를 이미지로 비교 시각화
# 4. 여러 방법의 결과를 나란히 비교
#

import argparse
import numpy as np
import pandas as pd
from pathlib import Path
import matplotlib.pyplot as plt
from matplotlib.patches import Circle
import seaborn as sns
from scipy import ndimage
from scipy.signal import savgol_filter
from sklearn.cluster import DBSCAN
from sklearn.neighbors import LocalOutlierFactor
from sklearn.ensemble import IsolationForest
from sklearn.covariance import EllipticEnvelope
from sklearn.preprocessing import StandardScaler
import warnings
warnings.filterwarnings('ignore')

# 한글 폰트 설정
plt.rcParams['font.family'] = 'DejaVu Sans'
plt.rcParams['axes.unicode_minus'] = False


def load_trajectory_from_csv(csv_path: Path) -> tuple[np.ndarray, np.ndarray, int, int]:
    """
    단일 CSV 파일에서 궤적 데이터를 로드합니다.

    Args:
        csv_path: CSV 파일 경로

    Returns:
        (x_coords, y_coords, frame_w, frame_h): 좌표와 프레임 크기
    """
    try:
        df = pd.read_csv(csv_path)

        # blue_dot 좌표 추출
        if 'blue_dot_x' in df.columns and 'blue_dot_y' in df.columns:
            x_coords = df['blue_dot_x'].dropna().to_numpy()
            y_coords = df['blue_dot_y'].dropna().to_numpy()

            # 프레임 크기 정보
            frame_w = int(df['frame_w'].iloc[0]
                          ) if 'frame_w' in df.columns else 1920
            frame_h = int(df['frame_h'].iloc[0]
                          ) if 'frame_h' in df.columns else 1080

            return x_coords, y_coords, frame_w, frame_h
        else:
            print(f"Warning: blue_dot_x/y columns not found in {csv_path}")
            return np.array([]), np.array([]), 0, 0

    except Exception as e:
        print(f"Error reading {csv_path}: {e}")
        return np.array([]), np.array([]), 0, 0


def remove_spikes_iqr(x_coords, y_coords, multiplier=1.5):
    """
    IQR 기반 스파이크 제거
    """
    if len(x_coords) < 3:
        return x_coords, y_coords

    # X, Y 각각에 대해 IQR 계산
    x_q1, x_q3 = np.percentile(x_coords, [25, 75])
    y_q1, y_q3 = np.percentile(y_coords, [25, 75])

    x_iqr = x_q3 - x_q1
    y_iqr = y_q3 - y_q1

    # 이상치 범위 설정
    x_lower = x_q1 - multiplier * x_iqr
    x_upper = x_q3 + multiplier * x_iqr
    y_lower = y_q1 - multiplier * y_iqr
    y_upper = y_q3 + multiplier * y_iqr

    # 이상치가 아닌 점들만 선택
    mask = ((x_coords >= x_lower) & (x_coords <= x_upper) &
            (y_coords >= y_lower) & (y_coords <= y_upper))

    return x_coords[mask], y_coords[mask]


def remove_spikes_dbscan(x_coords, y_coords, eps=0.1, min_samples=3):
    """
    DBSCAN 클러스터링을 이용한 스파이크 제거
    """
    if len(x_coords) < 3:
        return x_coords, y_coords

    # 좌표를 정규화
    scaler = StandardScaler()
    coords_scaled = scaler.fit_transform(np.column_stack([x_coords, y_coords]))

    # DBSCAN 적용
    dbscan = DBSCAN(eps=eps, min_samples=min_samples)
    labels = dbscan.fit_predict(coords_scaled)

    # 노이즈가 아닌 점들만 선택 (label != -1)
    mask = labels != -1

    return x_coords[mask], y_coords[mask]


def remove_spikes_lof(x_coords, y_coords, contamination=0.1):
    """
    Local Outlier Factor를 이용한 스파이크 제거
    """
    if len(x_coords) < 3:
        return x_coords, y_coords

    # 좌표를 정규화
    scaler = StandardScaler()
    coords_scaled = scaler.fit_transform(np.column_stack([x_coords, y_coords]))

    # LOF 적용
    lof = LocalOutlierFactor(contamination=contamination)
    labels = lof.fit_predict(coords_scaled)

    # 이상치가 아닌 점들만 선택 (label != -1)
    mask = labels != -1

    return x_coords[mask], y_coords[mask]


def remove_spikes_isolation_forest(x_coords, y_coords, contamination=0.1):
    """
    Isolation Forest를 이용한 스파이크 제거
    """
    if len(x_coords) < 3:
        return x_coords, y_coords

    # 좌표를 정규화
    scaler = StandardScaler()
    coords_scaled = scaler.fit_transform(np.column_stack([x_coords, y_coords]))

    # Isolation Forest 적용
    iso_forest = IsolationForest(contamination=contamination, random_state=42)
    labels = iso_forest.fit_predict(coords_scaled)

    # 이상치가 아닌 점들만 선택 (label != -1)
    mask = labels != -1

    return x_coords[mask], y_coords[mask]


def remove_spikes_elliptic_envelope(x_coords, y_coords, contamination=0.1):
    """
    Elliptic Envelope를 이용한 스파이크 제거
    """
    if len(x_coords) < 3:
        return x_coords, y_coords

    # 좌표를 정규화
    scaler = StandardScaler()
    coords_scaled = scaler.fit_transform(np.column_stack([x_coords, y_coords]))

    # Elliptic Envelope 적용
    elliptic_env = EllipticEnvelope(
        contamination=contamination, random_state=42)
    labels = elliptic_env.fit_predict(coords_scaled)

    # 이상치가 아닌 점들만 선택 (label != -1)
    mask = labels != -1

    return x_coords[mask], y_coords[mask]


def remove_spikes_savgol_filter(x_coords, y_coords, window_length=5, polyorder=2):
    """
    Savitzky-Golay 필터를 이용한 스파이크 제거
    """
    if len(x_coords) < window_length:
        return x_coords, y_coords

    # x 좌표로 정렬
    sorted_indices = np.argsort(x_coords)
    x_sorted = x_coords[sorted_indices]
    y_sorted = y_coords[sorted_indices]

    # Savitzky-Golay 필터 적용
    try:
        y_filtered = savgol_filter(y_sorted, window_length, polyorder)

        # 필터링된 결과와 원본의 차이가 큰 점들을 스파이크로 간주
        diff = np.abs(y_sorted - y_filtered)
        threshold = np.std(diff) * 2  # 표준편차의 2배를 임계값으로 사용

        mask = diff <= threshold

        return x_sorted[mask], y_sorted[mask]
    except:
        return x_coords, y_coords


def remove_spikes_moving_median(x_coords, y_coords, window_size=5):
    """
    이동 중앙값을 이용한 스파이크 제거
    """
    if len(x_coords) < window_size:
        return x_coords, y_coords

    # x 좌표로 정렬
    sorted_indices = np.argsort(x_coords)
    x_sorted = x_coords[sorted_indices]
    y_sorted = y_coords[sorted_indices]

    # 이동 중앙값 계산
    y_median = ndimage.median_filter(y_sorted, size=window_size)

    # 중앙값과의 차이가 큰 점들을 스파이크로 간주
    diff = np.abs(y_sorted - y_median)
    threshold = np.std(diff) * 2

    mask = diff <= threshold

    return x_sorted[mask], y_sorted[mask]


def remove_spikes_distance_based(x_coords, y_coords, max_distance=50):
    """
    거리 기반 스파이크 제거 - 연속된 점들 간의 거리가 너무 큰 점들을 제거
    """
    if len(x_coords) < 3:
        return x_coords, y_coords

    # x 좌표로 정렬
    sorted_indices = np.argsort(x_coords)
    x_sorted = x_coords[sorted_indices]
    y_sorted = y_coords[sorted_indices]

    # 연속된 점들 간의 거리 계산
    distances = np.sqrt(np.diff(x_sorted)**2 + np.diff(y_sorted)**2)

    # 거리가 임계값을 초과하는 점들의 인덱스 찾기
    spike_indices = np.where(distances > max_distance)[0]

    # 스파이크 인덱스를 실제 좌표 인덱스로 변환 (다음 점이 스파이크)
    spike_indices = spike_indices + 1

    # 스파이크가 아닌 점들만 선택
    mask = np.ones(len(x_sorted), dtype=bool)
    mask[spike_indices] = False

    return x_sorted[mask], y_sorted[mask]


def remove_spikes_cluster_based(x_coords, y_coords, min_cluster_size=3):
    """
    클러스터링 기반 스파이크 제거 - 주요 클러스터에서 멀리 떨어진 점들을 제거
    """
    if len(x_coords) < 3:
        return x_coords, y_coords

    # 좌표를 정규화
    scaler = StandardScaler()
    coords_scaled = scaler.fit_transform(np.column_stack([x_coords, y_coords]))

    # DBSCAN으로 클러스터링 (더 관대한 파라미터)
    dbscan = DBSCAN(eps=0.8, min_samples=min_cluster_size)
    labels = dbscan.fit_predict(coords_scaled)

    # 가장 큰 클러스터 찾기
    unique_labels, counts = np.unique(labels[labels != -1], return_counts=True)
    if len(unique_labels) > 0:
        main_cluster = unique_labels[np.argmax(counts)]
        mask = labels == main_cluster
    else:
        # 클러스터가 없으면 모든 점 유지
        mask = np.ones(len(x_coords), dtype=bool)

    return x_coords[mask], y_coords[mask]


def remove_spikes_hybrid(x_coords, y_coords, max_distance=30, min_cluster_size=3):
    """
    하이브리드 스파이크 제거 - 거리 기반 + 클러스터링 조합
    """
    if len(x_coords) < 3:
        return x_coords, y_coords

    # 1단계: 거리 기반으로 명백한 스파이크 제거
    x_filtered, y_filtered = remove_spikes_distance_based(
        x_coords, y_coords, max_distance)

    # 2단계: 클러스터링으로 남은 이상치 제거 (더 관대한 파라미터)
    x_final, y_final = remove_spikes_cluster_based(
        x_filtered, y_filtered, min_cluster_size)

    return x_final, y_final


def remove_spikes_local_outlier(x_coords, y_coords, n_neighbors=10, contamination=0.1):
    """
    지역적 이상치 감지 - 주변 점들과 비교하여 이상치 제거
    """
    if len(x_coords) < n_neighbors + 1:
        return x_coords, y_coords

    # 좌표를 정규화
    scaler = StandardScaler()
    coords_scaled = scaler.fit_transform(np.column_stack([x_coords, y_coords]))

    # LOF로 지역적 이상치 감지
    lof = LocalOutlierFactor(n_neighbors=n_neighbors,
                             contamination=contamination)
    labels = lof.fit_predict(coords_scaled)

    # 이상치가 아닌 점들만 선택
    mask = labels == 1

    return x_coords[mask], y_coords[mask]


def remove_spikes_adaptive(x_coords, y_coords, max_distance=50, contamination=0.05):
    """
    적응형 스파이크 제거 - 데이터 특성에 따라 필터링 강도 조절
    """
    if len(x_coords) < 3:
        return x_coords, y_coords

    # 1단계: 데이터 품질 평가
    # 연속된 점들 간의 거리 계산
    sorted_indices = np.argsort(x_coords)
    x_sorted = x_coords[sorted_indices]
    y_sorted = y_coords[sorted_indices]

    distances = np.sqrt(np.diff(x_sorted)**2 + np.diff(y_sorted)**2)

    # 스파이크 비율 계산
    spike_ratio = np.sum(distances > max_distance) / len(distances)

    print(f"  스파이크 비율: {spike_ratio:.2%}")

    # 2단계: 스파이크 비율에 따른 적응형 필터링
    if spike_ratio < 0.1:  # 스파이크가 적음 (002 케이스)
        print("  → 스파이크가 적음: 관대한 필터링 적용")
        # 매우 관대한 필터링
        x_filtered, y_filtered = remove_spikes_distance_based(
            x_coords, y_coords, max_distance * 2)

    elif spike_ratio < 0.3:  # 스파이크가 보통 (019 케이스)
        print("  → 스파이크가 보통: 표준 필터링 적용")
        # 표준 필터링
        x_filtered, y_filtered = remove_spikes_distance_based(
            x_coords, y_coords, max_distance)

    else:  # 스파이크가 많음
        print("  → 스파이크가 많음: 엄격한 필터링 적용")
        # 엄격한 필터링
        x_filtered, y_filtered = remove_spikes_distance_based(
            x_coords, y_coords, max_distance * 0.7)

    # 3단계: 결과 검증 및 후처리
    removal_ratio = (len(x_coords) - len(x_filtered)) / len(x_coords)

    if removal_ratio > 0.5:  # 너무 많이 제거된 경우
        print(f"  → 과도한 제거 ({removal_ratio:.1%}): 원본 반환")
        return x_coords, y_coords
    else:
        print(f"  → 적절한 제거 ({removal_ratio:.1%}): 필터링된 결과 반환")
        return x_filtered, y_filtered


def smooth_trajectory(x_coords, y_coords, window_size=5):
    """
    궤적을 스무딩하여 자연스러운 곡선으로 만듦
    """
    if len(x_coords) < window_size:
        return x_coords, y_coords

    # x 좌표로 정렬
    sorted_indices = np.argsort(x_coords)
    x_sorted = x_coords[sorted_indices]
    y_sorted = y_coords[sorted_indices]

    # 이동 평균으로 스무딩
    x_smoothed = np.convolve(x_sorted, np.ones(
        window_size)/window_size, mode='valid')
    y_smoothed = np.convolve(y_sorted, np.ones(
        window_size)/window_size, mode='valid')

    # 경계 처리 (처음과 끝 점들 추가)
    x_final = np.concatenate(
        [x_sorted[:window_size//2], x_smoothed, x_sorted[-(window_size//2):]])
    y_final = np.concatenate(
        [y_sorted[:window_size//2], y_smoothed, y_sorted[-(window_size//2):]])

    return x_final, y_final


def remove_spikes_velocity_based(x_coords, y_coords, max_velocity=100):
    """
    속도 기반 스파이크 제거 - 연속된 점들 간의 속도가 너무 큰 점들을 제거
    """
    if len(x_coords) < 3:
        return x_coords, y_coords

    # x 좌표로 정렬
    sorted_indices = np.argsort(x_coords)
    x_sorted = x_coords[sorted_indices]
    y_sorted = y_coords[sorted_indices]

    # 연속된 점들 간의 속도 계산 (픽셀/프레임 단위로 가정)
    dx = np.diff(x_sorted)
    dy = np.diff(y_sorted)
    velocities = np.sqrt(dx**2 + dy**2)

    # 속도가 임계값을 초과하는 점들의 인덱스 찾기
    spike_indices = np.where(velocities > max_velocity)[0]

    # 스파이크 인덱스를 실제 좌표 인덱스로 변환 (다음 점이 스파이크)
    spike_indices = spike_indices + 1

    # 스파이크가 아닌 점들만 선택
    mask = np.ones(len(x_sorted), dtype=bool)
    mask[spike_indices] = False

    return x_sorted[mask], y_sorted[mask]


def remove_spikes_adaptive_iqr(x_coords, y_coords, multiplier=1.0):
    """
    적응형 IQR - 각 축별로 다른 임계값 적용
    """
    if len(x_coords) < 3:
        return x_coords, y_coords

    # X, Y 각각에 대해 IQR 계산
    x_q1, x_q3 = np.percentile(x_coords, [25, 75])
    y_q1, y_q3 = np.percentile(y_coords, [25, 75])

    x_iqr = x_q3 - x_q1
    y_iqr = y_q3 - y_q1

    # 데이터의 분산을 고려한 적응형 임계값
    x_std = np.std(x_coords)
    y_std = np.std(y_coords)

    # 분산이 클수록 더 관대한 임계값 적용
    x_multiplier = multiplier * (1 + x_std / 100)
    y_multiplier = multiplier * (1 + y_std / 100)

    # 이상치 범위 설정
    x_lower = x_q1 - x_multiplier * x_iqr
    x_upper = x_q3 + x_multiplier * x_iqr
    y_lower = y_q1 - y_multiplier * y_iqr
    y_upper = y_q3 + y_multiplier * y_iqr

    # 이상치가 아닌 점들만 선택
    mask = ((x_coords >= x_lower) & (x_coords <= x_upper) &
            (y_coords >= y_lower) & (y_coords <= y_upper))

    return x_coords[mask], y_coords[mask]


def visualize_trajectory_comparison(x_orig, y_orig, x_filtered, y_filtered,
                                    frame_w, frame_h, method_name, save_path):
    """
    원본과 필터링된 궤적을 비교하여 시각화
    """
    fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(18, 6))

    # 원본 궤적
    ax1.scatter(x_orig, y_orig, c='red', alpha=0.6,
                s=20, label='Original points')
    ax1.plot(x_orig, y_orig, 'r-', alpha=0.3, linewidth=1)
    ax1.set_title(f'Original Trajectory\n({len(x_orig)} points)', fontsize=12)
    ax1.set_xlabel('X coordinate')
    ax1.set_ylabel('Y coordinate')
    ax1.grid(True, alpha=0.3)
    # 궤적 데이터의 실제 범위 계산
    all_x = np.concatenate([x_orig, x_filtered])
    all_y = np.concatenate([y_orig, y_filtered])

    # 데이터 범위에 여백 추가 (10%)
    x_margin = (all_x.max() - all_x.min()) * 0.1
    y_margin = (all_y.max() - all_y.min()) * 0.1

    x_min = max(0, all_x.min() - x_margin)
    x_max = min(frame_w, all_x.max() + x_margin)
    y_min = max(0, all_y.min() - y_margin)
    y_max = min(frame_h, all_y.max() + y_margin)

    ax1.set_xlim(x_min, x_max)
    ax1.set_ylim(y_min, y_max)
    ax1.invert_yaxis()  # 이미지 좌표계에 맞게 y축 반전

    # 필터링된 궤적
    ax2.scatter(x_filtered, y_filtered, c='blue',
                alpha=0.6, s=20, label='Filtered points')
    ax2.plot(x_filtered, y_filtered, 'b-', alpha=0.7, linewidth=2)
    ax2.set_title(
        f'Filtered Trajectory ({method_name})\n({len(x_filtered)} points)', fontsize=12)
    ax2.set_xlabel('X coordinate')
    ax2.set_ylabel('Y coordinate')
    ax2.grid(True, alpha=0.3)
    ax2.set_xlim(x_min, x_max)
    ax2.set_ylim(y_min, y_max)
    ax2.invert_yaxis()

    # 비교 (겹쳐서 표시)
    ax3.scatter(x_orig, y_orig, c='red', alpha=0.4, s=15, label='Original')
    ax3.scatter(x_filtered, y_filtered, c='blue',
                alpha=0.8, s=20, label='Filtered')
    ax3.plot(x_orig, y_orig, 'r-', alpha=0.3, linewidth=1)
    ax3.plot(x_filtered, y_filtered, 'b-', alpha=0.7, linewidth=2)
    ax3.set_title(
        f'Comparison\nOriginal: {len(x_orig)} → Filtered: {len(x_filtered)}', fontsize=12)
    ax3.set_xlabel('X coordinate')
    ax3.set_ylabel('Y coordinate')
    ax3.grid(True, alpha=0.3)
    ax3.legend()
    ax3.set_xlim(x_min, x_max)
    ax3.set_ylim(y_min, y_max)
    ax3.invert_yaxis()

    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()

    print(f"Comparison saved to: {save_path}")


def process_single_csv(csv_path: Path, output_dir: Path, methods: dict):
    """
    단일 CSV 파일을 처리하여 여러 방법으로 스파이크 제거 후 비교
    """
    print(f"\nProcessing: {csv_path.name}")

    # 궤적 데이터 로드
    x_orig, y_orig, frame_w, frame_h = load_trajectory_from_csv(csv_path)

    if len(x_orig) < 3:
        print(f"Not enough data points in {csv_path.name}")
        return

    # 각 방법별로 스파이크 제거 및 시각화
    for method_name, method_func in methods.items():
        try:
            x_filtered, y_filtered = method_func(x_orig, y_orig)

            # 결과 저장
            output_filename = f"{csv_path.stem}_{method_name}_comparison.png"
            output_path = output_dir / output_filename

            visualize_trajectory_comparison(
                x_orig, y_orig, x_filtered, y_filtered,
                frame_w, frame_h, method_name, output_path
            )

            # 필터링된 데이터를 CSV로 저장
            filtered_df = pd.DataFrame({
                'blue_dot_x': x_filtered,
                'blue_dot_y': y_filtered,
                'frame_w': frame_w,
                'frame_h': frame_h
            })

            filtered_csv_path = output_dir / \
                f"{csv_path.stem}_{method_name}_filtered.csv"
            filtered_df.to_csv(filtered_csv_path,
                               index=False, encoding='utf-8-sig')

        except Exception as e:
            print(f"Error processing {method_name} for {csv_path.name}: {e}")


def create_summary_comparison(csv_paths: list, output_dir: Path, methods: dict):
    """
    모든 CSV 파일의 결과를 요약하여 비교
    """
    print("\nCreating summary comparison...")

    # 각 방법별로 모든 파일의 결과를 수집
    method_results = {method: {'points_removed': [], 'files': []}
                      for method in methods.keys()}

    for csv_path in csv_paths:
        x_orig, y_orig, frame_w, frame_h = load_trajectory_from_csv(csv_path)

        if len(x_orig) < 3:
            continue

        for method_name, method_func in methods.items():
            try:
                x_filtered, y_filtered = method_func(x_orig, y_orig)
                points_removed = len(x_orig) - len(x_filtered)

                method_results[method_name]['points_removed'].append(
                    points_removed)
                method_results[method_name]['files'].append(csv_path.stem)

            except Exception as e:
                print(f"Error in summary for {method_name}: {e}")

    # 요약 시각화
    fig, axes = plt.subplots(2, 2, figsize=(15, 12))

    # 1. 각 방법별 제거된 점의 수
    ax1 = axes[0, 0]
    method_names = list(methods.keys())
    avg_points_removed = [np.mean(method_results[method]['points_removed'])
                          for method in method_names]

    bars = ax1.bar(method_names, avg_points_removed,
                   color='skyblue', alpha=0.7)
    ax1.set_title('Average Points Removed by Method')
    ax1.set_ylabel('Average Points Removed')
    ax1.tick_params(axis='x', rotation=45)

    # 막대 위에 값 표시
    for bar, value in zip(bars, avg_points_removed):
        ax1.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.1,
                 f'{value:.1f}', ha='center', va='bottom')

    # 2. 각 방법별 제거율 분포
    ax2 = axes[0, 1]
    removal_rates = []
    for method in method_names:
        if method_results[method]['points_removed']:
            rates = []
            for i, file_stem in enumerate(method_results[method]['files']):
                # 원본 파일에서 총 점 수 계산
                original_file = None
                for csv_path in csv_paths:
                    if csv_path.stem == file_stem:
                        original_file = csv_path
                        break

                if original_file:
                    x_orig, _, _, _ = load_trajectory_from_csv(original_file)
                    total_points = len(x_orig)
                    if total_points > 0:
                        removed = method_results[method]['points_removed'][i]
                        rates.append(removed / total_points)
            removal_rates.append(rates)
        else:
            removal_rates.append([])

    ax2.boxplot([rates for rates in removal_rates if rates],
                labels=[name for name, rates in zip(method_names, removal_rates) if rates])
    ax2.set_title('Removal Rate Distribution by Method')
    ax2.set_ylabel('Removal Rate')
    ax2.tick_params(axis='x', rotation=45)

    # 3. 파일별 비교 (처음 10개 파일만)
    ax3 = axes[1, 0]
    files_to_show = method_results[method_names[0]]['files'][:10]
    x_pos = np.arange(len(files_to_show))

    for i, method in enumerate(method_names):
        method_points = []
        for file in files_to_show:
            if file in method_results[method]['files']:
                idx = method_results[method]['files'].index(file)
                method_points.append(
                    method_results[method]['points_removed'][idx])
            else:
                method_points.append(0)

        ax3.plot(x_pos, method_points, marker='o', label=method, alpha=0.7)

    ax3.set_title('Points Removed by File (First 10 files)')
    ax3.set_xlabel('File Index')
    ax3.set_ylabel('Points Removed')
    ax3.legend()
    ax3.grid(True, alpha=0.3)

    # 4. 방법별 효과성 비교
    ax4 = axes[1, 1]
    effectiveness = []
    for method in method_names:
        if method_results[method]['points_removed']:
            # 제거된 점의 수가 적당하고 일관성이 있는 방법이 효과적
            avg_removed = np.mean(method_results[method]['points_removed'])
            std_removed = np.std(method_results[method]['points_removed'])
            effectiveness_score = avg_removed / \
                (std_removed + 1)  # 표준편차가 작을수록 좋음
            effectiveness.append(effectiveness_score)
        else:
            effectiveness.append(0)

    bars = ax4.bar(method_names, effectiveness, color='lightcoral', alpha=0.7)
    ax4.set_title('Method Effectiveness Score\n(Higher = Better)')
    ax4.set_ylabel('Effectiveness Score')
    ax4.tick_params(axis='x', rotation=45)

    for bar, value in zip(bars, effectiveness):
        ax4.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01,
                 f'{value:.2f}', ha='center', va='bottom')

    plt.tight_layout()
    summary_path = output_dir / "spike_removal_summary.png"
    plt.savefig(summary_path, dpi=300, bbox_inches='tight')
    plt.close()

    print(f"Summary comparison saved to: {summary_path}")


def parse_args():
    parser = argparse.ArgumentParser(
        description='Remove spikes from individual trajectory CSV files and compare results')
    parser.add_argument('--root', type=str, default='ultralytics/runs/detect',
                        help='Root directory containing CSV files')
    parser.add_argument('--pattern', type=str, default='*_trajectories.csv',
                        help='CSV file pattern to match')
    parser.add_argument('--output', type=str, default='spike_removal_results',
                        help='Output directory for results')
    parser.add_argument('--methods', nargs='+',
                        default=['iqr', 'dbscan', 'lof', 'isolation_forest',
                                 'elliptic_envelope', 'savgol', 'moving_median',
                                 'distance_based', 'velocity_based', 'adaptive_iqr',
                                 'cluster_based', 'hybrid', 'local_outlier', 'adaptive'],
                        help='Spike removal methods to use')
    parser.add_argument('--contamination', type=float, default=0.1,
                        help='Contamination parameter for outlier detection methods')
    parser.add_argument('--iqr_multiplier', type=float, default=1.5,
                        help='IQR multiplier for outlier detection')
    parser.add_argument('--max_distance', type=float, default=50,
                        help='Maximum distance between consecutive points')
    parser.add_argument('--max_velocity', type=float, default=100,
                        help='Maximum velocity between consecutive points')

    return parser.parse_args()


def main():
    args = parse_args()

    # 출력 디렉토리 생성
    output_dir = Path(args.output)
    output_dir.mkdir(exist_ok=True)

    # CSV 파일 찾기
    root_dir = Path(args.root)
    csv_paths = list(root_dir.rglob(args.pattern))

    if not csv_paths:
        print(
            f"No CSV files found matching pattern '{args.pattern}' in {root_dir}")
        return

    print(f"Found {len(csv_paths)} CSV files")

    # 스파이크 제거 방법 정의
    methods = {}

    if 'iqr' in args.methods:
        methods['iqr'] = lambda x, y: remove_spikes_iqr(
            x, y, args.iqr_multiplier)
    if 'dbscan' in args.methods:
        methods['dbscan'] = lambda x, y: remove_spikes_dbscan(x, y)
    if 'lof' in args.methods:
        methods['lof'] = lambda x, y: remove_spikes_lof(
            x, y, args.contamination)
    if 'isolation_forest' in args.methods:
        methods['isolation_forest'] = lambda x, y: remove_spikes_isolation_forest(
            x, y, args.contamination)
    if 'elliptic_envelope' in args.methods:
        methods['elliptic_envelope'] = lambda x, y: remove_spikes_elliptic_envelope(
            x, y, args.contamination)
    if 'savgol' in args.methods:
        methods['savgol'] = lambda x, y: remove_spikes_savgol_filter(x, y)
    if 'moving_median' in args.methods:
        methods['moving_median'] = lambda x, y: remove_spikes_moving_median(
            x, y)
    if 'distance_based' in args.methods:
        methods['distance_based'] = lambda x, y: smooth_trajectory(*remove_spikes_distance_based(
            x, y, args.max_distance))
    if 'velocity_based' in args.methods:
        methods['velocity_based'] = lambda x, y: remove_spikes_velocity_based(
            x, y, args.max_velocity)
    if 'adaptive_iqr' in args.methods:
        methods['adaptive_iqr'] = lambda x, y: remove_spikes_adaptive_iqr(
            x, y, args.iqr_multiplier)
    if 'cluster_based' in args.methods:
        methods['cluster_based'] = lambda x, y: remove_spikes_cluster_based(
            x, y)
    if 'hybrid' in args.methods:
        methods['hybrid'] = lambda x, y: remove_spikes_hybrid(
            x, y, args.max_distance)
    if 'local_outlier' in args.methods:
        methods['local_outlier'] = lambda x, y: remove_spikes_local_outlier(
            x, y, contamination=args.contamination)
    if 'adaptive' in args.methods:
        methods['adaptive'] = lambda x, y: remove_spikes_adaptive(
            x, y, args.max_distance)

    print(f"Using methods: {list(methods.keys())}")

    # 각 CSV 파일 처리
    for csv_path in csv_paths:
        process_single_csv(csv_path, output_dir, methods)

    # 요약 비교 생성
    create_summary_comparison(csv_paths, output_dir, methods)

    print(f"\nAll results saved to: {output_dir}")


if __name__ == '__main__':
    main()

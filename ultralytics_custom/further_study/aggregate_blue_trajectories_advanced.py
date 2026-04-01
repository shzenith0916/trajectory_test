from sklearn.covariance import EllipticEnvelope
from sklearn.ensemble import IsolationForest
from sklearn.neighbors import LocalOutlierFactor
from sklearn.cluster import DBSCAN
import sklearn
import pandas as pd
import numpy as np
import argparse
import sys
import os
from pathlib import Path
import glob
import re

# ultralytics 모듈을 찾기 위해 상위 디렉토리를 sys.path에 추가
parent_dir = Path(__file__).parent.parent
sys.path.insert(0, str(parent_dir))

# increment_path 함수를 직접 구현


def increment_path(path, exist_ok=False, sep='', mkdir=False):
    """
    Increment file or directory path, i.e. runs/exp --> runs/exp{sep}2, runs/exp{sep}3, ... etc.
    """
    path = Path(path)  # os-agnostic
    if path.exists() and not exist_ok:
        suffix = path.suffix
        path = path.with_suffix('')
        dirs = glob.glob(f"{path}{sep}*")  # similar paths
        matches = [re.search(rf"%s{sep}(\d+)" % path.stem, d) for d in dirs]
        i = [int(m.groups()[0]) for m in matches if m]  # indices
        n = max(i) + 1 if i else 2  # increment number
        path = Path(f"{path}{sep}{n}{suffix}")  # update path
    dir = path if path.suffix == '' else path.parent  # directory
    if not dir.exists() and mkdir:
        dir.mkdir(parents=True, exist_ok=True)  # make directory
    return path


try:
    import cv2
    CV2_AVAILABLE = True
except ImportError:
    CV2_AVAILABLE = False
    print("Warning: cv2 not available. Some features may be limited.")


def blue_pts_from_csv(root_dir: Path, pattern: str = "*_trajectories.csv") -> tuple[np.ndarray, np.ndarray, int, int]:
    """
    Read CSV files and extract blue dot x,y point coordinates and max frame dimensions.

    Returns
        (x, y, max_width, max_height): tuple
            - x, y: concatenated coordinates with NaNs removed
            - max_width, max_height: maximum frame dimensions found
    """

    # x, y 좌표 리스트
    x_list: list[np.ndarray] = []
    y_list: list[np.ndarray] = []

    # 프레임 크기 추적
    max_width = 0
    max_height = 0

    # CSV 파일 경로 # rglob로 recursively 탐색
    csv_paths = list(root_dir.rglob(pattern))

    for csv_path in csv_paths:
        try:
            df = pd.read_csv(csv_path)
        except Exception:
            print(f"Error on reading {csv_path}")
            continue

        xcol = "blue_dot_x"
        ycol = "blue_dot_y"

        if xcol in df and ycol in df:
            x = df[xcol].dropna().to_numpy()
            y = df[ycol].dropna().to_numpy()
            if len(x) and len(y):
                x_list.append(x)
                y_list.append(y)

                # 프레임 크기 정보가 있으면 최대값 업데이트
                if "frame_w" in df.columns and "frame_h" in df.columns:
                    try:
                        frame_w = int(df["frame_w"].iloc[0])
                        frame_h = int(df["frame_h"].iloc[0])
                        max_width = max(max_width, frame_w)
                        max_height = max(max_height, frame_h)
                    except Exception:
                        pass

    # 좌표가 없으면 빈 배열 반환
    if not x_list:
        return np.array([]), np.array([]), 0, 0

    x_all = np.concatenate(x_list)
    y_all = np.concatenate(y_list)

    return x_all, y_all, max_width, max_height


def remove_outliers_iqr(x_data, y_data, multiplier=1.0):
    """
    IQR 기반 아웃라이어 제거
    """
    q1_x, q3_x = np.percentile(x_data, [25, 75])
    q1_y, q3_y = np.percentile(y_data, [25, 75])
    iqr_x = q3_x - q1_x
    iqr_y = q3_y - q1_y

    valid_mask = (
        (x_data >= q1_x - multiplier * iqr_x) & (x_data <= q3_x + multiplier * iqr_x) &
        (y_data >= q1_y - multiplier * iqr_y) & (y_data <= q3_y + multiplier * iqr_y)
    )
    return x_data[valid_mask], y_data[valid_mask]


def remove_outliers_dbscan(x_data, y_data, eps=0.1, min_samples=5):
    """
    DBSCAN 클러스터링을 사용한 아웃라이어 제거
    가장 큰 클러스터만 유지하고 나머지는 제거
    """
    try:
        from sklearn.cluster import DBSCAN

        # 좌표를 2D 배열로 결합
        points = np.column_stack([x_data, y_data])

        # DBSCAN 클러스터링
        clustering = DBSCAN(eps=eps, min_samples=min_samples).fit(points)
        labels = clustering.labels_

        # 가장 큰 클러스터 찾기 (노이즈 제외)
        unique_labels = set(labels)
        if -1 in unique_labels:
            unique_labels.remove(-1)  # 노이즈 제거

        if len(unique_labels) == 0:
            return x_data, y_data

        largest_cluster = max(unique_labels, key=lambda x: np.sum(labels == x))

        # 가장 큰 클러스터에 속하는 점들만 유지
        mask = labels == largest_cluster
        return x_data[mask], y_data[mask]

    except ImportError:
        print("sklearn not available, using IQR method")
        return x_data, y_data


def remove_outliers_lof(x_data, y_data, n_neighbors=20, contamination=0.1):
    """
    Local Outlier Factor를 사용한 아웃라이어 제거
    지역적 밀도를 고려하여 이상치 탐지
    """
    try:
        from sklearn.neighbors import LocalOutlierFactor

        # 좌표를 2D 배열로 결합
        points = np.column_stack([x_data, y_data])

        # LOF 모델 학습
        lof = LocalOutlierFactor(
            n_neighbors=n_neighbors, contamination=contamination)
        outlier_labels = lof.fit_predict(points)

        # 정상 점들만 유지 (1: 정상, -1: 이상치)
        mask = outlier_labels == 1
        return x_data[mask], y_data[mask]

    except ImportError:
        print("sklearn not available, using IQR method")
        return x_data, y_data


def remove_outliers_isolation_forest(x_data, y_data, contamination=0.1):
    """
    Isolation Forest를 사용한 아웃라이어 제거
    트리 기반 이상치 탐지
    """
    try:
        from sklearn.ensemble import IsolationForest

        # 좌표를 2D 배열로 결합
        points = np.column_stack([x_data, y_data])

        # Isolation Forest 모델 학습
        iso_forest = IsolationForest(
            contamination=contamination, random_state=42)
        outlier_labels = iso_forest.fit_predict(points)

        # 정상 점들만 유지 (1: 정상, -1: 이상치)
        mask = outlier_labels == 1
        return x_data[mask], y_data[mask]

    except ImportError:
        print("sklearn not available, using IQR method")
        return x_data, y_data


def remove_outliers_elliptic_envelope(x_data, y_data, contamination=0.1):
    """
    Elliptic Envelope를 사용한 아웃라이어 제거
    가우시안 분포를 가정한 이상치 탐지
    """
    try:
        from sklearn.covariance import EllipticEnvelope

        # 좌표를 2D 배열로 결합
        points = np.column_stack([x_data, y_data])

        # Elliptic Envelope 모델 학습
        ee = EllipticEnvelope(contamination=contamination, random_state=42)
        outlier_labels = ee.fit_predict(points)

        # 정상 점들만 유지 (1: 정상, -1: 이상치)
        mask = outlier_labels == 1
        return x_data[mask], y_data[mask]

    except ImportError:
        print("sklearn not available, using IQR method")
        return x_data, y_data


def build_trajectory_from_points(x_coor: np.ndarray, y_coor: np.ndarray, grid_height: int, grid_width: int, gaussian_kernel: int = 9) -> tuple[np.ndarray, np.ndarray]:
    """
    실제 점들을 연결하여 궤적을 만드는 방법
    """
    assert grid_height > 0 and grid_width > 0, "Grid size must be positive"
    assert gaussian_kernel % 2 == 1, "Gaussian kernel size must be odd"

    # 원시 좌표에 대해 직접 min-max 정규화 (선형 binning)
    x_data = x_coor.astype(np.float32)
    y_data = y_coor.astype(np.float32)

    x_min, x_max = float(x_data.min()), float(x_data.max())
    y_min, y_max = float(y_data.min()), float(y_data.max())

    # 분모 계산: 최대값과 최소값의 차이 (범위)
    den_x = x_max - x_min
    den_y = y_max - y_min

    # X 좌표 정규화 (0~1 범위로 변환)
    if den_x <= 0:  # 모든 x 값이 같으면 (분모가 0)
        x_norm = np.zeros_like(x_data, dtype=np.float32)  # 0으로 채움
    else:
        x_norm = (x_data - x_min) / den_x  # min-max 정규화: (값 - 최소값) / 범위

    # Y 좌표 정규화 (0~1 범위로 변환)
    if den_y <= 0:  # 모든 y 값이 같으면 (분모가 0)
        y_norm = np.zeros_like(y_data, dtype=np.float32)  # 0으로 채움
    else:
        y_norm = (y_data - y_min) / den_y  # min-max 정규화: (값 - 최소값) / 범위

    # 0~1 → grid 인덱스 매핑
    xi = np.clip((x_norm * (grid_width - 1)).astype(np.int32),
                 0, grid_width - 1)
    yi = np.clip((y_norm * (grid_height - 1)).astype(np.int32),
                 0, grid_height - 1)

    # 밀도 맵 초기화
    density = np.zeros((grid_height, grid_width), dtype=np.float32)
    np.add.at(density, (yi, xi), 1.0)

    # 가우시안 블러
    k = int(gaussian_kernel) if gaussian_kernel and gaussian_kernel > 0 else 9
    if k % 2 == 0:
        k += 1
    if CV2_AVAILABLE:
        density_blur = cv2.GaussianBlur(density, (k, k), 0)
    else:
        # cv2가 없을 때는 scipy.ndimage 사용
        from scipy.ndimage import gaussian_filter
        density_blur = gaussian_filter(density, sigma=k/3)

    # 실제 점들을 시간 순서대로 연결하여 궤적 생성
    # x 좌표로 정렬하여 시간 순서 가정
    sorted_indices = np.argsort(xi)
    trajectory_points = np.column_stack(
        [xi[sorted_indices], yi[sorted_indices]])

    return density_blur, trajectory_points


def render_heatmap_with_ridge(
    density_blur: np.ndarray,
    ridge_points: np.ndarray,
    out_path: Path,
    method_name: str = ""
) -> None:
    if CV2_AVAILABLE:
        heat = cv2.normalize(density_blur, None, 0, 255,
                             cv2.NORM_MINMAX).astype(np.uint8)
        heat_bgr = cv2.applyColorMap(heat, cv2.COLORMAP_JET)
    else:
        # cv2가 없을 때는 matplotlib으로 대체
        import matplotlib.pyplot as plt
        import matplotlib.cm as cm

        # 정규화
        heat = ((density_blur - density_blur.min()) /
                (density_blur.max() - density_blur.min() + 1e-9) * 255).astype(np.uint8)

        # 컬러맵 적용
        cmap = cm.get_cmap('jet')
        heat_rgba = cmap(heat / 255.0)
        heat_bgr = (heat_rgba[:, :, :3] *
                    255).astype(np.uint8)[:, :, ::-1]  # RGB to BGR

    # 최빈 경로를 부드럽고 명확하게 그리기
    if len(ridge_points) > 1:
        # 스무딩된 경로 생성 (이동평균)
        window_size = min(5, len(ridge_points) // 10)
        if window_size > 1:
            smoothed_y = []
            for i in range(len(ridge_points)):
                start = max(0, i - window_size // 2)
                end = min(len(ridge_points), i + window_size // 2 + 1)
                avg_y = np.mean(ridge_points[start:end, 1])
                smoothed_y.append(avg_y)
            smoothed_points = np.column_stack([ridge_points[:, 0], smoothed_y])
        else:
            smoothed_points = ridge_points

        if CV2_AVAILABLE:
            # 모달 경로 그리기 비활성화 - 히트맵만 표시
            # for i in range(len(smoothed_points) - 1):
            #     x1, y1 = smoothed_points[i]
            #     x2, y2 = smoothed_points[i + 1]
            #     cv2.line(heat_bgr, (int(x1), int(y1)),
            #              (int(x2), int(y2)), (255, 255, 255), 3)

            # for i in range(0, len(smoothed_points), max(1, len(smoothed_points) // 20)):
            #     x, y = smoothed_points[i]
            #     cv2.circle(heat_bgr, (int(x), int(y)), 4, (255, 255, 255), -1)

            cv2.imwrite(str(out_path), heat_bgr)
        else:
            # matplotlib으로 저장
            import matplotlib.pyplot as plt
            import matplotlib.cm as cm

            fig, ax = plt.subplots(figsize=(12, 8))

            # 히트맵 표시
            im = ax.imshow(density_blur, cmap='jet', aspect='auto')

            # 모달 경로 그리기 비활성화 - 히트맵만 표시
            # if len(smoothed_points) > 1:
            #     ax.plot(smoothed_points[:, 0], smoothed_points[:, 1],
            #             'w-', linewidth=3, alpha=0.8)
            #     ax.scatter(smoothed_points[::max(1, len(smoothed_points) // 20), 0],
            #                smoothed_points[::max(
            #                    1, len(smoothed_points) // 20), 1],
            #                c='white', s=50, alpha=0.9)

            title = f'Trajectory Heatmap ({method_name})' if method_name else 'Trajectory Heatmap'
            ax.set_title(title)
            ax.set_xlabel('X coordinate')
            ax.set_ylabel('Y coordinate')
            plt.colorbar(im, ax=ax)
            plt.tight_layout()
            plt.savefig(str(out_path), dpi=150, bbox_inches='tight')
            plt.close()


def save_ridge_csv(ridge_points: np.ndarray, out_csv: Path) -> None:
    grid_x = ridge_points[:, 0].astype(np.float32)
    grid_y = ridge_points[:, 1].astype(np.float32)
    # Normalized [0,1] for convenience
    norm_x = grid_x / max(1.0, float(grid_x.max()))
    norm_y = grid_y / max(1.0, float(grid_y.max()))
    df = pd.DataFrame({"grid_x": grid_x, "grid_y": grid_y,
                      "norm_x": norm_x, "norm_y": norm_y})
    df.to_csv(out_csv, index=False, encoding="utf-8-sig")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Advanced blue_dot trajectory aggregation with multiple outlier removal methods")
    parser.add_argument("--root", type=str, default=".",
                        help="Root directory to search recursively for *_trajectories.csv")
    parser.add_argument("--pattern", type=str, default="*_trajectories.csv",
                        help="CSV filename pattern to match")
    parser.add_argument("--width", type=int, default=1280,
                        help="Output grid width")
    parser.add_argument("--height", type=int, default=720,
                        help="Output grid height")
    parser.add_argument("--blur", type=int, default=9,
                        help="Gaussian blur kernel size (odd)")
    parser.add_argument("--outdir", type=str, default="runs/aggregate_advanced",
                        help="Output directory for results")
    parser.add_argument("--methods", type=str, nargs="+",
                        default=["iqr", "lof", "isolation_forest",
                                 "dbscan", "elliptic_envelope"],
                        help="Outlier removal methods to compare")
    parser.add_argument("--contamination", type=float, default=0.1,
                        help="Contamination ratio for outlier detection")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    root = Path(args.root).resolve()
    if not root.exists():
        raise FileNotFoundError(f"Root directory not found: {root}")

    x_all, y_all, max_width, max_height = blue_pts_from_csv(root, args.pattern)
    if x_all.size == 0:
        raise SystemExit(
            "No blue_dot points found. Ensure *_trajectories.csv files exist with blue_dot_x/y columns.")

    # 원본 영상 해상도를 우선 사용, 없으면 명령행 인수 사용
    if max_width > 0 and max_height > 0:
        grid_width = max_width
        grid_height = max_height
        print(f"Using original video resolution: {grid_width}x{grid_height}")
    else:
        grid_width = args.width
        grid_height = args.height
        print(f"Using default resolution: {grid_width}x{grid_height}")

    print(f"Total points before outlier removal: {len(x_all)}")

    # 각 방법별로 처리
    for method in args.methods:
        print(f"\n=== Processing with {method.upper()} method ===")

        # 아웃라이어 제거
        x_clean, y_clean = x_all.copy(), y_all.copy()

        if len(x_clean) > 10:
            original_count = len(x_clean)

            if method == "iqr":
                x_clean, y_clean = remove_outliers_iqr(
                    x_clean, y_clean, multiplier=1.0)

            elif method == "lof":
                x_clean, y_clean = remove_outliers_lof(
                    x_clean, y_clean, contamination=args.contamination)

            elif method == "isolation_forest":
                x_clean, y_clean = remove_outliers_isolation_forest(
                    x_clean, y_clean, contamination=args.contamination)

            elif method == "dbscan":
                x_clean, y_clean = remove_outliers_dbscan(x_clean, y_clean)

            elif method == "elliptic_envelope":
                x_clean, y_clean = remove_outliers_elliptic_envelope(
                    x_clean, y_clean, contamination=args.contamination)

            print(
                f"Outlier removal ({method}): {original_count} -> {len(x_clean)} points")

        # 밀도맵 생성 (실제 점들을 연결하는 방법 사용)
        density_blur, trajectory_points = build_trajectory_from_points(
            x_clean, y_clean, grid_height=grid_height, grid_width=grid_width, gaussian_kernel=args.blur
        )

        # 결과 저장
        out_base = increment_path(Path(args.outdir) / method, exist_ok=False)
        out_base.mkdir(parents=True, exist_ok=True)

        heatmap_path = out_base / f"blue_trajectory_density_{method}.png"
        render_heatmap_with_ridge(
            density_blur, trajectory_points, heatmap_path, method)

        trajectory_csv_path = out_base / f"blue_trajectory_path_{method}.csv"
        save_ridge_csv(trajectory_points, trajectory_csv_path)

        np.save(out_base / f"blue_density_{method}.npy", density_blur)

        print(f"Saved {method} results:")
        print(f"  - Heatmap: {heatmap_path}")
        print(f"  - Trajectory path CSV: {trajectory_csv_path}")
        print(f"  - Density array: {out_base / f'blue_density_{method}.npy'}")

    print(f"\n=== Comparison Summary ===")
    print(f"All methods completed. Check {args.outdir} directory for results.")


if __name__ == "__main__":
    main()

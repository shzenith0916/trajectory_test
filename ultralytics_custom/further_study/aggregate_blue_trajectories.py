import re
import argparse
import numpy as np
import pandas as pd
import glob
import sys
import os
from pathlib import Path

# ultralytics 모듈을 찾기 위해 상위 디렉토리를 sys.path에 추가
parent_dir = Path(__file__).parent.parent
sys.path.insert(0, str(parent_dir))


try:
    import cv2
    CV2_AVAILABLE = True
except ImportError:
    CV2_AVAILABLE = False
    print("Warning: cv2 not available. Some features may be limited.")

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


# 설정
csv_dir = "runs/detect"  # 하위 모든 폴더 탐색
out_size = (720, 1280)  # (w,h) 최종 격자 해상도
sigma = (9, 9)  # (w,h) 가우시안 필터 크기


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


def to_grid_from_norm(x_norm, W, use_log=False, k=1.0):
    """
    x ∈ [0,1], 로그 압축 옵션 (use_log=True) 사용 시, 0→0, 1→1 유지
    """
    x = np.clip(x_norm.astype(np.float32), 0.0, 1.0)
    if use_log:
        x = np.log1p(k * x) / np.log1p(k)  # 0→0, 1→1 유지
    return np.clip((x * (W - 1)).astype(np.int32), 0, W - 1)


def build_density_map(x_coor: np.ndarray, y_coor: np.ndarray, grid_height: int, grid_width: int, gaussian_kernel: int = 9, bins: int = 100, outlier_threshold: float = 2.0) -> tuple[np.ndarray, np.ndarray]:
    """
    Build a smoothed density map from scattered points using logscaling to the grid.

    Parameters
    ----------
    x_coor: np.ndarray(N) - x coordinates of scattered points
    y_coor: np.ndarray(N) - y coordinates of scattered points
    grid_height: int - height of the grid
    grid_width: int - width of the grid
    gaussian_kernel: int - size of the gaussian kernel
    bins: int - number of bins for the histogram

    Returns
    -------
    density_blur: np.ndarray(H, W)
        Smoothed density map
    ridge_points: np.ndarray(W, 2)
        Modal path per x-column as (x, y) integer grid coordinates
    """
    assert grid_height > 0 and grid_width > 0, "Grid size must be positive"
    assert gaussian_kernel % 2 == 1, "Gaussian kernel size must be odd"

    # 원시 좌표에 대해 직접 min-max 정규화 (선형 binning)
    x_data = x_coor.astype(np.float32)
    y_data = y_coor.astype(np.float32)

    # 아웃라이어 제거 (더 강력한 방법)
    if len(x_data) > 10:
        # 더 엄격한 IQR 기반 아웃라이어 제거 (1.5 -> 1.0)
        q1_x, q3_x = np.percentile(x_data, [25, 75])
        q1_y, q3_y = np.percentile(y_data, [25, 75])
        iqr_x = q3_x - q1_x
        iqr_y = q3_y - q1_y

        # 1.0 * IQR 범위 내의 점들만 유지 (더 엄격)
        valid_mask = (
            (x_data >= q1_x - 1.0 * iqr_x) & (x_data <= q3_x + 1.0 * iqr_x) &
            (y_data >= q1_y - 1.0 * iqr_y) & (y_data <= q3_y + 1.0 * iqr_y)
        )
        x_data = x_data[valid_mask]
        y_data = y_data[valid_mask]
        print(f"Outlier removal: {len(x_coor)} -> {len(x_data)} points")

    # # 음수 방지용 쉬프트 후 log1p 적용
    # x_shift = x_data - x_data.min() if x_data.min() < 0 else x_data
    # y_shift = y_data - y_data.min() if y_data.min() < 0 else y_data
    # # log1p를 쓰는 이유는, 좌표가 0을 포함하면 np.log(0) = -inf 가 되어서 처리가 어려움.
    # # log1p(x) = log(1+x) 이므로 0을 포함해도 0으로 나와 처리가 가능함.
    # x_log = np.log1p(x_shift)
    # y_log = np.log1p(y_shift)

    # min-max 정규화
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

    # 밀도 맵 초기화 ( grid_height, grid_width )
    # 모든 좌표를 0으로 초기화한 2D 배열 생성
    density = np.zeros((grid_height, grid_width), dtype=np.float32)

    # 각 좌표 위치에 점의 개수를 누적 (히스토그램 생성)
    # yi, xi는 정규화된 좌표를 격자 인덱스로 변환한 값
    np.add.at(density, (yi, xi), 1.0)

    # Smooth density - 가우시안 블러로 노이즈 제거 및 부드러운 밀도맵 생성
    # 가우시안 커널은 홀수 및 >=3 권장 (짝수면 중앙점이 없어서 부자연스러움)
    k = int(gaussian_kernel) if gaussian_kernel and gaussian_kernel > 0 else 9
    if k % 2 == 0:  # 짝수면 홀수로 변환
        k += 1

    if CV2_AVAILABLE:
        density_blur = cv2.GaussianBlur(density, (k, k), 0)
    else:
        # cv2가 없을 때는 scipy.ndimage 사용
        from scipy.ndimage import gaussian_filter
        density_blur = gaussian_filter(density, sigma=k/3)

    # Modal y per x-column - 각 x 열에서 가장 빈번한 y 위치 찾기 (최빈 경로 추출)
    col_sums = density_blur.sum(axis=0)  # 각 열의 총 밀도 합계
    ridge_y = np.zeros(grid_width, dtype=np.float32)  # 최빈 y 좌표 저장 배열
    valid_cols = col_sums > 0  # 데이터가 있는 열들만 선택

    if valid_cols.any():  # 유효한 열이 있으면
        # 각 유효한 열에서 밀도가 최대인 y 인덱스 찾기
        ridge_y[valid_cols] = density_blur[:, valid_cols].argmax(
            axis=0).astype(np.float32)

        # Interpolate for empty columns - 빈 열들에 대해 보간으로 채우기
        x_idx = np.arange(grid_width, dtype=np.float32)  # 모든 x 인덱스
        valid_x = x_idx[valid_cols]  # 유효한 x 인덱스들
        valid_y = ridge_y[valid_cols]  # 유효한 y 값들

        if (~valid_cols).any():  # 빈 열이 있으면
            # 선형 보간으로 빈 열들의 y 값 추정
            ridge_y[~valid_cols] = np.interp(
                x_idx[~valid_cols], valid_x, valid_y)

    # 최종 y 좌표를 정수로 변환하고 격자 범위 내로 제한
    ridge_y = np.clip(np.rint(ridge_y).astype(np.int32), 0, grid_height - 1)

    # 최빈 경로 점들을 (x, y) 좌표 쌍으로 구성
    # x는 0부터 grid_width-1까지, y는 각 열의 최빈값
    ridge_points = np.stack(
        [np.arange(grid_width, dtype=np.int32), ridge_y], axis=1)

    return density_blur, ridge_points


def render_heatmap_with_ridge(
    density_blur: np.ndarray,
    ridge_points: np.ndarray,
    out_path: Path,
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

    # for x, y in ridge_points:
    #     cv2.circle(heat_bgr, (int(x), int(y)), 1, (255, 255, 255), -1)

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
            #            'w-', linewidth=3, alpha=0.8)
            #     ax.scatter(smoothed_points[::max(1, len(smoothed_points) // 20), 0],
            #               smoothed_points[::max(1, len(smoothed_points) // 20), 1],
            #               c='white', s=50, alpha=0.9)

            ax.set_title('Trajectory Heatmap')
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
        description="Aggregate blue_dot trajectories and extract modal path")
    parser.add_argument("--root", type=str, default=".",
                        help="Root directory to search recursively for *_trajectories.csv")
    parser.add_argument("--pattern", type=str, default="*_trajectories.csv",
                        help="CSV filename pattern to match")
    parser.add_argument("--width", type=int, default=800,
                        help="Output grid width")
    parser.add_argument("--height", type=int, default=600,
                        help="Output grid height")
    parser.add_argument("--blur", type=int, default=9,
                        help="Gaussian blur kernel size (odd)")
    parser.add_argument("--outdir", type=str, default="runs/aggregate",
                        help="Output directory for results")
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

    density_blur, ridge_points = build_density_map(
        x_all, y_all, grid_height=grid_height, grid_width=grid_width, gaussian_kernel=args.blur
    )

    # Prepare output directory
    out_base = increment_path(Path(args.outdir) / "blue", exist_ok=False)
    out_base.mkdir(parents=True, exist_ok=True)

    # Save heatmap with ridge overlay
    heatmap_path = out_base / "blue_trajectory_density.png"
    render_heatmap_with_ridge(density_blur, ridge_points, heatmap_path)

    # Save ridge points
    ridge_csv_path = out_base / "blue_modal_path.csv"
    save_ridge_csv(ridge_points, ridge_csv_path)

    # Also save raw density (optional)
    np.save(out_base / "blue_density.npy", density_blur)

    print(f"Saved heatmap: {heatmap_path}")
    print(f"Saved modal path CSV: {ridge_csv_path}")
    print(f"Saved raw density (npy): {out_base / 'blue_density.npy'}")


if __name__ == "__main__":
    main()

"""
궤적 CSV 파일을 읽어서 trajectory plot을 생성하는 스크립트
==========================================================
detectron2_custom_detect.py에서 저장한 CSV 파일을 기반으로
원하는 프레임 범위 또는 시간 범위를 지정하여 궤적 그래프를 생성합니다.

사용법:
    # CSV 파일 전체 범위로 궤적 그리기
    python plot_trajectories.py --csv path/to/red_trajectories.csv

    # 프레임 범위 지정
    python plot_trajectories.py --csv path/to/red_trajectories.csv --frame-start 100 --frame-end 300

    # 시간 범위 지정 (초 단위, CSV에 time_sec 컬럼 필요)
    python plot_trajectories.py --csv path/to/blue_trajectories.csv --time-start 2.0 --time-end 5.0

    # 출력 파일명 지정
    python plot_trajectories.py --csv path/to/red_trajectories.csv --output my_plot.jpg

    # red + blue CSV 동시에 그리기
    python plot_trajectories.py \
        --csv path/to/red_trajectories.csv path/to/blue_trajectories.csv \
        --frame-start 50 --frame-end 200
"""

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


def detect_dot_type(df):
    """CSV 컬럼명으로 red/blue 타입 자동 감지"""
    if "red_dot_x" in df.columns:
        return "red", "red_dot_x", "red_dot_y"
    elif "blue_dot_x" in df.columns:
        return "blue", "blue_dot_x", "blue_dot_y"
    else:
        raise ValueError("CSV에 red_dot_x/red_dot_y 또는 blue_dot_x/blue_dot_y 컬럼이 없습니다.")


def filter_by_range(df, frame_start, frame_end, time_start, time_end):
    """프레임 범위 또는 시간 범위로 필터링"""
    if frame_start is not None or frame_end is not None:
        if "frame_index" not in df.columns:
            raise ValueError("CSV에 frame_index 컬럼이 없습니다. 최신 detectron2_custom_detect.py로 다시 생성하세요.")
        if frame_start is not None:
            df = df[df["frame_index"] >= frame_start]
        if frame_end is not None:
            df = df[df["frame_index"] <= frame_end]
    elif time_start is not None or time_end is not None:
        if "time_sec" not in df.columns:
            raise ValueError("CSV에 time_sec 컬럼이 없습니다. FPS 정보가 포함된 CSV가 필요합니다.")
        if time_start is not None:
            df = df[df["time_sec"] >= time_start]
        if time_end is not None:
            df = df[df["time_sec"] <= time_end]
    return df.reset_index(drop=True)


def plot_trajectory(x_coords, y_coords, title_label, output_path,
                    line_color="cadetblue", quiver_color="darksalmon"):
    """궤적 그래프를 생성하고 저장"""
    if len(x_coords) == 0:
        print(f"필터링 후 데이터가 없습니다: {title_label}")
        return

    x_coor = np.array(x_coords)
    y_coor = np.array(y_coords)

    min_x_idx = np.argmin(x_coor)
    max_x_idx = np.argmax(x_coor)
    min_y_idx = np.argmin(y_coor)
    max_y_idx = np.argmax(y_coor)

    plt.figure(figsize=(8, 6))
    plt.plot(x_coor, y_coor, marker="o",
             color="black", linestyle="-", markersize=3, linewidth=0.7)

    plt.gca().invert_yaxis()

    # A (Start) - 빨간 원
    plt.scatter(x_coor[0], y_coor[0], color="tab:red", s=150,
                marker="o", zorder=5, label="A (Start)")
    plt.annotate("A", (x_coor[0], y_coor[0]),
                 textcoords="offset points", xytext=(12, -12),
                 ha="center", fontsize=11, fontweight="bold")

    # B (Highest) - 파란 사각형 (y 최소 = 화면 최상단)
    plt.scatter(x_coor[min_y_idx], y_coor[min_y_idx], color="tab:blue", s=150,
                marker="s", zorder=5, label="B (Highest)")
    plt.annotate("B", (x_coor[min_y_idx], y_coor[min_y_idx]),
                 textcoords="offset points", xytext=(-12, -12),
                 ha="center", fontsize=11, fontweight="bold")

    # C (X-min) - 초록 삼각형
    plt.scatter(x_coor[min_x_idx], y_coor[min_x_idx], color="tab:green", s=150,
                marker="^", zorder=5, label="C (X-min)")
    plt.annotate("C", (x_coor[min_x_idx], y_coor[min_x_idx]),
                 textcoords="offset points", xytext=(-12, -5),
                 ha="center", fontsize=11, fontweight="bold")

    # D (End) - 보라 다이아몬드
    plt.scatter(x_coor[-1], y_coor[-1], color="tab:purple", s=150,
                marker="D", zorder=5, label="D (End)")
    plt.annotate("D", (x_coor[-1], y_coor[-1]),
                 textcoords="offset points", xytext=(12, 12),
                 ha="center", fontsize=11, fontweight="bold")

    plt.title(title_label)
    plt.xlabel("X Coordinate of Hyoid (px)", fontsize=10)
    plt.ylabel("Y Coordinate of Hyoid (px)", fontsize=10)
    plt.legend(loc="upper right", fontsize=9, framealpha=0.9, markerscale=0.6)
    plt.grid(True)
    plt.tight_layout()

    plt.savefig(output_path, dpi=150)
    plt.close()
    print(f"Trajectory plot saved to {output_path}")


def process_csv(csv_path, args, output_path=None):
    """CSV 파일 하나를 처리하여 궤적 그래프 생성"""
    df = pd.read_csv(csv_path, encoding="utf-8-sig")
    dot_type, x_col, y_col = detect_dot_type(df)

    df = filter_by_range(df, args.frame_start, args.frame_end,
                         args.time_start, args.time_end)

    x_coords = df[x_col].dropna().tolist()
    y_coords = df[y_col].dropna().tolist()

    if dot_type == "red":
        title = "Trajectory of Detected Hyoid (Red Dot)"
        line_color, quiver_color = "cadetblue", "darksalmon"
        default_suffix = "_red_dot_trajectory.jpg"
    else:
        title = "Trajectory of Corrected Hyoid (Blue Dot)"
        line_color, quiver_color = "royalblue", "mediumpurple"
        default_suffix = "_blue_dot_trajectory.jpg"

    # 범위 정보를 제목에 포함
    range_info = ""
    if args.frame_start is not None or args.frame_end is not None:
        fs = args.frame_start if args.frame_start is not None else 0
        fe = args.frame_end if args.frame_end is not None else "end"
        range_info = f" [frames {fs}-{fe}]"
    elif args.time_start is not None or args.time_end is not None:
        ts = f"{args.time_start:.1f}s" if args.time_start is not None else "0s"
        te = f"{args.time_end:.1f}s" if args.time_end is not None else "end"
        range_info = f" [{ts}-{te}]"

    title += range_info

    if output_path is None:
        csv_stem = Path(csv_path).stem.replace("_red_trajectories", "").replace("_blue_trajectories", "")
        output_path = Path(csv_path).parent / f"{csv_stem}{default_suffix}"

    plot_trajectory(x_coords, y_coords, title, str(output_path),
                    line_color=line_color, quiver_color=quiver_color)


def main():
    parser = argparse.ArgumentParser(
        description="CSV 기반 궤적 그래프 생성",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
사용 예시:
  # 전체 범위
  python plot_trajectories.py --csv result_red_trajectories.csv

  # 프레임 100~300만
  python plot_trajectories.py --csv result_red_trajectories.csv --frame-start 100 --frame-end 300

  # 시간 2.0초~5.0초
  python plot_trajectories.py --csv result_blue_trajectories.csv --time-start 2.0 --time-end 5.0

  # red + blue 동시 처리
  python plot_trajectories.py --csv red.csv blue.csv --frame-start 50
        """,
    )

    parser.add_argument("--csv", nargs="+", required=True,
                        help="궤적 CSV 파일 경로 (여러 개 지정 가능)")
    parser.add_argument("--frame-start", type=int, default=None,
                        help="시작 프레임 인덱스 (포함)")
    parser.add_argument("--frame-end", type=int, default=None,
                        help="끝 프레임 인덱스 (포함)")
    parser.add_argument("--time-start", type=float, default=None,
                        help="시작 시간(초)")
    parser.add_argument("--time-end", type=float, default=None,
                        help="끝 시간(초)")
    parser.add_argument("--output", default=None,
                        help="출력 파일 경로 (CSV가 1개일 때만 적용)")

    args = parser.parse_args()

    for csv_path in args.csv:
        csv_path = Path(csv_path)
        if not csv_path.exists():
            print(f"CSV 파일을 찾을 수 없습니다: {csv_path}")
            continue

        output = args.output if len(args.csv) == 1 else None
        process_csv(str(csv_path), args, output_path=output)


if __name__ == "__main__":
    main()

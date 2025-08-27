import ultralytics
import argparse
import re
import pandas as pd
import cv2
import numpy as np
import math
import matplotlib.pyplot as plt
from pathlib import Path
from ultralytics import YOLO
from ultralytics.utils.files import increment_path

# 추가 함수: crosspoint, find_min_max_points


def crosspoint(m1, c1, m2, c2):
    if m1 == m2:
        raise ValueError("The lines are parallel and do not intersect")
    x = (c2 - c1) / (m1 - m2)
    y = m1 * x + c1
    return x, y


def find_min_max_points(data_points):
    data_points = np.array(data_points)
    second_elements = data_points[:, 1]
    min_value = np.min(second_elements)
    max_value = np.max(second_elements)
    min_index = np.argmin(second_elements)
    max_index = np.argmax(second_elements)
    sublist_with_min = data_points[min_index]
    sublist_with_max = data_points[max_index]
    return sublist_with_min, sublist_with_max


def remove_korean(filename):
    """
    한글이 포함된 파일명에서 한글을 제거합니다.
    """
    return re.sub('[ㄱ-ㅎㅏ-ㅣ가-힣]+', '', filename)


def plot_trajectory(x_coords, y_coords, save_dir, base_filename, title_label, filename_suffix, line_color='cadetblue', quiver_color='darksalmon'):
    """
    주어진 좌표 리스트를 사용하여 궤적 그래프를 생성하고 저장합니다.
    """
    if not x_coords or not y_coords:
        print(f"No data to plot for {title_label}")
        return

    x_coor = np.array(x_coords)
    y_coor = np.array(y_coords)

    dx = np.diff(x_coor)
    dy = np.diff(y_coor)

    min_x_idx = np.argmin(x_coor)
    max_x_idx = np.argmax(x_coor)
    min_y_idx = np.argmin(y_coor)
    max_y_idx = np.argmax(y_coor)

    plt.figure(figsize=(8, 6))
    plt.plot(x_coor, y_coor, 'b-', marker='o',
             color=line_color, linestyle='-', markersize=3, linewidth=1)

    if len(dx) > 0:
        plt.quiver(x_coor[:-1], y_coor[:-1], dx, dy, angles='xy',
                   scale_units='xy', scale=1.5, color=quiver_color, width=0.003)

    plt.gca().invert_yaxis()

    plt.scatter(x_coor[0], y_coor[0],
                color='orange', s=100, label='Start')
    plt.annotate('Start(A)', (x_coor[0], y_coor[0]), textcoords='offset points', xytext=(
        20, 10), ha='center', fontsize=10)
    plt.scatter(x_coor[-1], y_coor[-1],
                color='seagreen', s=100, label='End')
    plt.annotate('End', (x_coor[-1], y_coor[-1]),
                 textcoords='offset points', xytext=(0, 10), ha='center', fontsize=10)
    plt.scatter(x_coor[min_x_idx], y_coor[min_x_idx],
                color='peru', s=80, label='X_Min_Point')
    plt.annotate('X_Min(C)', (x_coor[min_x_idx], y_coor[min_x_idx]), textcoords='offset points', xytext=(
        60, -30), ha='center', arrowprops=dict(arrowstyle="->", connectionstyle="arc3,rad=.8"))
    plt.scatter(x_coor[max_x_idx], y_coor[max_x_idx],
                color='gold', s=80, label='X_Max_Point')
    plt.annotate('X_Max', (x_coor[max_x_idx], y_coor[max_x_idx]), textcoords='offset points', xytext=(
        -10, 20), ha='center', arrowprops=dict(arrowstyle="->", connectionstyle="arc3,rad=.5"))
    plt.scatter(x_coor[min_y_idx], y_coor[min_y_idx],
                color='orchid', s=80, label='Highest_Point')
    plt.annotate('Highest Point(B)', (x_coor[min_y_idx], y_coor[min_y_idx]), textcoords='offset points', xytext=(
        40, -10), ha='center', arrowprops=dict(arrowstyle="->", connectionstyle="arc3,rad=.5"))
    plt.scatter(x_coor[max_y_idx], y_coor[max_y_idx],
                color='darkblue', s=80, label='Lowest_Point')
    plt.annotate('Lowest Point(D)', (x_coor[max_y_idx], y_coor[max_y_idx]), textcoords='offset points', xytext=(
        30, 30), ha='center', arrowprops=dict(arrowstyle="->", connectionstyle="arc3,rad=.8"))

    title = f"{title_label} - {base_filename}"
    plt.title(title)
    plt.xlabel("X Coordinate of Hyoid", fontsize=10)
    plt.ylabel("Y Coordinate of Hyoid", fontsize=10)
    plt.grid(True)

    trajectory_path = save_dir / f"{base_filename}{filename_suffix}"
    plt.savefig(trajectory_path)
    plt.close()
    print(f"Trajectory plot saved to {trajectory_path}")


def save_trajectories_to_csv(red_x, red_y, blue_x, blue_y, save_dir, base_filename):
    """
    주어진 궤적 좌표 리스트들을 pandas DataFrame으로 만들어 CSV 파일로 저장합니다.
    """
    # pandas Series를 사용하면 길이가 다른 리스트도 NaN으로 채워져 안전하게 DataFrame으로 만들 수 있습니다.
    df = pd.DataFrame({
        'red_dot_x': pd.Series(red_x),
        'red_dot_y': pd.Series(red_y),
        'blue_dot_x': pd.Series(blue_x),
        'blue_dot_y': pd.Series(blue_y),
    })

    csv_path = save_dir / f"{base_filename}_trajectories.csv"
    # 한글 경로/파일명 문제를 피하기 위해 encoding을 'utf-8-sig'로 지정
    df.to_csv(csv_path, index=False, encoding='utf-8-sig')
    print(f"Trajectories data saved to {csv_path}")


def run(weights, source):
    # 모델 로드
    model = YOLO(weights)

    # 예측할 소스 경로 예시 (영상 파일 경로)
    # source = 'C:/Users/USER/Documents/AKAS관련/AKAS/ultralytics/********_조*자(1)SF_trimmed.avi'

    # 출력 파일명에서 한글 제거
    source_stem = Path(source).stem
    clean_stem = remove_korean(source_stem)

    # 결과 저장을 위한 디렉토리 설정 (한글 없는 이름으로)
    save_dir = increment_path(Path('runs/detect') / clean_stem, exist_ok=False)
    save_dir.mkdir(parents=True, exist_ok=True)

    # 비디오 라이터 초기화
    cap = cv2.VideoCapture(source)
    w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = cap.get(cv2.CAP_PROP_FPS)
    output_video_path = str(save_dir / f"{clean_stem}_custom.avi")
    vid_writer = cv2.VideoWriter(
        output_video_path, cv2.VideoWriter_fourcc(*'mp4v'), fps, (w, h))
    cap.release()  # 정보만 얻고 해제

    # 모델 예측 실행 (스트림 모드로 메모리 효율성 확보)
    results = model.predict(source, stream=True)

    # 궤적을 그리기 위한 리스트 초기화
    hyoid_x_list = []
    hyoid_y_list = []
    red_dot_trajectory_points = []  # 비디오에 빨간점 궤적을 그리기 위한 리스트
    trajectory_points = []  # 비디오에 설골점 궤적을 그리기 위한 리스트
    blue_dot_x_list = []
    blue_dot_y_list = []

    # 결과 처리
    for r in results:
        # im0 = r.plot()  # plot()은 바운딩 박스와 라벨이 그려진 numpy 배열을 반환
        im0 = r.orig_img.copy()

        boxes = r.boxes.cpu().numpy()  # CPU로 이동 후 numpy로 변환

        neck_centers = []
        hyoid_bone_center = None

        # r.names 를 통해 클래스 이름을 가져올 수 있음음. 예: {0: 'hyoid bone', 1: 'neck bone'}
        # 현재는 ID 0, 1을 직접 사용
        for box in boxes:
            cls = int(box.cls[0])

            # xywh 형식의 좌표
            x_center, y_center, _, _ = box.xywh[0]
            pixel_center = (int(x_center), int(y_center))

            if cls == 0:  # hyoid bone
                hyoid_bone_center = pixel_center
                hyoid_x_list.append(pixel_center[0])
                hyoid_y_list.append(pixel_center[1])
                # 비디오에 그릴 빨간 점 궤적 좌표 추가
                red_dot_trajectory_points.append(hyoid_bone_center)

            elif cls == 1:  # neck bone
                neck_centers.append(pixel_center)

        # 선 그리기 및 거리 계산 로직 (yolov5에서 진행했던 커스텀 코드 detect_test(1).py에서 가져옴)
        if len(neck_centers) >= 2 and hyoid_bone_center:
            data_points = np.array(neck_centers, dtype=np.float32)
            sublist_with_min, sublist_with_max = find_min_max_points(
                data_points)

            [vx, vy, x0, y0] = cv2.fitLine(
                data_points, cv2.DIST_L2, 0, 0.01, 0.01)

            m1 = vy / vx if vx != 0 else float('inf')

            if m1 != float('inf'):
                c1 = y0 - m1 * x0

                if m1 != 0:
                    x_min = (sublist_with_min[1] - c1) / m1
                    x_max = (sublist_with_max[1] - c1) / m1
                    start_point = (int(x_min), int(sublist_with_min[1]))
                    end_point = (int(x_max), int(sublist_with_max[1]))
                else:  # 수평선
                    start_point = (
                        int(sublist_with_min[0]), int(sublist_with_min[1]))
                    end_point = (int(sublist_with_max[0]), int(
                        sublist_with_max[1]))

                cv2.line(im0, start_point, end_point,
                         color=(0, 255, 0), thickness=2)

                if m1 != 0:
                    m2 = -1 / m1
                    c2 = hyoid_bone_center[1] - m2 * hyoid_bone_center[0]

                    try:
                        intercept_x, intercept_y = crosspoint(m1, c1, m2, c2)
                        intercept_x_scalar = intercept_x.item()
                        intercept_y_scalar = intercept_y.item()

                        cv2.line(im0, hyoid_bone_center, (int(intercept_x_scalar), int(
                            intercept_y_scalar)), (0, 255, 0), 2)
                        cv2.circle(im0, hyoid_bone_center, 5, (0, 0, 255), -1)

                        distance = math.dist(
                            hyoid_bone_center, (intercept_x_scalar, intercept_y_scalar))

                        horizontal_start_point = (
                            int(intercept_x_scalar - distance), int(intercept_y_scalar))
                        cv2.line(im0, horizontal_start_point, (int(intercept_x_scalar), int(
                            intercept_y_scalar)), color=(255, 255, 255), thickness=2)
                        cv2.circle(im0, horizontal_start_point,
                                   5, (255, 0, 0), -1)

                        # 파란 점의 좌표를 궤적 리스트에 추가
                        trajectory_points.append(horizontal_start_point)

                        # Matplotlib 그래프용으로 파란 점 좌표 저장
                        blue_dot_x_list.append(horizontal_start_point[0])
                        blue_dot_y_list.append(horizontal_start_point[1])

                        vertical_start_point = (
                            int(intercept_x_scalar), start_point[1])
                        vertical_end_point = (
                            int(intercept_x_scalar), end_point[1])
                        cv2.line(im0, vertical_start_point, vertical_end_point,
                                 color=(255, 255, 255), thickness=2)
                    except (ValueError, ZeroDivisionError) as e:
                        print(f"Could not calculate intersection: {e}")
            else:  # 수직선
                # 수직선에 대한 로직 추가 (필요 시)
                pass

        # 현재까지의 궤적을 비디오 프레임에 그리기 (파란 점)
        if len(trajectory_points) > 1:
            pts_blue = np.array(trajectory_points, np.int32)
            pts_blue = pts_blue.reshape((-1, 1, 2))
            cv2.polylines(im0, [pts_blue], isClosed=False,
                          color=(255, 0, 0), thickness=2)

        # 현재까지의 궤적을 비디오 프레임에 그리기 (빨간 점)
        if len(red_dot_trajectory_points) > 1:
            pts_red = np.array(red_dot_trajectory_points, np.int32)
            pts_red = pts_red.reshape((-1, 1, 2))
            cv2.polylines(im0, [pts_red], isClosed=False,
                          color=(0, 0, 255), thickness=2)

        # 수정된 프레임을 비디오 파일에 쓰기
        vid_writer.write(im0)

    # 비디오 라이터 해제
    vid_writer.release()
    print(f"Custom annotated video saved to {output_video_path}")

    # 궤적 Plotting
    plot_trajectory(hyoid_x_list, hyoid_y_list, save_dir, clean_stem,
                    title_label="Trajectory of Detected Hyoid (Red Dot)",
                    filename_suffix="_red_dot_trajectory.jpg",
                    line_color='cadetblue', quiver_color='darksalmon')

    plot_trajectory(blue_dot_x_list, blue_dot_y_list, save_dir, clean_stem,
                    title_label="Trajectory of Corrected Hyoid (Blue Dot)",
                    filename_suffix="_blue_dot_trajectory.jpg",
                    line_color='royalblue', quiver_color='mediumpurple')

    # 궤적 데이터 CSV로 저장
    save_trajectories_to_csv(hyoid_x_list, hyoid_y_list,
                             blue_dot_x_list, blue_dot_y_list, save_dir, clean_stem)

    print("Custom detection script finished.")


def parse_opt():
    """Parses command-line arguments for YOLOv5 detection, setting inference options and model configurations."""
    parser = argparse.ArgumentParser()
    parser.add_argument('--weights', type=str,
                        default='yolov8n.pt', help='model path or triton URL')
    parser.add_argument('--source', type=str, required=True,
                        help='source directory for images or videos')
    # parser.add_argument('--source', type=str, default='0', help='source default to webcam')
    return opt


if __name__ == '__main__':
    opt = parse_opt()
    run(weights=opt.weights, source=opt.source)

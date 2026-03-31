import ultralytics
import argparse
import re
import pandas as pd
import cv2
import numpy as np
import math
from pathlib import Path
from ultralytics import YOLO
from ultralytics.utils.files import increment_path
import sys
import os

# 상위 디렉토리를 Python 경로에 추가
parent_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if parent_dir not in sys.path:
    sys.path.insert(0, parent_dir)


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
    한글이 포함된 파일명에서 한글과 공백을 제거합니다.
    """
    # 한글 제거
    cleaned = re.sub('[ㄱ-ㅎㅏ-ㅣ가-힣]+', '', filename)
    # 공백 제거
    cleaned = re.sub(r'\s+', '', cleaned)
    return cleaned


def run(weights, source, conf=0.25):
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
    results = model.predict(source, conf=conf, stream=True)

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

    # Red trajectory CSV 저장 (설골 원본 좌표)
    if hyoid_x_list:
        red_df = pd.DataFrame({
            'frame': range(len(hyoid_x_list)),
            'red_x': hyoid_x_list,
            'red_y': hyoid_y_list,
        })
        red_csv_path = save_dir / f"{clean_stem}_red_trajectories.csv"
        red_df.to_csv(red_csv_path, index=False, encoding='utf-8-sig')
        print(f"Red trajectories saved to {red_csv_path}")
    else:
        print("WARNING: 설골이 탐지되지 않아 red trajectory CSV를 저장하지 않습니다.")

    # Blue trajectory CSV 저장 (보정된 설골 좌표)
    if blue_dot_x_list:
        blue_df = pd.DataFrame({
            'frame': range(len(blue_dot_x_list)),
            'blue_x': blue_dot_x_list,
            'blue_y': blue_dot_y_list,
        })
        blue_csv_path = save_dir / f"{clean_stem}_blue_trajectories.csv"
        blue_df.to_csv(blue_csv_path, index=False, encoding='utf-8-sig')
        print(f"Blue trajectories saved to {blue_csv_path}")
    else:
        print("WARNING: 보정된 설골 좌표가 없어 blue trajectory CSV를 저장하지 않습니다.")

    print("Custom detection script finished.")


def parse_opt():
    """Parses command-line arguments for YOLOv5 detection, setting inference options and model configurations."""
    parser = argparse.ArgumentParser()
    parser.add_argument('--weights', type=str,
                        default='yolov8n.pt', help='model path or triton URL')
    parser.add_argument('--source', type=str, required=True,
                        help='source directory for images or videos')
    parser.add_argument('--conf', type=float, default=0.25,
                        help='confidence threshold (default: 0.25)')
    opt = parser.parse_args()
    return opt


if __name__ == '__main__':
    opt = parse_opt()
    run(weights=opt.weights, source=opt.source, conf=opt.conf)

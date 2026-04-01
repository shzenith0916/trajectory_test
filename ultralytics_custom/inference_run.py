"""cascade detection"""

import sys
import os
from pathlib import Path
import argparse
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import cv2
import math
from ultralytics import YOLO
from ultralytics.utils.files import increment_path
from two_stage_utils import anonymize_filename, crosspoint, find_min_max_points, find_abcd_pts  # plot_trajectory
from two_stage_utils import save_points_to_csv, save_stage2_to_csv, calculate_speed, calculate_abcd_speeds, get_airway_roi


# 상위 디렉토리를 Python 경로에 추가
# os.path.dirname 상위디렉토리 경로 반환, abspath는 절대경로 반환
# __file__은 현재 파일의 경로 반환
parent_dir = os.path.dirname(os.path.abspath(__file__))
if parent_dir not in sys.path:
    sys.path.insert(0, parent_dir)

# 현재 디렉토리를 Python 경로에 추가
current_dir = os.getcwd()
if current_dir not in sys.path:
    sys.path.insert(0, current_dir)


def run(weights_stage1, weights_stage2, source, output_dir=None,
        class1_id=0, class2_id=1, class3_id=0,
        conf_stage1=0.5, conf_stage2=0.25, visualize=True):
    """
    2단계 추론을 수행하는 메인 함수
    Stage 1: 전체 프레임에서 설골(hyoid bone)과 목뼈(neck bone) 탐지
    Stage 2: 설골 주변 영역을 크롭하여 세밀한 객체(bolus, epiglottis 등) 탐지

    Args:
        weights_stage1: Stage 1 모델 weight 파일 경로
        weights_stage2: Stage 2 모델 weight 파일 경로
        source: 입력 비디오/이미지 경로
        class1_id: hyoid bone 클래스 ID (기본값: 0)
        class2_id: neck bone 클래스 ID (기본값: 1)
        class3_id: Stage 2에서 감지할 클래스 ID (기본값: 0)
        conf_stage1: Stage 1 confidence threshold
        conf_stage2: Stage 2 confidence threshold
        visualize: 시각화 여부
    """
    # ======================== 모델 로드 =========================
    # stage1 모델 로드
    print(f"Loading Stage 1 model: {weights_stage1}")
    model_stage1 = YOLO(weights_stage1)

    # stage2 모델 로드
    model_stage2 = None
    use_stage2 = False
    if weights_stage2 and weights_stage2 != weights_stage1:
        print(f"Loading Stage 2 model: {weights_stage2}")
        model_stage2 = YOLO(weights_stage2)
        use_stage2 = True
    else:
        print("Stage 2 model not provided or same as Stage 1. \n"
              "Skipping Stage 2 detection.")

    # =============== 파일명 및 저장 디렉토리 생성 ================
    # 출력 파일명에서 비식별화 진행
    source_stem = Path(source).stem
    clean_stem = anonymize_filename(source_stem)
    base_filename = clean_stem

    # 결과 저장을 위한 디렉토리 설정
    if output_dir:
        save_dir = Path(output_dir) / f"{clean_stem}_two_stage"
        save_dir.mkdir(parents=True, exist_ok=True)
    else:
        save_dir = increment_path(Path('runs/detect') /
                                  f"{clean_stem}_two_stage", exist_ok=False)
        save_dir.mkdir(parents=True, exist_ok=True)

    # =============== 비디오 불러오기  ================
    # 비디오 정보 가져오기
    cap = cv2.VideoCapture(source)
    if not cap.isOpened():
        print(f"Error: Could not open video source.")

    w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = cap.get(cv2.CAP_PROP_FPS)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    if fps > 0:
        duration_seconds = total_frames / fps
    else:
        duration_seconds = 0
    duration_mins = duration_seconds / 60
    output_path = str(save_dir / f"{clean_stem}_two_stage.avi")
    vid_writer = cv2.VideoWriter(
        output_path, cv2.VideoWriter_fourcc(*'mp4v'), fps, (w, h))
    # 정보만 얻고 해제
    cap.release()

    video_info = {
        'video_duration_seconds': duration_seconds,
        'video_duration_mins': duration_mins,
        'video_fps': fps,
        'video_width': w,
        'video_height': h
    }

    print('\n====================== video info =========================')
    print(video_info)

    video_df = pd.DataFrame([video_info])
    csv_path = save_dir / f"{base_filename}_VideoInfo.csv"
    video_df.to_csv(csv_path, index=False, encoding='utf-8-sig')

    # ================= 궤적 저장용 리스트 초기화 =================
    red_x_list = []
    red_y_list = []
    blue_x_list = []
    blue_y_list = []
    red_trajectory = []  # 비디오에 빨간점 궤적을 그리기 위한 리스트
    blue_trajectory = []  # 비디오에 파란점(보정된 설골) 궤적을 그리기 위한 리스트

    hb_xyxy_tuples = None
    stage2_detections = []

    # ==================== Stage 1 추론 ====================
    result_stage1 = model_stage1.predict(
        source, stream=True)  # 스트림 모드로 메모리 효율성 확보
    frame_idx = 0

    # 프레임별 결과 처리
    for r in result_stage1:
        im0 = r.orig_img.copy()
        boxes = r.boxes

        neck_centers = []
        nb_xyxy_per_frame_list = []  # 프레임마다 초기화 해야됨.
        hb_center = None

        # 각 박스에 대해 클래스별로 처리
        for box in boxes:
            cls = int(box.cls[0])

    # =================== 설골 및 목뼈 좌표 계산 ===================
            # xywh 형식의 좌표
            x_center, y_center, _, _ = box.xywh[0]  # 스칼라 값이므로 변환 필요
            x_int, y_int = int(x_center), int(y_center)

            # xyxy 형식의 좌표
            x1, y1, x2, y2 = box.xyxy[0]

            if cls == class1_id:  # hb
                red_x_list.append(x_int)
                red_y_list.append(y_int)
                hb_center = (x_int, y_int)
                # 비디오에 그릴 빨간 점 궤적 좌표 추가
                red_trajectory.append((hb_center))

                # xyxy 형식의 좌표를 hb_tuples 값에 넣음
                hb_xyxy_tuples = (x1, y1, x2, y2)

            elif cls == class2_id:  # neck bone
                neck_centers.append((x_int, y_int))
                # xyxy 좌표 사용 코드 위치
                nb_xyxy_per_frame_list.append((x1, y1, x2, y2))

    # ======================= 설골-목뼈 교점 계산 =========================
        # 선 그리기 및 거리 계산 로직 (설골-목뼈 교점 계산)
        # 들여쓰기가 이 위치인 이유는, 프레임별로 박스가 찾아지기 때문 -> 선도 계속 업데이트
        if len(neck_centers) >= 2 and hb_center:
            data_points = np.array(neck_centers, dtype=np.float32)
            y_min_pts_highest, y_max_pts_lowest = find_min_max_points(
                data_points)

            # 목뼈들의 중심선 fitting
            [vx, vy, x0, y0] = cv2.fitLine(
                data_points, cv2.DIST_L2, 0, 0.01, 0.01)

            if vx == 0:  # 수직선인 특이 케이스
                continue

            # 기울기 계산 # numpy scalar을 명시적으로 float으로 변환
            m1 = float(vy / vx) if vx != 0 else float('inf')
            # y절편 계산
            c1 = float(y0 - m1 * x0)

            # 목뼈 중심선의 시작점과 끝점 계산: 위쪽 아래쪽 목뼈의 y 좌표
            y_start_nb = float(y_min_pts_highest[1])  # 가장 위쪽 목뼈
            y_end_nb = float(y_max_pts_lowest[1])   # 가장 아래쪽 목뼈

            # 목뼈 중심선 방정식: y = m1*x + c1  # 방정식 변형: x = (y - c1) / m1 # 좌표 역산하여 x값 계산
            x_start_nb = (y_start_nb - c1) / m1
            x_end_nb = (y_end_nb - c1) / m1

            # OpenCV에서 사용할 수 있도록 정수 좌표로 변환 필요. 튜플.
            start_point = int(x_start_nb), int(y_start_nb)
            end_point = int(x_end_nb), int(y_end_nb)

            # 목뼈 중심선 그리기 # 0,255,0 은 초록색
            cv2.line(im0, start_point, end_point,
                     color=(0, 255, 0), thickness=2)

            # 만나는 지점 및 거리 계산
            if m1 == 0:  # 수평선인 특이 케이스
                intercept_pt = (hb_center[0], int(c1))
                distance = abs(hb_center[1] - c1)

            else:  # 일반적 케이스
                m2 = -1 / m1  # class1에서 class2 목뼈 중심선으로 수직선
                c2 = float(hb_center[1] - m2 * hb_center[0])

                try:
                    # 교점 계산
                    intercept_x, intercept_y = crosspoint(m1, c1, m2, c2)
                    intercept_pt = (int(intercept_x), int(intercept_y))

                    # 수직선 그리기
                    vertical_start_point = (int(intercept_x), start_point[1])
                    vertical_end_point = (int(intercept_x), end_point[1])
                    # 흰색으로 수직선 그리기
                    cv2.line(im0, vertical_start_point, vertical_end_point,
                             color=(255, 255, 255), thickness=2)

                    # 설골에서 교점까지의 거리 계산
                    distance = math.dist(hb_center, intercept_pt)

                except (ValueError, ZeroDivisionError) as e:
                    print(
                        f"Frame {frame_idx} - Could not calculate intersection: {e}")
                    continue

            # 공통: 선 그리는 로직
            cv2.line(im0, hb_center, intercept_pt, (0, 255, 0), 2)
            # 설골 위치에 빨간 점 그리기
            cv2.circle(im0, hb_center, 5, (0, 0, 255), -1)  # 0,0,255가 빨간색

            # Blue dot: 교점에서 거리만큼 떨어진 수평 위치
            blue_dot = (int(intercept_pt[0] - distance), int(intercept_pt[1]))
            # 파란 점의 좌표 각 x,y 리스트와 궤적 리스트에 추가
            blue_x_list.append(blue_dot[0])
            blue_y_list.append(blue_dot[1])
            blue_trajectory.append(blue_dot)

            # 90도 직각의 수평선 그리기
            cv2.line(im0, blue_dot, (int(intercept_pt[0]),
                                     int(intercept_pt[1])), color=(255, 255, 255), thickness=2)
            cv2.circle(im0, blue_dot, 5, (255, 0, 0), -1)  # 255,0,0이 파란색

        # 현재까지의 궤적을 비디오 프레임에 그리기 (파란 점)
        if len(blue_trajectory) > 1:
            pts_blue = np.array(blue_trajectory, np.int32)
            pts_blue = pts_blue.reshape((-1, 1, 2))
            cv2.polylines(im0, [pts_blue], isClosed=False,
                          color=(255, 0, 0), thickness=2)

        # 현재까지의 궤적을 비디오 프레임에 그리기 (빨간 점)
        if len(red_trajectory) > 1:
            pts_red = np.array(red_trajectory, np.int32)
            pts_red = pts_red.reshape((-1, 1, 2))
            cv2.polylines(im0, [pts_red], isClosed=False,
                          color=(0, 0, 255), thickness=2)

    # ======================= Stage2 추론 =========================
        # stage2 추론을 위한 데이터 저장
        stage2_results_frame = []
        hb_coords = hb_xyxy_tuples

        if use_stage2 and hb_center is not None:
            """
            Stage 2 작동 방식:
            1. hyoid bone bbox 아래의 주요 관심심 영역을 크롭
            2. 크롭된 영역에서만 세밀한 객체(bolus, epiglottis 등) 탐지
            3. 탐지 결과를 원본 프레임 좌표로 변환하여 시각화

            장점:
            - 관심 영역(ROI)만 처리하여 속도 향상
            - 작은 객체도 고해상도로 탐지 가능
            - False positive 감소
            """

            # 프레임 크롭
            crop_x1, crop_y1, crop_x2, crop_y2 = get_airway_roi(
                hb_coords, nb_xyxy_per_frame_list, im0.shape)
            # 영역을 잘라내면, (0,0)은 (crop_x1, crop_y1)과 같음
            cropped_region = im0[int(crop_y1): int(
                crop_y2), int(crop_x1):int(crop_x2)]

            # ROI 시각화하되, 크롭 후 한번만
            if visualize:
                cv2.rectangle(im0, (crop_x1, crop_y1),
                              (crop_x2, crop_y2), (0, 255, 255), 1)  # 노란색 점선으로 크롭 영역 처리 -> 1은 점선

            # Stage2 모델 추론
            if cropped_region.size > 0:
                results_stage2 = model_stage2.predict(cropped_region,
                                                      conf=conf_stage2,
                                                      verbose=False)

                for result in results_stage2:
                    boxes_stage2 = result.boxes

                    for box in boxes_stage2:
                        # 위 stage1에서 cls 사용해서, cls2로 변수 생성하여 사용
                        cls2 = int(box.cls[0])

                        if cls2 == class3_id:
                            # croppend region에서 stage2 모델이 반환한 좌표
                            x1_crop, y1_crop, x2_crop, y2_crop = box.xyxy[0]

                            # 때문에, 원본 프레임 좌표로 변환
                            x1_orig = int(x1_crop + crop_x1)  # crop 왼쪽 시작점 더함
                            y1_orig = int(y1_crop + crop_y1)  # crop 위쪽 시작점 더함
                            # crop 왼쪽 시작점 더함 (x1과 동일한 offset)
                            x2_orig = int(x2_crop + crop_x1)
                            # crop 위쪽 시작점 더함 (y1과 동일한 offset)
                            y2_orig = int(y2_crop + crop_y1)

                            conf_2 = float(box.conf[0])

                            # 탐지된 결과 저장하기
                            stage2_results_frame.append({
                                'bbox': (x1_orig, y1_orig, x2_orig, y2_orig),
                                'conf': conf_2,
                                'class': cls2
                            })

                            # 시각화: stage2 탐지 객체 bbox 자주색으로 처리
                            if visualize:
                                cv2.rectangle(im0, (x1_orig, y1_orig),
                                              (x2_orig, y2_orig),
                                              (128, 0, 128), 2)  # 자주색
                                cv2.putText(im0, f'{conf_2:.2f}',
                                            (x1_orig, y1_orig - 5),
                                            cv2.FONT_HERSHEY_SIMPLEX,
                                            0.5, (128, 0, 128), 2)

                    # for loop 밖으로 위치하게 하여, 프레임당 한번만 저장하도록
                    if len(stage2_results_frame) > 0:
                        # Stage 2 탐지 결과 저장
                        stage2_detections.append({
                            'frame': frame_idx,
                            'detections': stage2_results_frame
                        })

        # 수정된 프레임을 비디오 파일에 쓰기
        vid_writer.write(im0)

        frame_idx += 1

    # ================== 비디오 저장 완료 ===================
    vid_writer.release()  # 비디오 라이터 해제
    print(f"Annotated video saved to {output_path}")

# ===========================================================
# ABCD 포인트 및 속도 계산
# ===========================================================
    if not red_x_list or not red_y_list:
        print("\nWARNING: 설골(hyoid bone)이 탐지되지 않았습니다. "
              "conf_stage1 값을 낮추거나 모델을 확인하세요.")
        red_abcd_pts = None
        blue_abcd_pts = None
        red_speed_data = None
        blue_speed_data = None
        red_abcd_speeds = None
        blue_abcd_speeds = None
    else:
        print("\n======== Calculating ABCD points =========")
        red_abcd_pts = find_abcd_pts(red_x_list, red_y_list)
        blue_abcd_pts = find_abcd_pts(
            blue_x_list, blue_y_list) if blue_x_list else None

        print("\n======== Calculating velocities/speed =========")
        red_speed_data = calculate_speed(red_x_list, red_y_list, fps)
        blue_speed_data = calculate_speed(
            blue_x_list, blue_y_list, fps) if blue_x_list else None

        print("\n======== Calculating ABCD points speed =========")
        red_abcd_speeds = calculate_abcd_speeds(
            red_x_list, red_y_list, red_abcd_pts, fps)
        blue_abcd_speeds = calculate_abcd_speeds(
            blue_x_list, blue_y_list, blue_abcd_pts, fps) if blue_x_list else None

# # ===========================================================
# # 궤적 Plotting
# # ===========================================================
#     plot_trajectory(red_x_list, red_y_list, red_abcd_pts, save_dir, clean_stem,
#                     title_label="Trajectory of Detected Hyoid (Red Dot)",
#                     filename_suffix="_red_dot_trajectory.jpg",
#                     line_color='cadetblue', quiver_color='darksalmon')

#     plot_trajectory(blue_x_list, blue_y_list, blue_abcd_pts, save_dir, clean_stem,
#                     title_label="Trajectory of Corrected Hyoid (Blue Dot)",
#                     filename_suffix="_blue_dot_trajectory.jpg",
#                     line_color='royalblue', quiver_color='mediumpurple')

# ===========================================================
# CSV 저장 (DataFrame 반환받기)
# ===========================================================
    # 궤적과 속도데이터 df 그리고, abcd 포인트 관련 df를 각각 CSV로 저장
    moving_df, abcd_df = save_points_to_csv(red_x_list, red_y_list, blue_x_list, blue_y_list,
                                            red_speed_data, blue_speed_data,
                                            red_abcd_speeds, blue_abcd_speeds,
                                            save_dir, base_filename,
                                            frame_w=w, frame_h=h)

    print("First stage of detection with hyoid-neck intersection calculation finished.")

    # stage2 탐지 결과도 csv로 저장하고 데이터 프레임 받기
    stage2_df = save_stage2_to_csv(stage2_detections, save_dir, base_filename)

    return {'video_path': str(source),
            'output_dir': str(save_dir),
            'video_output': str(output_path),
            # csv 파일 경로들
            'csv_files': {
                'video_info': str(csv_path),
                'moving_points': str(save_dir / f"{base_filename}_moving_points.csv"),
                'abcd_points': str(save_dir / f"{base_filename}_abcd_points.csv"),
    },
        # # 이미지 파일 경로들들
        # 'trajectory_plots': {
        #         'red': str(save_dir / f"{base_filename}_red_dot_trajectory.jpg"),
        #         'blue': str(save_dir / f"{base_filename}_blue_dot_trajectory.jpg"),
        # },
        # 데이터 프레임 객체들 (DB 직접 저장용)
        'dataframes': {
                'video_info': video_df,
                'moving_points': moving_df,
                'abcd_points': abcd_df,
                'stage2_detection': stage2_df
    }
    }


def parse_opt():
    """커맨드 라인 인자 파싱"""
    parser = argparse.ArgumentParser()
    parser.add_argument('--weights-stage1', type=str,
                        default='yolov8n.pt', help='Stage 1 model path (hyoid & neck bone detection)')
    parser.add_argument('--weights-stage2', type=str,
                        default='yolov8n.pt', help='Stage 2 model path (optional)')
    parser.add_argument('--source', type=str, required=True,
                        help='source directory for images or videos')
    parser.add_argument('--class1-id', type=int, default=0,
                        help='Class ID for hyoid bone (default: 0)')
    parser.add_argument('--class2-id', type=int, default=1,
                        help='Class ID for neck bone (default: 1)')
    parser.add_argument('--class3-id', type=int, default=0,
                        help='Class ID for stage 2 detection (default: 0)')
    parser.add_argument('--conf-stage1', type=float, default=0.5,
                        help='Confidence threshold for stage 1 (default: 0.5)')
    parser.add_argument('--conf-stage2', type=float, default=0.25,
                        help='Confidence threshold for stage 2 (default: 0.25)')
    parser.add_argument('--visualize', action='store_true',
                        help='Visualize results')
    opt = parser.parse_args()
    return opt


if __name__ == '__main__':
    opt = parse_opt()
    result = run(weights_stage1=opt.weights_stage1,
                 weights_stage2=opt.weights_stage2,
                 source=opt.source,
                 class1_id=opt.class1_id,
                 class2_id=opt.class2_id,
                 class3_id=opt.class3_id,
                 conf_stage1=opt.conf_stage1,
                 conf_stage2=opt.conf_stage2,
                 visualize=opt.visualize)
    print(f"\nResults saved to: {result['output_dir']}")

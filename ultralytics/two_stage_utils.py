import sys
import os
import re
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import math
from ultralytics import YOLO
from ultralytics.utils.files import increment_path

# 상위 디렉토리를 Python 경로에 추가
# os.path.dirname 상위디렉토리 경로 반환, abspath는 절대경로 반환
# __file__은 현재 파일의 경로 반환
parent_dir = os.path.dirname(os.path.abspath(__file__))
if parent_dir not in sys.path:
    sys.path.insert(0, parent_dir)


def anonymize_filename(filename):
    """
    1) 영문이름 비식별화 
    2) 공백제거
    3) 한글이름 비식별화
    """

    def mask_english(match):
        first_name = match.group(1)  # John
        last_name = match.group(2)  # Smith

        # 이름만 마스킹
        if len(first_name) <= 2:
            masked_first = first_name[0] + '-'
        else:
            masked_first = first_name[0] + '-' * (len(first_name) - 1)

        return f"{last_name}_{masked_first}"

    pattern = r'\b([A-Z][a-z]+)\s+([A-Z][a-z]+)\b'
    masked_english = re.sub(pattern, mask_english, filename)

    # 공백 제거
    no_space = re.sub(r'\s+', '', masked_english)

    def mask_korean(match):
        name = match.group()
        length = len(name)

        if length <= 2:
            return name[0] + '-'
        elif length == 3:
            return name[0] + '-' + name[-1]
        else:
            return name[0] + '-' * (length-2) + name[-1]

    masked_korean = re.sub('[가-힣]+', mask_korean, no_space)

    return masked_korean


def crosspoint(m1, c1, m2, c2):
    if m1 == m2:
        raise ValueError("The lines are parallel and do not intersect")
    x = (c2 - c1) / (m1 - m2)
    y = m1 * x + c1
    return x, y


def find_min_max_points(data_points):
    """
    주어진 데이터 포인트에서 Y 좌표의 최소값과 최대값을 가진 포인트를 찾기.

    이미지 좌표계에서:
    - Y 최소값 = 화면상 가장 위쪽에 있는 점
    - Y 최대값 = 화면상 가장 아래쪽에 있는 점

    목뼈(cervical vertebrae) 검출 시, 여러 개의 목뼈 중심점들이 감지되는데,
    이 중에서 가장 위쪽과 가장 아래쪽 목뼈를 찾아서 중심선의 범위를 결정.

    Args:
        data_points: numpy 배열 형태의 좌표 리스트 [[x1, y1], [x2, y2], ...]

    Returns:
        y_min_pts: Y 좌표가 최소인 포인트 (가장 위쪽 목뼈)
        y_max_pts: Y 좌표가 최대인 포인트 (가장 아래쪽 목뼈)
    """
    data_points = np.array(data_points)
    second_elements = data_points[:, 1]  # 모든 y 좌표 추출
    min_index = np.argmin(second_elements)  # y 최소값의 인덱스
    max_index = np.argmax(second_elements)  # y 최대값의 인덱스
    y_min_pts = data_points[min_index]  # 가장 위쪽 점
    y_max_pts = data_points[max_index]  # 가장 아래쪽 점

    return y_min_pts, y_max_pts


def get_bbox_region(box, img_shape, margin_ratio=0.2):
    """
    Args:
        box: YOLO box 객체
        img_shape: (height, width) 이미지 크기
        margin_ratio: 추가 여유 공간 비율 (기본 20%)

    Returns:
        (x1, y1, x2, y2): 크롭할 영역의 좌표
    """

    h, w = img_shape[:2]
    x1, y1, x2, y2 = box.xyxy[0].cpu().numpy()

    # 박스크기 계산
    box_w = x2 - x1
    box_h = y2 - y1

    # margin ratio 만큼 여유 공간 만들기.
    margin_w = box_w * margin_ratio
    margin_h = box_h * margin_ratio

    # 크롭할 최종 영역 계산
    crop_x1 = max(0, int(x1 - margin_w))
    crop_y1 = max(0, int(y1 - margin_h))
    crop_x2 = min(w, int(x2 + margin_w))
    crop_y2 = min(w, int(y2 + margin_h))

    return (crop_x1, crop_y1, crop_x2, crop_y2)


def get_airway_roi(box1_coords, box2_list, img_shape):
    """
    두개 bounding box 주변 영역을 계산합니다.

    Args:
        box1: class1의 YOLO box 객체의 xyxy 좌표 튜플 (x1, y1, x2, y2)
        box2: class2의 YOLO box 객체의 xyxy 좌표 리스트 [(x1, y1, x2, y2), (x1, y1, x2, y2), ...]
        img_shape: (height, width) 이미지 크기


    Returns:
        (x1, y1, x2, y2): 크롭할 영역의 좌표
    """
    # 이미지 크기 처리
    h, w = img_shape[:2]

    # box1(class1-hb)의 xyxy 좌표 가져오기
    x1_hb, y1_hb, x2_hb, y2_hb = box1_coords
    hb_length = x2_hb - x1_hb

    # ===================== x 좌표 계산 ==========================
    # box1의 x1 지점이 최종 roi 영역의 min x 좌표
    roi_min_x = int(x1_hb)
    # box1의 x2 지점에 여유 공간 추가하여 최종 roi 영역의 max x 좌표
    roi_max_x = int(x1_hb + hb_length * 3)  # 원래 박스의 3배 길이만큼 여유 공간 추가

    # ===================== y 좌표 계산 ==========================
   # ROI의 최대 y 좌표는 box1의 y2
    roi_y_start = y2_hb
    max_neck_y2 = 0
    max_neck_height = 0

    # box2의 y좌표 리스트 가져오기
    for box2_coord in box2_list:
        x1_nb, y1_nb, x2_nb, y2_nb = box2_coord  # 튜플 언패킹

        # class2의 경우 박스가 여러개 탐지될 수 있으므로, 모든 박스 중 가장 아래 박스 값 사용용
        # y2가 가장 큰 (화면상 가장 아래) neck bone 찾기
        if y2_nb > max_neck_y2:
            # y2의 최대값 구하기 -> 화면상 가장 아래쪽 neck_bone의 오른쪽 아래 코너 좌표
            max_neck_y2 = y2_nb
            max_neck_height = y2_nb - y1_nb

    # 최종 roi 영역의 min y좌표는 box2의 y2의 두배만큼 아래쪽으로 이동
    roi_y_end = int(max_neck_y2 + max_neck_height * 2)

    return (roi_min_x, roi_y_start, roi_max_x, roi_y_end)


def calculate_speed(x_coords, y_coords, fps):
    """ 주어진 좌표를 사용하여 이동 속도 계산

    Args:
        x_coords: x 좌표 리스트
        y_coords: y 좌표 리스트
        fps: 프레임 속도

    Returns:
        speed: 이동 속도
    """

    if not x_coords or not y_coords:
        return None

    x_arr = np.array(x_coords)
    y_arr = np.array(y_coords)

    dx = np.diff(x_arr)
    dy = np.diff(y_arr)
    frame_dist = np.sqrt(dx**2 + dy**2)  # 피타고라스 정리: 대각선 거리 = √(dx² + dy²)

    # 누적 거리 계산
    cumulative_dist = np.cumsum(frame_dist)

    # 프레임 간 시간 간격 계산
    time_per_frame = 1.0 / fps  # seconds
    # 프레임 간 시간 간격 리스트 생성. np.full로 주어진 길이만큼 채워진 배열 생성. 값은 time_per_frame
    time_intervals = np.full(len(frame_dist), time_per_frame)

    # 순간 속도 계산
    instantaneous_speed = frame_dist / time_per_frame  # pixel/frame 단위

    # 통계량 계산
    avg_velocity = np.mean(instantaneous_speed)
    max_velocity = np.max(instantaneous_speed)
    total_time = len(x_coords) * time_per_frame  # seconds

    return {'frame_dist': frame_dist.tolist(),
            'culmulative_dist': cumulative_dist.tolist(),
            'time_intervals': time_intervals.tolist(),
            'instantaneous_speed': instantaneous_speed.tolist(),
            'avg_velocity': float(avg_velocity),
            'max_velocity': float(max_velocity),
            'total_time': float(total_time)}


def save_stage2_to_csv(stage2_prediction, save_dir, base_filename):
    """
      Args:
      stage2_prediction: stage2 모델 추론 결과 값 딕셔너리
      save_dir: 저장 디렉토리
      base_filename: csv로 저장할 파일명의 베이스 파일명
      """

    stage2_data = []

    for frame_data in stage2_prediction:
        for det in frame_data['detections']:
            stage2_data.append({
                'frame': frame_data['frame'],
                'x1': det['bbox'][0],
                'y1': det['bbox'][1],
                'x2': det['bbox'][2],
                'y2': det['bbox'][3],
                'confidence': det['conf'],
                'class': det['class']
            })

    df_stage2 = pd.DataFrame(stage2_data)
    csv_path = save_dir / f"{base_filename}_stage2_detection.csv"
    df_stage2.to_csv(csv_path, index=False, encoding='utf-8-sig')

    print(
        f"Stage2 Detections saved to {csv_path} and this method returns DataFrame")
    return df_stage2


def find_abcd_pts(x_coords, y_coords):
    """
    좌표 리스트로부터 ABCD 포인트와 추가 특수 포인트를 계산합니다.

    주의: 이미지 좌표계에서는 Y축이 아래로 증가합니다 (원점이 왼쪽 위)
    - y_min (min y value) = 화면상 가장 위 = Highest point in visual
    - y_max (max y value) = 화면상 가장 아래 = Lowest point in visual

    Args:
        x_coords: x 좌표 리스트
        y_coords: y 좌표 리스트
    """

    x_arr = np.array(x_coords)
    y_arr = np.array(y_coords)

    # 특수 포인트 인덱스 계산
    min_x_idx = np.argmin(x_coords)
    min_y_idx = np.argmin(y_coords)  # 이미지 좌표계에서 가장 위 (Highest in visual)
    max_x_idx = np.argmax(x_coords)
    max_y_idx = np.argmax(y_coords)  # 이미지 좌표계에서 가장 아래 (Lowest in visual)

    # 딕셔너리 형태로 저장
    abcd_and_special_pts = {
        # 시작점
        'A_start': (float(x_arr[0]), float(y_arr[0])),
        # y좌표가 최대인 점
        'B_highest': (float(x_arr[min_y_idx]), float(y_arr[min_y_idx])),
        # x좌표가 최소인 점
        'C_xmin': (float(x_arr[min_x_idx]), float(y_arr[min_x_idx])),
        # 끝점
        'D_end': (float(x_arr[-1]), float(y_arr[-1])),
        # y좌표가 최소인 점
        'y_min': (float(x_arr[max_y_idx]), float(y_arr[max_y_idx])),
        # x좌표가 최고인 점
        'x_max': (float(x_arr[max_x_idx]), float(y_arr[max_x_idx]))
    }

    return abcd_and_special_pts


def calculate_abcd_speeds(x_coords, y_coords, abcd_points, fps):
    """ABCD 포인트 간 구간별 이동 속도 계산"""

    x_arr = np.array(x_coords)
    y_arr = np.array(y_coords)

    # abcd 포인트 값 불러오기
    a_start = abcd_points['A_start']
    b_highest = abcd_points['B_highest']
    c_xmin = abcd_points['C_xmin']
    d_end = abcd_points['D_end']

    # 각 포인트의 프레임 인덱스 찾기
    idx_A = 0  # 시작점
    # np.where 는 튜플 반환 (row_inx,col_idx) ->[0] 첫번째
    # 두번째 [0]은 매칭된 인덱스 값 중 첫번째 값. 같은 좌표가 여러개 있을 수 있으므로, 첫번째 발생 시점만 필요
    matched_indices = np.where(
        (x_arr == b_highest[0]) & (y_arr == b_highest[1]))
    indices_array = matched_indices[0]
    idx_B = indices_array[0]  # 배열에서 첫 번째 값: 스칼라 인덱스
    idx_C = np.where((x_arr == c_xmin[0]) & (y_arr == c_xmin[1]))[0][0]
    idx_D = len(x_arr) - 1  # 끝점

    def euclidean_dist(p1, p2):
        return math.sqrt((p2[0] - p1[0])**2 + (p2[1] - p1[1])**2)

    # 구간별 거리 계산
    dist_A_to_B = euclidean_dist(a_start, b_highest)
    dist_B_to_C = euclidean_dist(b_highest, c_xmin)
    dist_C_to_D = euclidean_dist(c_xmin, d_end)

    # 구간별 소요 시간 (프레임 수/ fps)
    time_A_to_B = (idx_B - idx_A) / fps
    time_B_to_C = (idx_C - idx_B) / fps
    time_C_to_D = (idx_D - idx_C) / fps

    # 구간별 평균 속도 (pixels/second)
    speed_A_to_B = dist_A_to_B / time_A_to_B if time_A_to_B > 0 else 0
    speed_B_to_C = dist_B_to_C / time_B_to_C if time_B_to_C > 0 else 0
    speed_C_to_D = dist_C_to_D / time_C_to_D if time_C_to_D > 0 else 0

    # 전체 이동 거리 및 시간
    total_distance = dist_A_to_B + dist_B_to_C + dist_C_to_D
    total_time = time_A_to_B + time_B_to_C + time_C_to_D
    avg_speed_total = total_distance / total_time if total_time > 0 else 0

    return {
        # 구간별 거리
        'distance_A_to_B': float(dist_A_to_B),
        'distance_B_to_C': float(dist_B_to_C),
        'distance_C_to_D': float(dist_C_to_D),
        'total_distance': float(total_distance),

        # 구간별 시간
        'time_A_to_B': float(time_A_to_B),
        'time_B_to_C': float(time_B_to_C),
        'time_C_to_D': float(time_C_to_D),
        'total_time': float(total_time),

        # 구간별 속도 (pixels/second)
        'speed_A_to_B': float(speed_A_to_B),
        'speed_B_to_C': float(speed_B_to_C),
        'speed_C_to_D': float(speed_C_to_D),
        'avg_speed_total': float(avg_speed_total),

        # 프레임 인덱스
        'frame_idx_A': int(idx_A),
        'frame_idx_B': int(idx_B),
        'frame_idx_C': int(idx_C),
        'frame_idx_D': int(idx_D)
    }


def save_points_to_csv(red_x, red_y, blue_x, blue_y,
                       red_speed_data, blue_speed_data,
                       red_abcd_speeds, blue_abcd_speeds,
                       save_dir, base_filename,
                       frame_w=None, frame_h=None):
    """
    주어진 궤적 좌표 리스트들을 pandas DataFrame으로 만들어 CSV 파일로 저장

    정규화 좌표의 필요성:
    - 해상도 독립성: 다른 해상도 비디오 간 비교 가능
      예) Video A (640x480):  hyoid at (320, 240) → normalized (0.5, 0.5)
          Video B (1920x1080): hyoid at (960, 540) → normalized (0.5, 0.5)

    - 임상 연구 활용: 환자 간 정량적 비교 가능
      예) 환자 A (1280x720): 150 pixels 이동 → 0.117 (11.7%)
          환자 B (640x480):  100 pixels 이동 → 0.156 (15.6%)
          → 환자 B가 상대적으로 더 많이 움직임

    - 임상 기준 설정: 표준화된 threshold 설정 가능
      예) if hyoid_excursion < 0.10:  # 화면 크기의 10% 미만
              diagnosis = "reduced hyoid movement"

    Args:
        red_x, red_y: 빨간 점(탐지된 설골) 좌표 리스트
        blue_x, blue_y: 파란 점(보정된 설골) 좌표 리스트
        red_speed_data: calculate_speed() 반환값 (red trajectory)
        blue_speed_data: calculate_speed() 반환값 (blue trajectory)
        save_dir: 저장 디렉토리
        base_filename: 파일명 베이스
        frame_w, frame_h: 프레임 크기 (선택적)
    """
    # ======================기본 궤적 데이터=========================
    max_len = max(len(red_x), len(blue_x))

    df = pd.DataFrame({
        'frame_idx': range(max_len),

        # pandas Series를 사용하면 길이가 다른 리스트도 NaN으로 채워져 안전하게 DataFrame으로 만들 수 있음.
        # RED DOT 절대 좌표
        'red_dot_x': pd.Series(red_x),
        'red_dot_y': pd.Series(red_y),
        # RED DOT 정규화 좌표
        'red_dot_x_norm': pd.Series([x / frame_w for x in red_x]),
        'red_dot_y_norm': pd.Series([y / frame_h for y in red_y]),

        # BLUE DOT 절대 좌표
        'blue_dot_x': pd.Series(blue_x),
        'blue_dot_y': pd.Series(blue_y),
        # BLUE DOT 정규화 좌표
        'blue_dot_x_norm': pd.Series([x / frame_w for x in blue_x]),
        'blue_dot_y_norm': pd.Series([y / frame_h for y in blue_y]),
    })
    # ======================속도 데이터 추가=========================

    if red_speed_data is not None:
        # NaN으로 첫 행을 채우기 (차분diff이므로)
        # np.insert(arr, idx, values): arr 배열의 idx 위치에 values 값을 삽입한 새로운 배열 반환
        df['red_distance'] = pd.Series(
            np.insert(red_speed_data['frame_dist'], 0, np.nan))
        df['red_time_interval'] = pd.Series(
            np.insert(red_speed_data['time_intervals'], 0, np.nan))
        df['red_velocity'] = pd.Series(
            np.insert(red_speed_data['instantaneous_speed'], 0, np.nan))
    else:
        df['red_distnace'] = np.nan
        df['red_time_interval'] = np.nan
        df['red_velocity'] = np.nan

    if blue_speed_data is not None:
        df['blue_distance'] = pd.Series(
            np.insert(blue_speed_data['frame_dist'], 0, np.nan))
        df['blue_time_interval'] = pd.Series(
            np.insert(blue_speed_data['time_intervals'], 0, np.nan))
        df['blue_velocity'] = pd.Series(
            np.insert(blue_speed_data['instantaneous_speed'], 0, np.nan))
    else:
        df['blue_distance'] = np.nan
        df['blue_time_interval'] = np.nan
        df['blue_velocity'] = np.nan

    # ================  ABCD 특수 포인트 계산 및 저장 ================
    # find_abcd_pts 함수의 리턴값 딕셔너리로 value가 (x,y) 튜플
    abcd_data = {}

    if red_x and red_y:
        red_abcd = find_abcd_pts(red_x, red_y)
        for key, (x, y) in red_abcd.items():
            abcd_data[f'red_{key}_x'] = x
            abcd_data[f'red_{key}_y'] = y
            abcd_data[f'red_{key}_x_norm'] = x / frame_w
            abcd_data[f'red_{key}_y_norm'] = y / frame_h

    if blue_x and blue_y:
        blue_abcd = find_abcd_pts(blue_x, blue_y)
        for key, (x, y) in blue_abcd.items():
            abcd_data[f'blue_{key}_x'] = x
            abcd_data[f'blue_{key}_y'] = y
            abcd_data[f'blue_{key}_x_norm'] = x / frame_w
            abcd_data[f'blue_{key}_y_norm'] = y / frame_h

    # ABCD 구간별 속도 추가
    if red_abcd_speeds:
        abcd_data['red_distance_A_to_B'] = red_abcd_speeds['distance_A_to_B']
        abcd_data['red_distance_B_to_C'] = red_abcd_speeds['distance_B_to_C']
        abcd_data['red_distance_C_to_D'] = red_abcd_speeds['distance_C_to_D']
        abcd_data['red_total_distance'] = red_abcd_speeds['total_distance']

        abcd_data['red_time_A_to_B'] = red_abcd_speeds['time_A_to_B']
        abcd_data['red_time_B_to_C'] = red_abcd_speeds['time_B_to_C']
        abcd_data['red_time_C_to_D'] = red_abcd_speeds['time_C_to_D']
        abcd_data['red_total_time'] = red_abcd_speeds['total_time']

        abcd_data['red_speed_A_to_B'] = red_abcd_speeds['speed_A_to_B']
        abcd_data['red_speed_B_to_C'] = red_abcd_speeds['speed_B_to_C']
        abcd_data['red_speed_C_to_D'] = red_abcd_speeds['speed_C_to_D']
        abcd_data['red_avg_speed_total'] = red_abcd_speeds['avg_speed_total']

    if blue_abcd_speeds:
        abcd_data['blue_distance_A_to_B'] = blue_abcd_speeds['distance_A_to_B']
        abcd_data['blue_distance_B_to_C'] = blue_abcd_speeds['distance_B_to_C']
        abcd_data['blue_distance_C_to_D'] = blue_abcd_speeds['distance_C_to_D']
        abcd_data['blue_total_distance'] = blue_abcd_speeds['total_distance']

        abcd_data['blue_time_A_to_B'] = blue_abcd_speeds['time_A_to_B']
        abcd_data['blue_time_B_to_C'] = blue_abcd_speeds['time_B_to_C']
        abcd_data['blue_time_C_to_D'] = blue_abcd_speeds['time_C_to_D']
        abcd_data['blue_total_time'] = blue_abcd_speeds['total_time']

        abcd_data['blue_speed_A_to_B'] = blue_abcd_speeds['speed_A_to_B']
        abcd_data['blue_speed_B_to_C'] = blue_abcd_speeds['speed_B_to_C']
        abcd_data['blue_speed_C_to_D'] = blue_abcd_speeds['speed_C_to_D']
        abcd_data['blue_avg_speed_total'] = blue_abcd_speeds['avg_speed_total']

    # 보통 키가 컬럼명, 벨류가 여러행. 여기서는 전체가 1개 행(스칼라 값)만을 가짐.
    # df = pd.DataFrame(abcd_data) 로 하면, 판다스는 스칼라값만 있으면 몇개 행을 만들어야 할 지 모름.
    # 따라서, 1) 리스트로 감싸거나 2) pd.DataFrame(abcd_data, index=[0]) 로 인덱스를 명시해줘야 함.
    abcd_df = pd.DataFrame([abcd_data])

    # ========================= csv 저장===========================
    csv_path = save_dir / f"{base_filename}_moving_points.csv"
    abcd_path = save_dir / f"{base_filename}_abcd_points.csv"

    # 한글 경로/파일명 문제를 피하기 위해 encoding을 'utf-8-sig'로 지정
    df.to_csv(csv_path, index=False, encoding='utf-8-sig')
    abcd_df.to_csv(abcd_path, index=False, encoding='utf-8-sig')

    print(f"Trajectories data saved to {csv_path}")
    print(f"ABCD points data saved to {abcd_path}")

    return df, abcd_df


def plot_trajectory(x_coords, y_coords, abcd_points,
                    save_dir, base_filename,
                    title_label, filename_suffix,
                    line_color='cadetblue', quiver_color='darksalmon'):
    """
    주어진 좌표 리스트를 사용하여 궤적 그래프를 생성하고 저장합니다.

    Args:
        x_coords: x 좌표 리스트
        y_coords: y 좌표 리스트
        abcd_points: find_abcd_pts()가 반환한 dict
                     {'A_start': (x,y), 'B_highest': (x,y), 'C_xmin': (x,y),
                      'D_end': (x,y), 'y_min': (x,y), 'x_max': (x,y)}
        save_dir: 저장 디렉토리
        base_filename: 파일명 베이스
        title_label: 그래프 제목
        filename_suffix: 파일명 접미사 e.g. ".jpg"
        line_color: 궤적 선 색상
        quiver_color: 화살표 색상
    """
    if not x_coords or not y_coords:
        print(f"No data to plot for {title_label}")
        return

    x_coor = np.array(x_coords)
    y_coor = np.array(y_coords)

    dx = np.diff(x_coor)
    dy = np.diff(y_coor)

    # 미리 계산된 abcd 포인트 사용 (중복 계산 제거)
    A_start = abcd_points['A_start']
    B_lowest = abcd_points['B_highest']
    C_xmin = abcd_points['C_xmin']
    D_end = abcd_points['D_end']

    plt.figure(figsize=(8, 6))
    plt.plot(x_coor, y_coor, marker='o',
             color=line_color, linestyle='-', markersize=3, linewidth=1)

    if len(dx) > 0:
        # quiver 함수는 주어진 x, y 좌표에서 시작해 u, v 방향으로 벡터를 그리는 기능을 제공
        plt.quiver(x_coor[:-1], y_coor[:-1], dx, dy, angles='xy',
                   scale_units='xy', scale=1.5, color=quiver_color, width=0.003)

    plt.gca().invert_yaxis()

    # A: 시작점
    plt.scatter(A_start[0], A_start[1], color='orange', s=100, label='Start')
    plt.annotate('Start(A)', A_start, textcoords='offset points',
                 xytext=(20, 10), ha='center', fontsize=10)
    # D: End 포인트
    plt.scatter(D_end[0], D_end[1], color='seagreen', s=100, label='End')
    plt.annotate('End(D)', D_end, textcoords='offset points',
                 xytext=(0, 10), ha='center', fontsize=10)
    # C: X 최소 포인트
    plt.scatter(C_xmin[0], C_xmin[1], color='peru', s=80, label='X_Min_Point')
    plt.annotate('X_Min(C)', C_xmin, textcoords='offset points', xytext=(
        60, -30), ha='center', arrowprops=dict(arrowstyle="->", connectionstyle="arc3,rad=.8"))
    # B: 최저점 (Y 최대)
    plt.scatter(B_lowest[0], B_lowest[1],
                color='darkblue', s=80, label='Lowest_Point')
    plt.annotate('Lowest Point(B)', B_lowest, textcoords='offset points', xytext=(
        30, 30), ha='center', arrowprops=dict(arrowstyle="->", connectionstyle="arc3,rad=.8"))

    title = f"{title_label}-{base_filename}"
    plt.title(title)
    plt.xlabel("X Coordinate of Hyoid", fontsize=10)
    plt.ylabel("Y Coordinate of Hyoid", fontsize=10)
    plt.grid(True)

    trajectory_path = save_dir / f"{base_filename}{filename_suffix}"
    plt.savefig(trajectory_path)
    plt.close()
    print(f"Trajectory plot saved to {trajectory_path}")

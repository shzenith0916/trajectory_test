from scipy import stats
import math
import pandas as pd


def find_c3(neck_bboxs, y_coords):
    """ 목뼈 좌표틀에서 c3 좌표 찾기

    Args:
        neck_bboxs (list): 목뼈 바운딩 박스 정보 리스트
        y_coords (list): 목뼈 y 좌표 리스트

    Returns:
        c3_bbox (list): C3 목뼈 바운딩 박스 [x_min, y_min, x_max, y_max]
    """
    # y 좌표 값 기준으로 정렬
    sorted_y = sorted(y_coords)

    # c1은 학습하지 않았으므로, detect된 bbox의 두번째가 c3라고 가정하고 실행
    c3_approx = sorted_y[1]
    c3_approx_idx = y_coords.index(c3_approx)
    # 가끔 c2가 detected 안되는 경우도 있음. 탐지 실패 염두.

    # C3 목뼈 바운딩 박스와 y 좌표 선택
    c3_bbox = neck_bboxs[c3_approx_idx]

    return c3_bbox


def movement_up_left(intercept_pt, hb_center):
    """
    설골 이동을 척추 중심선 기준으로 왼쪽 위 방향으로 판정
    - 전방(Anterior) 이동: 왼쪽
    - 상방(Superior) 이동: 위쪽
    """

    if intercept_pt and hb_center:

        # 전방 이동 계산: class1(hb) x coord - intercept x coord (수평거리)
        anterior_displacement = hb_center[0] - intercept_pt[0]
        # 양수: 전방(앞으로), 음수: 후방(뒤로)

        # 상방 이동 계산: intercept y coord - class1(hb) y coord (수직거리)
        superior_displacement = intercept_pt[1] - hb_center[1]
        # 양수: 상방(위로), 음수: 하방(아래로)

        # 총 이동 거리 (Euclidean Distance)
        total_displacement = math.dist(hb_center, intercept_pt)

        # 이동 각도
        move_angle = math.degrees(math.atan2(
            superior_displacement, anterior_displacement))

        print(f"Anterior Displacement: {anterior_displacement:.1f} pixels")
        print(f"Superior Displacement: {superior_displacement:.1f} pixels")
        print(f"Total Displacement: {total_displacement:.1f} pixels")
        print(f"Movement Angle: {move_angle:.1f} degrees")

    df = pd.DataFrame({'superior_displacement': [superior_displacement],
                       'anterior_displacement': [anterior_displacement],
                       'total_displacement': [total_displacement],
                       'move_angle': [move_angle]})
    
    return df

# # 각 환자의 VFSS 비디오에서 ABCD 포인트 추출 후
# # 정상군 (n=30)
# normal_patients = {
#     'anterior_displacement': [42, 45, 38, ...],  # pixels
#     'superior_displacement': [28, 32, 25, ...],
#     'c3_normalized': [85, 92, 78, ...]  # % of C3 width
# }

# # 연하장애군 (n=30)
# dysphagia_patients = {
#     'anterior_displacement': [22, 18, 25, ...],  # 감소!
#     'superior_displacement': [15, 12, 18, ...],  # 감소!
#     'c3_normalized': [45, 38, 52, ...]  # 감소!
# }

# # 통계 분석
# t_stat, p_value = stats.ttest_ind(
#     normal_patients['c3_normalized'],
#     dysphagia_patients['c3_normalized']
# )
# print(f"p-value: {p_value}")  # 예: 0.001 (유의미한 차이!)

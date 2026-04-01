import argparse
import sys
import os

# 상위 디렉토리를 Python 경로에 추가 (import 전에 실행되어야 함)
current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(current_dir)

# 두 경로 모두 추가 (더 안전함)
if parent_dir not in sys.path:
    sys.path.insert(0, parent_dir)
if current_dir not in sys.path:
    sys.path.insert(0, current_dir)

# 디버깅: 경로 확인 (필요시 주석 해제)
# print(f"Current dir: {current_dir}")
# print(f"Parent dir: {parent_dir}")
# print(f"sys.path: {sys.path[:3]}")

# 경로 설정 후에 import (자동 포맷터가 위로 올리는 것을 방지하기 위해 주석으로 표시)
# noqa: E402 주석을 추가해 linter가 import 순서를 변경하지 않도록 함.

import ultralytics  # noqa: E402
from ultralytics import YOLO  # noqa: E402


def run(weights, source):
    """
    기본 객체 탐지 실행 - 바운딩 박스와 라벨만 표시
    """
    # 경로 정규화 (백슬래시 이스케이프 문제 해결)
    weights = os.path.normpath(weights)
    source = os.path.normpath(source)

    # 파일 존재 확인
    if not os.path.exists(weights):
        raise FileNotFoundError(f"모델 파일을 찾을 수 없습니다: {weights}")
    if not os.path.exists(source):
        raise FileNotFoundError(f"소스 파일을 찾을 수 없습니다: {source}")

    print(f"모델 경로: {weights}")
    print(f"소스 경로: {source}")

    # 모델 로드
    model = YOLO(weights)

    # 예측 실행 (기본 설정으로 바운딩 박스와 라벨이 그려진 결과 반환)
    results = model.predict(source, save=True, show=False)

    print("Detection completed!")
    print(f"Results saved to: runs/detect/")


def parse_opt():
    """명령줄 인자 파싱"""
    parser = argparse.ArgumentParser()
    parser.add_argument('--weights', type=str,
                        default='yolov8n.pt', help='모델 경로')
    parser.add_argument('--source', type=str, required=True,
                        help='이미지 또는 비디오 소스 경로')
    opt = parser.parse_args()
    return opt


if __name__ == '__main__':
    opt = parse_opt()
    run(weights=opt.weights, source=opt.source)

'''hnc best 모델 -> C:/Users/USER/Documents/AKAS관련/AKAS1.0_YOLO연구용/ultralytics/runs_HNC_yolo8/detect/yolo11_train1\weights\best.pt
'''

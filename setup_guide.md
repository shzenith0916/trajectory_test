# 환경 설정 가이드

## 1. Python 의존성 설치

```bash
pip install -r requirements.txt
```

## 2. YOLOv5 설치

```bash
git clone https://github.com/ultralytics/yolov5.git
cd yolov5
pip install -r requirements.txt
cd ..
```

## 3. 패치 적용

YOLOv5 원본 파일에 커스텀 수정사항을 적용합니다.

### detect.py 패치 (CSV 출력 기능 추가)

```bash
cd yolov5
git apply ../patches/yolov5_detect.patch
cd ..
```

### utils/plots.py 패치 (plot_single_image 함수 추가)

```bash
cd yolov5
git apply ../patches/yolov5_plots.patch
cd ..
```

## 4. 디렉토리 구조

```
ultralytics_custom/     # ultralytics 기반 커스텀 스크립트 (추론, 학습 등)
ultralytics_notebooks/  # ultralytics 학습/EDA 노트북
yolov5_custom/          # yolov5 기반 커스텀 감지 스크립트
data_configs/           # 데이터셋 YAML 설정 파일
patches/                # 원본 소스 수정 패치
preprocessing/          # 데이터 전처리 스크립트
```

## 5. 데이터셋 설정

`data_configs/` 내 YAML 파일의 경로를 로컬 데이터셋 위치에 맞게 수정하세요.

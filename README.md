## 📖 Table of Contents

- [📖 Table of Contents](#-table-of-contents)
- [🎯 Introduction](#-introduction)
- [✨ Features](#-features)
- [📂 Project Structure](#-project-structure)
- [🛠 Installation](#-installation)
- [🚀 Usage](#-usage)
  - [(Main) 2단계 추론 파이프라인 — `inference_run.py`](#main-2단계-추론-파이프라인--inference_runpy)
  - [단일 모델 궤적 탐지 — `custom_detect.py`](#단일-모델-궤적-탐지--custom_detectpy)
  - [모델 학습](#모델-학습)
  - [(Legacy) YOLOv5](#legacy-yolov5)
- [📄 License](#-license)

---

## 🎯 Introduction

본 프로젝트는 연하장애(삼킴 장애) 진단의 정확성과 효율성을 높이기 위한 **학술 연구(논문) 목적의 과제**입니다.

연구 초기에는 YOLOv5 아키텍처를 기반으로 VFSS 영상 분석의 가능성을 탐색했습니다. 현재는 더 발전된 모델인 YOLOv8, YOLOv11 등을 `Ultralytics` 프레임워크를 활용하여 중점적으로 연구하고 있습니다.

다만, 현재 주로 사용되는 Ultralytics의 YOLO 모델은 **AGPL-3.0 라이선스** 하에 있어 상업적 이용에 제약이 있습니다. 따라서, 본 연구의 최종 목표는 분석 성능을 유지하면서 **MIT 라이선스 등 허용적인 라이선스를 가진 탐지 모델로 대체**하여, 라이선스 제약 없이 자유롭게 활용 가능한 진단 보조 도구를 완성하는 것입니다.

This project is an **academic research initiative (for thesis/paper publication)** aimed at improving the accuracy and efficiency of dysphagia diagnosis. Initial research was conducted based on the YOLOv5 architecture. Currently, the main R&D efforts are focused on more advanced models like YOLOv8 and YOLOv11 within the Ultralytics framework.

However, the primary models from Ultralytics used in this project are under the **AGPL-3.0 license**, which imposes restrictions on commercial use. Therefore, the ultimate goal of this research is to replace the current models with a detection model under a permissive license (such as MIT) while maintaining analytical performance, in order to create a freely usable diagnostic aid.

## ✨ Features

- **Multi-Model Object Detection**: `YOLOv5`, `YOLOv8`, `YOLOv11` 등 다양한 YOLO 아키텍처를 활용하여 VFSS 영상에서 설골(Hyoid Bone)과 경추(Neck Bone)를 탐지합니다.
- **Two-Stage Detection Pipeline**: 1단계에서 설골/경추를 탐지한 뒤, 2단계에서 설골 주변을 크롭하여 bolus, epiglottis 등 세밀한 객체를 추가 탐지합니다.
- **Trajectory Analysis**: 탐지된 설골의 움직임을 프레임별로 추적하고 궤적 데이터를 생성합니다.
- **Movement Correction**: 경추의 움직임을 기준으로 설골의 움직임을 보정하여 머리 움직임의 영향을 최소화합니다.
- **Quantitative Analysis**: 분석된 궤적 데이터를 CSV 파일로 저장하여 정량적인 비교 분석을 지원합니다.
- **Stratified K-Fold Training**: Monte Carlo 기반 Repeated Stratified K-Fold 교차 검증을 통한 체계적 모델 학습을 지원합니다.
- **Visualization**: 원본 영상에 분석 결과를 시각화하고, 설골의 전체 이동 경로를 별도의 궤적 이미지로 생성합니다.

## 📂 Project Structure

```
trajectory_test연구용/
│
├── ultralytics/                  # (Main) YOLOv8, v11 등 최신 모델 연구
│   ├── custom_detect.py          # 설골-경추 궤적 탐지 (단일 모델)
│   ├── inference_run.py          # 2단계 VFSS 추론 파이프라인
│   ├── two_stage_utils.py        # 2단계 추론 유틸리티
│   ├── detect.py                 # 기본 탐지 스크립트
│   ├── train.py                  # 모델 학습 스크립트
│   ├── stratified_kfold_train.py # Stratified K-Fold 교차 검증 학습
│   ├── plot_trajectories.py      # 궤적 CSV → 시각화 플롯
│   ├── extract_label3.py         # 라벨 데이터 추출
│   └── relabel_label3.py         # 라벨 데이터 재분류
│
├── yolov5/                       # (Legacy) 초기 YOLOv5 연구 아카이브
│   ├── detect.py
│   └── detect_refactor.py
│
├── preprocessing/                # 데이터 전처리
│   ├── label_parse/              # 라벨 파싱, bbox 변환, 정리
│   └── until_extract_img/        # 영상 트리밍, 프레임 추출, 파일 정리
│
├── references/                   # 참고 스크립트 및 알고리즘 문서
│
└── README.md
```

## 🛠 Installation

1.  **Clone the repository:**
    ```bash
    git clone https://github.com/shzenith0916/trajectory_test.git
    cd AKAS-01
    ```

2.  **Create a virtual environment (recommended):**
    ```bash
    python -m venv venv
    source venv/bin/activate  # On Windows, use `venv\Scripts\activate`
    ```

3.  **Install dependencies:**
    각 YOLO 버전 폴더에 있는 `requirements.txt` 파일을 사용하여 필요한 라이브러리를 설치합니다. 메인인 `ultralytics`부터 설치하는 것을 권장합니다.
    ```bash
    # (Main) For YOLOv8 and later versions
    pip install -r ultralytics/requirements.txt

    # (Legacy) For YOLOv5
    pip install -r yolov5/requirements.txt
    ```

## 🚀 Usage

### (Main) 2단계 추론 파이프라인 — `inference_run.py`

Stage 1에서 설골/경추를 탐지하고, Stage 2에서 설골 주변을 크롭하여 세밀한 객체를 추가 탐지합니다.

```bash
python ultralytics/inference_run.py \
  --weights-stage1 /path/to/stage1_model.pt \
  --weights-stage2 /path/to/stage2_model.pt \
  --source /path/to/video.avi \
  --conf-stage1 0.6 \
  --conf-stage2 0.3 \
  --visualize
```

자세한 인자 설명은 [ultralytics/README.md](ultralytics/README.md)를 참고하세요.

### 단일 모델 궤적 탐지 — `custom_detect.py`

2단계 없이 설골-경추 궤적만 확인하고 싶을 때 사용합니다.

```bash
python ultralytics/custom_detect.py \
  --weights /path/to/model.pt \
  --source /path/to/video.avi
```

### 모델 학습

```bash
# 단일 학습
python ultralytics/train.py

# Stratified K-Fold 교차 검증
python ultralytics/stratified_kfold_train.py
```

### (Legacy) YOLOv5

```bash
python yolov5/detect_refactor.py --weights /path/to/yolov5_model.pt --source /path/to/video.avi
```

실행 결과는 각 프로젝트 폴더의 `runs/detect/` 내에 저장됩니다.

## 📄 License

This project is licensed under the AGPL-3.0 License. See the `LICENSE` file for details.


# AI-based Kinematic Analysis Software for Dysphagia Diagnosis

AKAS (Automatic Kinematic Analysis Software)는 비디오 투시 삼킴 검사(VFSS) 영상을 AI 기술로 분석하여, 연하장애(삼킴 장애) 진단을 보조하는 자동 운동학 분석 소프트웨어입니다.

This project, AKAS-01, is an automated analysis program for video fluoroscopic swallowing studies (VFSS). It utilizes AI technology to enable the automated, quantitative analysis of VFSS, aiming to verify the accuracy of dysphagia diagnosis and develop a systematic diagnostic framework.

---

## 📖 Table of Contents

- [AI-based Kinematic Analysis Software for Dysphagia Diagnosis](#ai-based-kinematic-analysis-software-for-dysphagia-diagnosis)
  - [📖 Table of Contents](#-table-of-contents)
  - [🎯 Introduction](#-introduction)
  - [✨ Features](#-features)
  - [📂 Project Structure](#-project-structure)
  - [🛠 Installation](#-installation)
  - [🚀 Usage](#-usage)
      - [(Main) YOLOv8 / YOLOv11](#main-yolov8--yolov11)
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
- **Trajectory Analysis**: 탐지된 설골의 움직임을 프레임별로 추적하고 궤적 데이터를 생성합니다.
- **Movement Correction**: 경추의 움직임을 기준으로 설골의 움직임을 보정하여 머리 움직임의 영향을 최소화합니다.
- **Quantitative Analysis**: 분석된 궤적 데이터를 CSV 파일로 저장하여 정량적인 비교 분석을 지원합니다.
- **Visualization**: 원본 영상에 분석 결과를 시각화하고, 설골의 전체 이동 경로를 별도의 궤적 이미지로 생성합니다.

## 📂 Project Structure

본 프로젝트는 서로 다른 기반을 가진 YOLO 버전을 실험하기 위해, 각 버전의 공식 프로젝트 폴더 구조를 유지하고 있습니다.

```
AKAS/
├── ultralytics/      # (Main) YOLOv8, v11 등 최신 모델 연구 개발을 위한 메인 폴더
│   ├── engine/
│   ├── models/
│   └── custom_detect.py
│
├── yolov5/           # (Legacy) 초기 YOLOv5 연구 아카이브
│   ├── data/
│   ├── models/
│   ├── notebooks/    # Jupyter Notebook을 활용한 학습 및 테스트 기록
│   └── detect_refactor.py
│
└── README.md
```

## 🛠 Installation

1.  **Clone the repository:**
    ```bash
    git clone http://192.168.200.5:3000/pangyo_rnd/AKAS.git
    cd AKAS
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

각 YOLO 버전에 맞는 탐지 스크립트를 사용하여 분석을 실행합니다.

#### (Main) YOLOv8 / YOLOv11

`ultralytics` 프레임워크의 `predict` 모드를 사용하거나 커스텀 스크립트를 실행합니다.

```bash
# Using the standard 'yolo' command
yolo predict model=/path/to/your/yolov8_model.pt source=/path/to/your/video.avi

# Using a custom script
python ultralytics/custom_weights /path/to/your/yolov8_model.pt --source=/path/to/your/video.avi
```

#### (Legacy) YOLOv5

`yolov5/detect_refactor.py` 스크립트를 사용합니다.

```bash
python yolov5/detect_refactor.py --weights /path/to/your/yolov5_model.pt --source /path/to/your/video.avi
```

실행 결과는 각 프로젝트 폴더(`ultralytics/runs/detect/`, `yolov5/runs/detect/`) 내에 저장됩니다.

## 📄 License

This project is licensed under the AGPL-3.0 License. See the `LICENSE` file for details.






# Inference Run - 2단계 VFSS 추론 파이프라인

## 개요

`inference_run.py`는 VFSS(Videofluoroscopic Swallowing Study) 영상에서 2단계 객체 탐지를 수행합니다.

- **Stage 1**: 전체 프레임에서 설골(hyoid bone)과 목뼈(neck bone) 탐지
- **Stage 2**: 설골 주변 영역을 크롭하여 세밀한 객체(bolus, epiglottis 등) 탐지

---

## 인자 (Arguments)

### 필수 인자

| 인자 | 타입 | 설명 |
|------|------|------|
| `--source` | str | 입력 비디오/이미지 경로 |

### 선택 인자

| 인자 | 타입 | 기본값 | 설명 |
|------|------|--------|------|
| `--weights-stage1` | str | `yolov8n.pt` | Stage 1 모델 경로 (설골 & 목뼈 탐지) |
| `--weights-stage2` | str | `yolov8n.pt` | Stage 2 모델 경로 (bolus 등 세밀 탐지) |
| `--class1-id` | int | `0` | 설골(hyoid bone) 클래스 ID |
| `--class2-id` | int | `1` | 목뼈(neck bone) 클래스 ID |
| `--class3-id` | int | `0` | Stage 2에서 탐지할 클래스 ID |
| `--conf-stage1` | float | `0.5` | Stage 1 confidence threshold |
| `--conf-stage2` | float | `0.25` | Stage 2 confidence threshold |
| `--visualize` | flag | `False` | 시각화 활성화 (플래그만 붙이면 True) |

---

## 실행 예시

### 1. Stage 1 + Stage 2 모두 실행

```powershell
python inference_run.py \
  --weights-stage1 .\runs_HNC_yolo8\detect\yolo8_train4\weights\best.pt \
  --weights-stage2 .\runs_food_yolo8\detect\train\weights\best.pt \
  --source "F:\vfss_child\영상파일.mp4" \
  --conf-stage1 0.6 \
  --conf-stage2 0.3 \
  --visualize
```

### 2. Stage 1만 실행 (Stage 2 건너뛰기)

Stage 2를 사용하지 않으려면 `--weights-stage2`를 `--weights-stage1`과 **동일한 경로**로 지정합니다.

```powershell
python inference_run.py \
  --weights-stage1 .\runs_HNC_yolo8\detect\yolo8_train4\weights\best.pt \
  --weights-stage2 .\runs_HNC_yolo8\detect\yolo8_train4\weights\best.pt \
  --source "F:\vfss_child\영상파일.mp4" \
  --conf-stage1 0.6 \
  --visualize
```

> **주의**: `--weights-stage2`를 생략하면 기본값(`yolov8n.pt`)이 적용되어 Stage 1 경로와 달라지므로, Stage 2가 의도치 않게 실행됩니다. 반드시 Stage 1과 동일한 경로를 명시해야 Stage 2를 건너뜁니다.

---

## 출력 결과

결과는 `runs/detect/{파일명}_two_stage/` 디렉토리에 저장됩니다.

| 파일 | 설명 |
|------|------|
| `*_two_stage.avi` | 탐지 결과가 그려진 비디오 |
| `*_VideoInfo.csv` | 비디오 메타 정보 (해상도, FPS, 길이 등) |
| `*_moving_points.csv` | 프레임별 설골/보정 설골 좌표 및 속도 |
| `*_abcd_points.csv` | ABCD 포인트 좌표 및 구간별 속도 |
| `*_stage2_detection.csv` | Stage 2 탐지 결과 (사용 시) |

---

## 참고: custom_detect.py

Stage 2 없이 설골-목뼈 궤적만 확인하고 싶다면 `custom_detect.py`를 사용할 수 있습니다.

```powershell
python custom_detect.py \
  --weights .\runs_HNC_yolo8\detect\yolo8_train4\weights\best.pt \
  --source "F:\vfss_child\영상파일.mp4"
```

| | `custom_detect.py` | `inference_run.py` |
|--|--------------------|--------------------|
| Stage 2 (크롭 탐지) | X | O |
| ABCD 포인트 계산 | X | O |
| 속도 계산 | X | O |
| 외부 유틸 의존 | 없음 (자체 함수) | `two_stage_utils` 필요 |

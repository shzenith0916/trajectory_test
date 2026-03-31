from ultralytics import YOLO
import torch


def main():
    # GPU 사용 가능 여부 확인
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"Using device: {device}")
    if torch.cuda.is_available():
        print(f"GPU: {torch.cuda.get_device_name(0)}")
        print(f"CUDA Version: {torch.version.cuda}")

    # 모델 로드
    model = YOLO('yolov8s.pt')  # 또는 yolov8n.pt, yolov8m.pt 등

    # 학습 실행
    results = model.train(
        data='data/all/custom_data_all.yaml',
        epochs=100,
        imgsz=640,
        batch=16,
        device=0,  # GPU 0번 사용 (CPU는 device='cpu' 또는 device=None)
        workers=8,  # 데이터로더 워커 수 증가
        name='all_data_train'
    )


if __name__ == "__main__":
    main()

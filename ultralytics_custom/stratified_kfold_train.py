"""
Stratified K-Fold Cross-Validation for YOLO Object Detection
=============================================================

Monte Carlo 기반 Repeated Stratified K-Fold 구현
- 다중 레이블 객체 탐지에 최적화
- 클래스 비율 기반 층화 샘플링
- 편향 제거 및 신뢰구간 추정

Author: AI Assistant
Usage: python stratified_kfold_train.py
"""

import os
import sys
import shutil
import yaml
import json
import numpy as np
import pandas as pd
from pathlib import Path
from collections import Counter, defaultdict
from datetime import datetime
from typing import List, Tuple, Dict, Optional

# Stratified K-Fold를 위한 라이브러리
try:
    from iterstrat.ml_stratifiers import MultilabelStratifiedKFold
    USE_MULTILABEL = True
except ImportError:
    from sklearn.model_selection import StratifiedKFold
    USE_MULTILABEL = False
    print("Warning: iterative-stratification not installed. Using standard StratifiedKFold.")
    print("For better multi-label stratification, run: pip install iterative-stratification")

from ultralytics import YOLO
import torch


# ============================================================================
# 설정
# ============================================================================
class Config:
    """실험 설정"""
    # 데이터셋 경로
    BASE_DIR = Path(__file__).parent
    DATA_DIR = BASE_DIR / "data" / "all"
    TRAIN_IMAGES_DIR = DATA_DIR / "train" / "images"
    TRAIN_LABELS_DIR = DATA_DIR / "train" / "labels"
    VALID_IMAGES_DIR = DATA_DIR / "valid" / "images"
    VALID_LABELS_DIR = DATA_DIR / "valid" / "labels"

    # K-Fold 설정
    N_SPLITS = 5          # K-Fold의 K
    N_REPEATS = 3         # 반복 횟수 (Monte Carlo)
    RANDOM_SEEDS = [42, 123, 456]  # 각 반복의 시드

    # 클래스 정보
    NUM_CLASSES = 4
    CLASS_NAMES = ['hyoid_bone', 'neck_bone', 'food_locate', 'food']

    # 학습 설정
    MODEL_NAME = 'yolov8s.pt'
    EPOCHS = 100
    BATCH_SIZE = 16
    IMG_SIZE = 640
    DEVICE = 0  # GPU
    WORKERS = 8
    PATIENCE = 20  # Early stopping

    # 출력 디렉토리
    OUTPUT_DIR = BASE_DIR / "runs_kfold"
    TEMP_DATA_DIR = BASE_DIR / "temp_kfold_data"


# ============================================================================
# 데이터 분석 및 층화 함수
# ============================================================================
def parse_label_file(label_path: Path) -> Dict[int, int]:
    """
    라벨 파일을 파싱하여 클래스별 객체 수 반환

    Returns:
        {class_id: count} 형태의 딕셔너리
    """
    class_counts = defaultdict(int)

    if not label_path.exists():
        return dict(class_counts)

    with open(label_path, 'r') as f:
        for line in f:
            parts = line.strip().split()
            if parts:
                class_id = int(parts[0])
                class_counts[class_id] += 1

    return dict(class_counts)


def analyze_dataset(images_dir: Path, labels_dir: Path) -> pd.DataFrame:
    """
    데이터셋 분석 및 이미지별 클래스 분포 계산

    Returns:
        DataFrame with columns: [image_path, label_path, class_0, class_1, ..., total_objects]
    """
    data = []

    # 이미지 확장자
    image_extensions = {'.jpg', '.jpeg', '.png', '.bmp', '.tiff'}

    for img_path in images_dir.iterdir():
        if img_path.suffix.lower() not in image_extensions:
            continue

        # 대응하는 라벨 파일 찾기
        label_path = labels_dir / (img_path.stem + '.txt')
        class_counts = parse_label_file(label_path)

        row = {
            'image_path': str(img_path),
            'label_path': str(label_path),
            'image_name': img_path.name
        }

        # 각 클래스별 객체 수
        total = 0
        for cls_id in range(Config.NUM_CLASSES):
            count = class_counts.get(cls_id, 0)
            row[f'class_{cls_id}'] = count
            total += count

        row['total_objects'] = total
        data.append(row)

    return pd.DataFrame(data)


def create_multilabel_matrix(df: pd.DataFrame) -> np.ndarray:
    """
    다중 레이블 행렬 생성 (iterstrat용)

    각 이미지에 대해 [class_0_count, class_1_count, ...] 형태의 행렬 생성
    층화를 위해 이진화 또는 구간화 적용
    """
    class_columns = [f'class_{i}' for i in range(Config.NUM_CLASSES)]

    # 방법 1: 이진화 (해당 클래스 존재 여부)
    # binary_matrix = (df[class_columns].values > 0).astype(int)

    # 방법 2: 구간화 (객체 수를 구간으로 나눔) - 더 세밀한 층화
    def binning(x):
        if x == 0:
            return 0
        elif x == 1:
            return 1
        elif x <= 3:
            return 2
        elif x <= 5:
            return 3
        else:
            return 4

    binned_matrix = df[class_columns].applymap(binning).values

    return binned_matrix


def create_composite_label(df: pd.DataFrame) -> np.ndarray:
    """
    단일 레이블 생성 (sklearn StratifiedKFold용)
    다중 클래스 정보를 하나의 복합 레이블로 인코딩
    """
    class_columns = [f'class_{i}' for i in range(Config.NUM_CLASSES)]

    # 각 클래스 존재 여부를 비트로 인코딩
    labels = []
    for _, row in df.iterrows():
        label = 0
        for i, col in enumerate(class_columns):
            if row[col] > 0:
                label |= (1 << i)
        labels.append(label)

    return np.array(labels)


# ============================================================================
# K-Fold 분할 및 데이터 준비
# ============================================================================
def get_kfold_splits(df: pd.DataFrame, n_splits: int, random_seed: int) -> List[Tuple[np.ndarray, np.ndarray]]:
    """
    Stratified K-Fold 분할 수행

    Returns:
        [(train_indices, val_indices), ...] 리스트
    """
    X = np.arange(len(df))

    if USE_MULTILABEL:
        # 다중 레이블 층화
        y = create_multilabel_matrix(df)
        kfold = MultilabelStratifiedKFold(n_splits=n_splits, shuffle=True, random_state=random_seed)
    else:
        # 단일 레이블 층화
        y = create_composite_label(df)
        kfold = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=random_seed)

    splits = list(kfold.split(X, y))
    return splits


def prepare_fold_data(df: pd.DataFrame, train_idx: np.ndarray, val_idx: np.ndarray,
                      fold_dir: Path) -> Tuple[Path, Path]:
    """
    특정 폴드의 학습/검증 데이터 준비 (심볼릭 링크 또는 복사)

    Returns:
        (train_images_dir, val_images_dir)
    """
    # 폴드 디렉토리 구조 생성
    train_img_dir = fold_dir / "train" / "images"
    train_lbl_dir = fold_dir / "train" / "labels"
    val_img_dir = fold_dir / "valid" / "images"
    val_lbl_dir = fold_dir / "valid" / "labels"

    for d in [train_img_dir, train_lbl_dir, val_img_dir, val_lbl_dir]:
        d.mkdir(parents=True, exist_ok=True)

    # 학습 데이터 링크/복사
    train_df = df.iloc[train_idx]
    for _, row in train_df.iterrows():
        src_img = Path(row['image_path'])
        src_lbl = Path(row['label_path'])

        dst_img = train_img_dir / src_img.name
        dst_lbl = train_lbl_dir / src_lbl.name

        # Windows에서는 심볼릭 링크 대신 하드링크 또는 복사 사용
        try:
            if not dst_img.exists():
                os.link(str(src_img), str(dst_img))
            if src_lbl.exists() and not dst_lbl.exists():
                os.link(str(src_lbl), str(dst_lbl))
        except OSError:
            # 하드링크 실패 시 복사
            if not dst_img.exists():
                shutil.copy2(src_img, dst_img)
            if src_lbl.exists() and not dst_lbl.exists():
                shutil.copy2(src_lbl, dst_lbl)

    # 검증 데이터 링크/복사
    val_df = df.iloc[val_idx]
    for _, row in val_df.iterrows():
        src_img = Path(row['image_path'])
        src_lbl = Path(row['label_path'])

        dst_img = val_img_dir / src_img.name
        dst_lbl = val_lbl_dir / src_lbl.name

        try:
            if not dst_img.exists():
                os.link(str(src_img), str(dst_img))
            if src_lbl.exists() and not dst_lbl.exists():
                os.link(str(src_lbl), str(dst_lbl))
        except OSError:
            if not dst_img.exists():
                shutil.copy2(src_img, dst_img)
            if src_lbl.exists() and not dst_lbl.exists():
                shutil.copy2(src_lbl, dst_lbl)

    return train_img_dir.parent, val_img_dir.parent


def create_fold_yaml(fold_dir: Path, yaml_path: Path) -> Path:
    """
    폴드별 데이터셋 YAML 파일 생성
    """
    yaml_content = {
        'path': str(fold_dir),
        'train': 'train/images',
        'val': 'valid/images',
        'nc': Config.NUM_CLASSES,
        'names': {i: name for i, name in enumerate(Config.CLASS_NAMES)}
    }

    with open(yaml_path, 'w', encoding='utf-8') as f:
        yaml.dump(yaml_content, f, allow_unicode=True, default_flow_style=False)

    return yaml_path


# ============================================================================
# 학습 및 평가
# ============================================================================
def train_fold(fold_yaml: Path, repeat_idx: int, fold_idx: int) -> Dict:
    """
    단일 폴드 학습 수행

    Returns:
        학습 결과 딕셔너리 (메트릭 포함)
    """
    print(f"\n{'='*60}")
    print(f"Training Repeat {repeat_idx + 1}, Fold {fold_idx + 1}")
    print(f"{'='*60}")

    # 모델 로드
    model = YOLO(Config.MODEL_NAME)

    # 실험 이름
    exp_name = f"repeat{repeat_idx + 1}_fold{fold_idx + 1}"

    # 학습 실행
    results = model.train(
        data=str(fold_yaml),
        epochs=Config.EPOCHS,
        imgsz=Config.IMG_SIZE,
        batch=Config.BATCH_SIZE,
        device=Config.DEVICE,
        workers=Config.WORKERS,
        patience=Config.PATIENCE,
        project=str(Config.OUTPUT_DIR),
        name=exp_name,
        exist_ok=True,
        verbose=True
    )

    # 최종 메트릭 추출
    metrics = {
        'repeat': repeat_idx + 1,
        'fold': fold_idx + 1,
        'seed': Config.RANDOM_SEEDS[repeat_idx],
        'mAP50': float(results.results_dict.get('metrics/mAP50(B)', 0)),
        'mAP50-95': float(results.results_dict.get('metrics/mAP50-95(B)', 0)),
        'precision': float(results.results_dict.get('metrics/precision(B)', 0)),
        'recall': float(results.results_dict.get('metrics/recall(B)', 0)),
        'best_epoch': int(results.results_dict.get('epoch', Config.EPOCHS)),
    }

    # 클래스별 AP (가능한 경우)
    for i, name in enumerate(Config.CLASS_NAMES):
        key = f'metrics/mAP50(B)_{name}'
        if key in results.results_dict:
            metrics[f'AP50_{name}'] = float(results.results_dict[key])

    return metrics


def cleanup_fold_data(fold_dir: Path):
    """
    폴드 임시 데이터 정리
    """
    if fold_dir.exists():
        shutil.rmtree(fold_dir)


# ============================================================================
# 결과 분석
# ============================================================================
def analyze_results(all_metrics: List[Dict]) -> Dict:
    """
    전체 결과 통계 분석

    Returns:
        요약 통계 딕셔너리
    """
    df = pd.DataFrame(all_metrics)

    # 주요 메트릭
    main_metrics = ['mAP50', 'mAP50-95', 'precision', 'recall']

    summary = {
        'total_experiments': len(df),
        'n_repeats': Config.N_REPEATS,
        'n_folds': Config.N_SPLITS,
    }

    for metric in main_metrics:
        if metric in df.columns:
            values = df[metric].values
            summary[f'{metric}_mean'] = float(np.mean(values))
            summary[f'{metric}_std'] = float(np.std(values))
            summary[f'{metric}_min'] = float(np.min(values))
            summary[f'{metric}_max'] = float(np.max(values))

            # 95% 신뢰구간 (t-분포 기반)
            n = len(values)
            se = np.std(values, ddof=1) / np.sqrt(n)
            t_critical = 2.145  # df=14 (15-1)에서 95% 신뢰구간
            ci_lower = np.mean(values) - t_critical * se
            ci_upper = np.mean(values) + t_critical * se
            summary[f'{metric}_ci95_lower'] = float(ci_lower)
            summary[f'{metric}_ci95_upper'] = float(ci_upper)

    # 반복별 통계 (Monte Carlo 분산 추정)
    repeat_stats = df.groupby('repeat')[main_metrics].mean()
    summary['repeat_variance'] = {
        metric: float(repeat_stats[metric].var()) for metric in main_metrics if metric in repeat_stats.columns
    }

    return summary


def print_results_summary(summary: Dict, all_metrics: List[Dict]):
    """
    결과 요약 출력
    """
    print("\n" + "="*70)
    print("STRATIFIED K-FOLD CROSS-VALIDATION RESULTS")
    print("="*70)

    print(f"\n[실험 설정]")
    print(f"   - K-Fold: {Config.N_SPLITS}-Fold x {Config.N_REPEATS} Repeats = {summary['total_experiments']} 실험")
    print(f"   - Random Seeds: {Config.RANDOM_SEEDS}")
    print(f"   - Model: {Config.MODEL_NAME}")
    print(f"   - Epochs: {Config.EPOCHS}")

    print(f"\n[주요 메트릭 (Mean +/- Std)]")
    print("-" * 50)

    for metric in ['mAP50', 'mAP50-95', 'precision', 'recall']:
        mean_key = f'{metric}_mean'
        std_key = f'{metric}_std'
        ci_lower = f'{metric}_ci95_lower'
        ci_upper = f'{metric}_ci95_upper'

        if mean_key in summary:
            print(f"   {metric:12}: {summary[mean_key]:.4f} +/- {summary[std_key]:.4f}")
            print(f"   {'':12}  95% CI: [{summary[ci_lower]:.4f}, {summary[ci_upper]:.4f}]")

    # 폴드별 결과 테이블
    print(f"\n[폴드별 상세 결과]")
    print("-" * 70)

    df = pd.DataFrame(all_metrics)
    print(df[['repeat', 'fold', 'mAP50', 'mAP50-95', 'precision', 'recall']].to_string(index=False))

    # Monte Carlo 분산 (반복 간 변동)
    print(f"\n[Monte Carlo 분산 (반복 간 변동)]")
    for metric, var in summary.get('repeat_variance', {}).items():
        print(f"   {metric}: {var:.6f}")

    print("\n" + "="*70)


def save_results(all_metrics: List[Dict], summary: Dict):
    """
    결과를 파일로 저장
    """
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    results_dir = Config.OUTPUT_DIR / "results"
    results_dir.mkdir(parents=True, exist_ok=True)

    # 상세 결과 CSV
    df = pd.DataFrame(all_metrics)
    csv_path = results_dir / f"kfold_results_{timestamp}.csv"
    df.to_csv(csv_path, index=False)
    print(f"\n[저장] 상세 결과: {csv_path}")

    # 요약 JSON
    json_path = results_dir / f"kfold_summary_{timestamp}.json"
    with open(json_path, 'w', encoding='utf-8') as f:
        json.dump(summary, f, indent=2, ensure_ascii=False)
    print(f"[저장] 요약: {json_path}")

    # 보고서 텍스트
    report_path = results_dir / f"kfold_report_{timestamp}.txt"
    with open(report_path, 'w', encoding='utf-8') as f:
        f.write("STRATIFIED K-FOLD CROSS-VALIDATION REPORT\n")
        f.write("=" * 50 + "\n\n")

        f.write(f"Date: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
        f.write(f"Configuration:\n")
        f.write(f"  - K-Fold: {Config.N_SPLITS}-Fold x {Config.N_REPEATS} Repeats\n")
        f.write(f"  - Random Seeds: {Config.RANDOM_SEEDS}\n")
        f.write(f"  - Model: {Config.MODEL_NAME}\n")
        f.write(f"  - Epochs: {Config.EPOCHS}\n\n")

        f.write("Results Summary:\n")
        for metric in ['mAP50', 'mAP50-95', 'precision', 'recall']:
            mean_key = f'{metric}_mean'
            std_key = f'{metric}_std'
            if mean_key in summary:
                f.write(f"  {metric}: {summary[mean_key]:.4f} +/- {summary[std_key]:.4f}\n")

        f.write("\n\nDetailed Results:\n")
        f.write(df.to_string(index=False))

    print(f"[저장] 보고서: {report_path}")


# ============================================================================
# 메인 실행
# ============================================================================
def main():
    """
    메인 실행 함수
    """
    print("\n" + "="*70)
    print("STRATIFIED K-FOLD CROSS-VALIDATION FOR YOLO")
    print("   Monte Carlo Repeated K-Fold (Bias-Free Evaluation)")
    print("="*70)

    # GPU 확인
    if torch.cuda.is_available():
        print(f"\n[OK] GPU 사용: {torch.cuda.get_device_name(0)}")
    else:
        print("\n[WARNING] GPU 없음, CPU 사용")
        Config.DEVICE = 'cpu'

    # 기존 train + valid 데이터 통합 분석
    print("\n[INFO] 데이터셋 분석 중...")

    # train 데이터 분석
    train_df = analyze_dataset(Config.TRAIN_IMAGES_DIR, Config.TRAIN_LABELS_DIR)
    print(f"   Train 데이터: {len(train_df)} 이미지")

    # valid 데이터 분석 (K-Fold에서는 train+valid를 통합하여 재분할)
    valid_df = analyze_dataset(Config.VALID_IMAGES_DIR, Config.VALID_LABELS_DIR)
    print(f"   Valid 데이터: {len(valid_df)} 이미지")

    # 전체 데이터 통합
    full_df = pd.concat([train_df, valid_df], ignore_index=True)
    print(f"   전체 데이터: {len(full_df)} 이미지")

    # 클래스 분포 출력
    print("\n[INFO] 클래스별 객체 분포:")
    for i, name in enumerate(Config.CLASS_NAMES):
        total = full_df[f'class_{i}'].sum()
        images_with_class = (full_df[f'class_{i}'] > 0).sum()
        print(f"   {name}: {total} 객체 ({images_with_class} 이미지)")

    # 출력 디렉토리 생성
    Config.OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    Config.TEMP_DATA_DIR.mkdir(parents=True, exist_ok=True)

    # 전체 결과 저장
    all_metrics = []

    # Repeated Stratified K-Fold 실행
    for repeat_idx, seed in enumerate(Config.RANDOM_SEEDS):
        print(f"\n{'#'*70}")
        print(f"# REPEAT {repeat_idx + 1}/{Config.N_REPEATS} (Seed: {seed})")
        print(f"{'#'*70}")

        # K-Fold 분할
        splits = get_kfold_splits(full_df, Config.N_SPLITS, seed)

        for fold_idx, (train_idx, val_idx) in enumerate(splits):
            print(f"\n[FOLD] {fold_idx + 1}/{Config.N_SPLITS}")
            print(f"   Train: {len(train_idx)} 이미지, Val: {len(val_idx)} 이미지")

            # 폴드 데이터 준비
            fold_dir = Config.TEMP_DATA_DIR / f"repeat{repeat_idx + 1}_fold{fold_idx + 1}"

            try:
                prepare_fold_data(full_df, train_idx, val_idx, fold_dir)

                # YAML 생성
                fold_yaml = fold_dir / "data.yaml"
                create_fold_yaml(fold_dir, fold_yaml)

                # 학습
                metrics = train_fold(fold_yaml, repeat_idx, fold_idx)
                all_metrics.append(metrics)

                print(f"\n   [OK] Fold {fold_idx + 1} 완료:")
                print(f"      mAP50: {metrics['mAP50']:.4f}")
                print(f"      mAP50-95: {metrics['mAP50-95']:.4f}")

            except Exception as e:
                print(f"   [ERROR] Fold {fold_idx + 1}: {e}")
                import traceback
                traceback.print_exc()

            finally:
                # 임시 데이터 정리 (옵션)
                # cleanup_fold_data(fold_dir)
                pass

    # 결과 분석 및 출력
    if all_metrics:
        summary = analyze_results(all_metrics)
        print_results_summary(summary, all_metrics)
        save_results(all_metrics, summary)
    else:
        print("\n[ERROR] 학습 결과가 없습니다.")

    # 임시 디렉토리 정리
    print(f"\n[INFO] 임시 데이터 디렉토리: {Config.TEMP_DATA_DIR}")
    print("   (필요시 수동으로 삭제하세요)")

    print("\n[OK] 모든 실험 완료!")


if __name__ == "__main__":
    main()

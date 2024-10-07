import os
import glob


def remove_unmatched_labels(label_dir, image_file_list):
    """
    이미지 파일 리스트에 해당하는 레이블 파일만 남기고 나머지는 삭제합니다.
    """
    # 이미지 파일 리스트를 생성
    image_file_names = [os.path.splitext(os.path.basename(file))[
        0] for file in image_file_list]

    # 레이블 디렉토리에서 모든 .txt 파일 찾기
    label_files = glob.glob(os.path.join(label_dir, '*.txt'))

    for label_file in label_files:
        label_name = os.path.splitext(os.path.basename(label_file))[0]

        # 레이블 파일이 이미지 파일 리스트에 없으면 삭제
        if label_name not in image_file_names:
            os.remove(label_file)
            print(f"삭제된 레이블 파일: {label_file}")


if __name__ == '__main__':
    # 작업할 디렉토리 명시
    root_dir = os.path.join(os.getcwd(), 'data')
    print(f"레이블 파일이 있는 디렉토리: {root_dir}")

    # 'train' 디렉토리에서 'labels' 폴더 안의 '.txt' 파일 찾기
    train_image_dir = os.path.join(root_dir, 'train', 'images')
    train_image_files = glob.glob(os.path.join(train_image_dir, '*.jpg'))

    # 'train' 디렉토리에서 레이블 정리 (이미지에 대응되지 않는 레이블 삭제)
    print(f"Train 레이블 파일 정리 중...")
    remove_unmatched_labels(os.path.join(
        root_dir, 'train', 'labels'), train_image_files)

    print("레이블 정리 완료")

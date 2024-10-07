import os
import glob


def remove_unmatched_images(image_dir, label_file_list):
    """
    레이블 파일 리스트에 해당하는 이미지 파일만 남기고 나머지는 삭제합니다.
    """
    # 레이블 파일 리스트를 생성
    label_file_names = [os.path.splitext(os.path.basename(file))[
        0] for file in label_file_list]

    # 이미지 디렉토리에서 모든 .jpg 파일 찾기
    image_files = glob.glob(os.path.join(image_dir, '*.jpg'))

    for image_file in image_files:
        image_name = os.path.splitext(os.path.basename(image_file))[0]

        # 이미지 파일이 레이블 파일 리스트에 없으면 삭제
        if image_name not in label_file_names:
            os.remove(image_file)
            print(f"삭제된 이미지 파일: {image_file}")


if __name__ == '__main__':

    # 작업할 디렉토리 명시
    root_dir = os.path.join(os.getcwd(), 'data')
    print(f"레이블 파일이 있는 디렉토리: {root_dir}")

    # 'train' 디렉토리에서 'labels' 폴더 안의 '.txt' 파일 찾기
    train_label_files = glob.glob(os.path.join(
        root_dir, 'train', 'labels', '*.txt'), recursive=True)
    print(f'Number of train label files: {len(train_label_files)}')

    # 'valid' 디렉토리에서 'labels' 폴더 안의 '.txt' 파일 찾기
    valid_label_files = glob.glob(os.path.join(
        root_dir, 'valid', 'labels', '*.txt'), recursive=True)
    print(f'Number of valid label files: {len(valid_label_files)}')

    # 각 폴더에서 이미지 파일과 레이블 파일의 일치를 확인
    train_image_dir = os.path.join(root_dir, 'train', 'images')
    valid_image_dir = os.path.join(root_dir, 'valid', 'images')

    print(f"Train 이미지 파일 정리 중...")
    remove_unmatched_images(train_image_dir, train_label_files)

    print(f"Valid 이미지 파일 정리 중...")
    remove_unmatched_images(valid_image_dir, valid_label_files)

    print("이미지 정리 완료")

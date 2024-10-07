import os
import glob

def remove_empty_label_files(label_file_list):
    """
    레이블 파일 리스트를 받아 파일이 비어있는 경우 해당 파일을 삭제합니다.
    """
    for file in label_file_list:

        with open(file, 'r') as f:
            lines = f.readlines()

        # 파일이 비어있는 경우 파일 삭제
        if len(lines) == 0:
            os.remove(file)
            print(f"삭제된 파일: {file}")


if __name__ == '__main__':

    # 작업할 디렉토리 명시
    root_dir = os.path.join(os.getcwd(), 'data')
    print(f"레이블 파일이 있는 디렉토리: {root_dir}")

    # 재귀적으로 모든 하위 디렉토리에서 '.txt' 파일 찾기
    files = glob.glob(os.path.join(root_dir, '**', '*.txt'), recursive=True)

    # 파일리스트가 빈 리스트인지 확인
    if len(files) == 0:
        print("file list is empty! 파일리스트가 비어있습니다.")
    else:
        remove_empty_label_files(files)
        print("Empty txt files are deleted. 빈 파일 삭제 완료")
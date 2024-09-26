import os
import glob


def process_label_files(label_file_list):
    """
    레이블 파일 리스트를 받아 각 파일에서 레이블이 '0', '1', '4'인 바운딩 박스만 남기고 나머지는 제거합니다.
    """
    for file in label_file_list:
        with open(file, 'r') as f:
            lines = f.readlines()

        new_lines = []
        for line in lines:
            parts = line.strip().split()
            if parts[0] not in ('2', '3'):  # 레이블 2,3 제거
                if parts[0] == '4':  # 레이블 4를 2로 변경
                    parts[0] = '2'
            new_lines.append(" ".join(parts) + "\n")  # 변경된 줄을 저장

        # 필터링된 내용으로 파일 덮어쓰기
        with open(file, 'w') as f:
            f.writelines(new_lines)  # 필터링된 내용을 파일에 다시 씀


if __name__ == '__main__':

    # 작업할 디렉토리 명시
    root_dir = os.path.join(os.getcwd(), 'NECK_HYOID\labels')
    print(f"레이블 파일이 있는 디렉토리: {root_dir}")

    # 재귀적으로 모든 하위 디렉토리에서 '.txt' 파일 찾기
    files = glob.glob(os.path.join(root_dir, '**', '*.txt'), recursive=True)

    # 파일리스트가 빈 리스트인지 확인
    if len(files) == 0:
        print("file list is empty! 파일리스트가 비어있습니다.")
    else:
        # 레이블 파일 필터링 함수 호출
        process_label_files(files)
        print("Completed txt files overwritting. 파일 덮어쓰기 완료")

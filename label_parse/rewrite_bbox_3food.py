import os
import glob


def label_file_rewrite(label_file_list):
    """
    레이블 파일 리스트를 받아 각 파일에서 레이블이 '3'인 바운딩 박스만 남기고 나머지는 제거합니다.
    """
    for file in label_file_list:

        with open(file, 'r') as f:
            lines = f.readlines()

        # 레이블이 3인 줄만 남기고, 레이블을 0으로 변경
        filtered_lines = []
        for line in lines:
            parts = line.strip().split()
            if parts[0] == '3':
                parts[0] = '0'  # 레이블을 0으로 변경
                filtered_lines.append(" ".join(parts) + "\n")

        # 필터링된 내용으로 파일 덮어쓰기
        with open(file, 'w') as f:
            f.writelines(filtered_lines)

        # 파일이 비어있으면 삭제
        if len(filtered_lines) == 0:
            os.remove(file)
            print(f"빈 파일 삭제됨: {file}")


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
        # 레이블 파일 필터링 함수 호출
        label_file_rewrite(files)
        print("파일 덮어쓰기 완료")

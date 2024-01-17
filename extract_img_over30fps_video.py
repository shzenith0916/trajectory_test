import cv2
import os
import argparse
import sys
from datetime import datetime


def rename_files_with_spaces(directory):
    for filename in os.listdir(directory):
        if " " in filename:
            new_filename = filename.replace(" ", "")
            os.rename(os.path.join(directory, filename),
                      os.path.join(directory, new_filename))
            print(f'Renamed "{filename}" to "{new_filename}"')
        else:
            print(f"No spaces in filename: {filename}")


def extract_frames(video_dir, output_dir):

    # Rename files to remove spaces
    rename_files_with_spaces(video_dir)

    # 동영상 파일을 검색
    video_files = [f for f in os.listdir(
        video_dir) if f.endswith('.avi') or f.endswith('.mp4')]

    for video_file in video_files:
        video_path = os.path.join(video_dir, video_file)
        video_capture = cv2.VideoCapture(video_path)

        # 동영상 이름 출력
        video_name = os.path.splitext(video_file)[0]
        print(f'Working on video: {video_name}')

        # 동영상의 프레임 레이트 확인
        fps = int(video_capture.get(cv2.CAP_PROP_FPS))
        print(f'프레임 레이트: {fps} FPS')

        # 동영상 비디오의 프레임 카운트
        frame_count = int(video_capture.get(cv2.CAP_PROP_FRAME_COUNT))
        print("동영상 전체 프레임 카운트:", frame_count)
        # 동영상의 길이 초 확인
        durationInSeconds = frame_count // fps
        print("동영상 전체 길이 초:", durationInSeconds)

        # 이미지 추출할 시작 초와 끝나는 초 시간, input으로 입력
        start_second = int(input('시작 초 시간 : '))
        end_second = int(input('끝나는 초 시간: '))

        # 시작 및 종료 프레임 변수 지정 # 프레임 계산 = 초(단위) * fps(frame_rate)
        start_frame = start_second * fps
        end_frame = end_second * fps

        # 각 동영상에 대한 별도의 폴더 생성
        video_output_dir = os.path.join(output_dir, video_name)
        os.makedirs(video_output_dir, exist_ok=True)

        for idx, frame_idx in enumerate(range(start_frame, end_frame + 1)):
            if 0 <= frame_idx < frame_count:
                process_frame = False  # process_frame변수를 루프를 돌때마다 False로 재지정

                #  fps보다 크면 홀수 번째 프레임 이미지만 추출
                if fps > 30 and frame_idx % 2 != 0:
                    print("추출하는 프레임 넘버:", frame_idx)
                    process_frame = True

                # 30fps 이하이면, 모든 프레임 이미지 추출
                elif fps <= 30:
                    process_frame = True

                if process_frame:
                    # Format the idx with leading zeros based on the total number of frames
                    # Assuming a maximum of 999 frames for this example, Use zfill to add leading zeros
                    idx_str = str(idx).zfill(3)
                    video_capture.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
                    ret, frame = video_capture.read()

                    if ret:
                        frame_filename = os.path.join(
                            video_output_dir, f'{video_name}_image{idx_str}.jpg')
                        cv2.imwrite(frame_filename, frame)

        video_capture.release()

        print("Image extraction completed.")


def main():
    parser = argparse.ArgumentParser(
        description='Extract frames from video files.')
    parser.add_argument(
        'video_dir', help='Video  directory containing input video files.')
    parser.add_argument(
        'output_dir', help='Directory to save extracted images.')

    args = parser.parse_args()

    today_date = datetime.now().strftime('%Y%m%d')
    output_dir_with_date = os.path.join(
        args.output_dir, f"extracted_images_{today_date}")
    print(output_dir_with_date)

    extract_frames(args.video_dir, output_dir_with_date)


if __name__ == '__main__':
    main()


# 입력 동영상 파일들이 있는 최상위 디렉토리의 경로.
# video_dir = '/home/sh_rsrehab/AKAS_Test/data/video_data/23년2월'
# 출력 이미지를 저장할 디렉토리
# output_dir = 'home/sh_rsrehab/AKAS_Test/data/extract_img'

# https://pioneergu.github.io/posts/sys.stdout-redirection/

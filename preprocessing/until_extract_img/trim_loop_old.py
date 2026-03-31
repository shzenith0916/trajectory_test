# -*- coding: utf-8 -*-

import os
import ffmpeg
import pandas as pd

# %pip install openpyxl


def rename_files(directory):

    for file in os.listdir(directory):
        old_name = os.path.join(directory, file)
        new_name = os.path.join(directory, file.replace(" ", ''))
        os.rename(old_name, new_name)
        print("Renamed: {} to {}".format(old_name, new_name))


def trim_video(input_file, output_file, start, end):
    try:
        (
            ffmpeg
            .input(input_file, ss=start, to=end)
            .output(output_file, vcodec="copy", acodec="copy")
            .run(capture_stdout=True, capture_stderr=True)
        )
        print(f"Successfully trimmed video.")
    except ffmpeg.Error as e:
        print(f"An error occurred while trimming video {input_file}:")
        print(e.stderr.decode())


if __name__ == '__main__':
    excel_path = os.path.join(os.getcwd(), "VFSS_video_list_김소현_궤적용.xlsx")

    try:
        xlsx_data = pd.read_excel(
            excel_path, engine='openpyxl', sheet_name='VFSS_2023년4월')
        df = pd.DataFrame(xlsx_data)

        video_path = os.path.join(os.getcwd(), "4월")
        rename_files(video_path)

        for index, row in df.iterrows():
            video_name = row["영상파일명"].replace(" ", "")
            start_second = row["삼킴_시작(초)"]
            end_second = row["삼킴_끝(초)"]

            input_v = os.path.join(video_path, video_name)
            if not os.path.exists(input_v):
                print("Video file does not exists: {}".format(input_v))
                continue

            file_name, file_extension = os.path.splitext(input_v)
            output_v = file_name + "_trimmed" + file_extension

            trim_video(input_v, output_v, start_second, end_second)

    except Exception as e:
        print("An error occurred {}".format(e))
        if hasattr(e, 'stderr'):
            print(e.stderr.decode())

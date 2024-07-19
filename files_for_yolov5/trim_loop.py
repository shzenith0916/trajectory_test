# -*- coding: utf-8 -*-

import os
import ffmpeg
import pandas as pd

# %pip install openpyxl


# ffmpeg -i input.mp4 -ss 00:04:40 -to 00:04:50 -vcodec -acodec output.mp4

def trim_video(input_file, output_file, start, end):
    (
        ffmpeg
        .input(input_file, ss=start, to=end)
        # 'codec="copy"' applies to both video and audio. Work as the line -vcodec -acodec in FFmpeg.
        # Without the codec line, the video quality is not assured.
        .output(output_file, codec="copy")
        .run(capture_stdout=True, capture_stderr=True)
    )


if __name__ == '__main__':
    excel_path = "//mnt/c//Users//USER//Downloads//ffmpeg//VFSS_video_list_김소현_궤적용.xlsx"
    xlsx_data = pd.read_excel(excel_path, engine='openpyxl', sheet_name='VFSS_2023년6월')
    df = pd.DataFrame(xlsx_data)

    for index, row in df.iterrows():
        


    file_name, file_extension = os.path.splitext(input_file)
    output_file = file_name + "_trimmed" + file_extension

    # Call the method
    trim_video(input_file, output_file, st, et)
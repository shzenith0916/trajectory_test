# -*- coding: utf-8 -*-

import os
import ffmpeg
import pandas as pd

def rename_files(directory):
    if not os.path.exists(directory):
        print(f"Directory not found: {directory}")
        return
    
    for file in os.listdir(directory):
        old_name = os.path.join(directory, file)        
        new_name = os.path.join(directory, file.replace(" ", ''))
        os.rename(old_name, new_name)
        print(f"Renamed: {old_name} to {new_name}")


def trim_video(input_file, output_file, start, end):
    try:
        (
            ffmpeg
            .input(input_file, ss=start, to=end)
            .output(output_file, codec="copy")
            .run()
        )
        print(f"Trimmed video saved as {output_file}")
    except ffmpeg.Error as e:
        print(f"Failed to trim video {input_file}. Error: {e}")



if __name__ == '__main__':
    excel_path = "//mnt/c/Users/USER/Downloads/ffmpeg/VFSS_video_list_김소현_궤적용.xlsx"
    try:
        xlsx_data = pd.read_excel(excel_path, engine='openpyxl', sheet_name='VFSS_2023년월')
        df = pd.DataFrame(xlsx_data)
        
        video_path = "//mnt/c/Users/USER/Downloads/ffmpeg"
        rename_files(video_path)
        
        for index, row in df.iterrows():
            video_name = row["영상파일명"].replace(" ", "")
            start_second = row["삼킴_시작(초)"]
            end_second = row["삼킴_끝(초)"]
            
            input_v = os.path.join(video_path, video_name)
            file_name, file_extension = os.path.splitext(input_v)
            output_v = file_name + "_trimmed" + file_extension
            
            trim_video(input_v, output_v, start_second, end_second)
    
    except Exception as e:
        print(f"An error occurred: {e}")
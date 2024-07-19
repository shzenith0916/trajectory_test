# -*- coding: utf-8 -*-


import os
import shutil
import pandas as pd

print("Please type input directory path:")
input_dir = input()
origin_dir = input_dir

print("Please type output directory path:")
output_dir = input()
final_dir = output_dir

print("Please type excel file path:")
excel_path = input()


def read_video_names_excel(path):

    df = pd.read_excel(path)
    return df["영상파일명"].tolist()

video_names = read_video_names_excel(excel_path)


def copy_files(root_dir, dest_dir, video_names):
    """Copy files from root directory to destination directory based on Excel file list."""
    
    for root, dirs, filenames in os.walk(root_dir): # Use os.walk since there are multiple folders inside the root
        for filename in filenames: 
            source = os.path.join(root, filename)
            destination = os.path.join(dest_dir, filename)
            
            # Ensure the destination directory exists
            os.makedirs(dest_dir, exist_ok=True)
            
            # Copy the video file
            shutil.copy2(source, destination)
            print("Copied: %s to %s" % (source, destination))

# Run the function
copy_files(origin_dir, final_dir, video_names)
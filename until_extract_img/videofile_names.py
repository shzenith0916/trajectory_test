# -*- coding: utf-8 -*-

import os
import shutil

# Define the root directory to search and the destination directory for copying
print("please type input directory path")
input_dir = input()
origin_dir = input_dir   # Update this path e.g.E:\VFSS영상\2023년\23년 4월
print("please type output directory path")
output_dir = input()
final_dir = output_dir  # Update this path e.g.E:\VFSS영상\2023년\4월_SF_YP




def copy_criteria(filename):
    """Check if the file name meets the criteria."""
    # File name contains "SF" or "YP", does not contain "AP", "(Rt rot)", or "(Lt rot)"
    return ("SF" in filename or "YP" in filename) and all(x not in filename for x in ["AP", "(Rt rot)",  "(Lt rot)", "fail"])


def copy_files(root_dir, dest_dir):
    """Copy files from root directory to destination directory based on criteria."""
    
    for (root_dir, dirs, filenames) in os.walk(root_dir): # os.walk returns 3 variables: root, dir, files 
        for filename in filenames:
            if copy_criteria(filename): 
                
                source_path = os.path.join(root_dir, filename)
                destination_path = os.path.join(dest_dir, filename)

                # Ensure the destination directory exists
                os.makedirs(dest_dir, exist_ok=True)

                # Copy the file
                shutil.copy2(source_path, destination_path)
                print("Copied: %s to %s" % (source_path, destination_path))

# Run the function
copy_files(origin_dir, final_dir)



def save_filenames_to_txt(dest_dir, txt_name):
    """Save all filenames in the destination directory to a text file."""
    
    output_file = os.path.join(dest_dir, txt_name)
    
    with open(output_file, 'w') as file:
        for filename in os.listdir(dest_dir):
            file.write(filename + '\n')
            
    print(f"Filenames have been saved to {output_file}")



for subdir in os.listdir(final_dir):
    print(os.path.join(final_dir, subdir))

# Save filenames in destination directory to a text file
save_filenames_to_txt(final_dir, "videolist.txt")


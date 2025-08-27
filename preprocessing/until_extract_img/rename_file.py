import os

def rename_and_extract_info(directory):
    for filename in os.listdir(directory):
        # Extract the Unique ID and the number between parentheses
        try:
            # Assuming the Unique ID is before the underscore
            unique_id = filename.split('_')[0]
            # Assuming the number is within parentheses
            number_in_parentheses = filename.split('(')[1].split(')')[0]
        except IndexError:
            print(
                f"Skipping file {filename} as it doesn't match the expected pattern.")
            continue

        new_filename = f"{unique_id}_{number_in_parentheses}.avi"

        # Rename the file
        os.rename(os.path.join(directory, filename), os.path.join(directory, new_filename))
        print(f"Renamed {filename} to {new_filename} ")

# Use the function as needed
directory_path = "/home/rsrehab/AKAS_Test/Residue/동영상에서이미지추출/2023_4~6월동영상"
rename_and_extract_info(directory_path)
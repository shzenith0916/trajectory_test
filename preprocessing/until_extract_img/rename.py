# Set the path for the input file and the directory prefix
input_file = 'train.txt'
directory_prefix = 'data/obj_train_data/'

# Open the input file and read all lines
with open(input_file, 'r') as file:
    file_names = file.readlines()

# Open the file again in write mode to overwrite with modified paths
with open(input_file, 'w') as file:
    for name in file_names:
        # Construct the new path by stripping any newlines and prepending the directory path
        new_path = directory_prefix + name.strip() + '\n'
        # Write the new path to the file
        file.write(new_path)

print("File paths have been updated in train.txt.")

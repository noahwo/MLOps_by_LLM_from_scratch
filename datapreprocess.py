import os
import re

# Define the directory path
directory_path = "./data/stages/exp1_data_eng/"

# Iterate over all files in the directory
for filename in os.listdir(directory_path):
    # Check if the filename matches the pattern **_ans*_**
    match = re.match(r"(.+)_ans(\d+)(_.+)", filename)
    if match:
        # Construct the new filename
        new_filename = f"ans{match.group(2)}_{match.group(1)}{match.group(3)}"
        # Get full file paths
        old_file = os.path.join(directory_path, filename)
        new_file = os.path.join(directory_path, new_filename)
        # Rename the file
        os.rename(old_file, new_file)
        print(f"Renamed: {filename} -> {new_filename}")

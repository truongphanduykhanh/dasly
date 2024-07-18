"""Mimic the integrator by copying files from the source directory to the
destination directory with a delay. Keeping the same directory structure when
copying files."""

__author__ = 'khanh.p.d.truong@ntnu.no'
__date__ = '2024-07-15'


import os
import shutil
import time

import yaml


# Define the path to the YAML file
yaml_path = 'config.yml'

# Open and read the YAML file
with open(yaml_path, 'r') as file:
    params = yaml.safe_load(file)

# Get the source and destination directories from the YAML file
source_dir = params['integrator_dir']
dest_dir = params['input_dir']


# Ensure the destination directory exists
os.makedirs(dest_dir, exist_ok=True)

# Copy each file from source to destination with a delay
for root, dirs, files in os.walk(source_dir):
    # Get the relative path from the source directory
    relative_path = os.path.relpath(root, source_dir)

    # Create the corresponding subdirectories in the destination directory
    destination_subdir = os.path.join(dest_dir, relative_path)
    os.makedirs(destination_subdir, exist_ok=True)

    sorted_files = sorted(files)

    for file_name in sorted_files:
        source_file_path = os.path.join(root, file_name)
        destination_file_path = os.path.join(destination_subdir, file_name)
        # Copy the file
        shutil.copy2(source_file_path, destination_file_path)
        print(f'Copied file: {file_name}')
        # # Move the file - Alternative for moving
        # shutil.move(source_file_path, destination_file_path)
        # print(f'Moved file: {file_name}')
        # Add a delay between copies
        time.sleep(10)

print('Copying process completed.')

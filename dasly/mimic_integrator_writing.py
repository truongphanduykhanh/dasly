import os
import shutil
import time


# Define the source and destination directories
source_dir = '/mnt/Datastore/usr/kptruong/dasly/test/test_aastfjordbrua_integrator'
destination_dir = '/mnt/Datastore/usr/kptruong/dasly/test/test_aastfjordbrua_input'

# Ensure the destination directory exists
os.makedirs(destination_dir, exist_ok=True)

# Copy each file from source to destination with a delay
for root, dirs, files in os.walk(source_dir):
    # Get the relative path from the source directory
    relative_path = os.path.relpath(root, source_dir)

    # Create the corresponding subdirectories in the destination directory
    destination_subdir = os.path.join(destination_dir, relative_path)
    os.makedirs(destination_subdir, exist_ok=True)

    sorted_files = sorted(files)

    for file_name in sorted_files[0:20]:
        source_file_path = os.path.join(root, file_name)
        destination_file_path = os.path.join(destination_subdir, file_name)

        # Copy the file
        shutil.copy2(source_file_path, destination_file_path)

        print(f"Copied file: {file_name}")

        # Add a delay between copies
        time.sleep(10)

print("Copying process completed.")

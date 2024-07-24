import os

def list_files_with_size(directory):
    for root, dirs, files in os.walk(directory):
        for file in files:
            file_path = os.path.join(root, file)
            file_size = os.path.getsize(file_path)
            print(f"{file_path}: {file_size} bytes")

# Replace 'your_directory' with the path to the directory you want to list
directory = '/mnt/Datastore/usr/kptruong/dasly_repo/data/Svalbard_whale/20220822/dphi'
list_files_with_size(directory)

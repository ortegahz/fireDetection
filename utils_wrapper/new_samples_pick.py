import filecmp
import os
import shutil

from utils_wrapper.utils import make_dirs


def find_and_copy_unique_files(source_folder, target_folder, destination_folder):
    source_files = os.listdir(source_folder)
    target_files = os.listdir(target_folder)

    print(f'{len(source_files)} files found in source_folder')
    print(f'{len(target_files)} files found in target_files')

    for source_file in source_files:
        source_file_path = os.path.join(source_folder, source_file)

        if os.path.isfile(source_file_path):
            if source_file not in target_files:
                shutil.copy2(source_file_path, os.path.join(destination_folder, source_file))
                print(f"Copying {source_file} to {destination_folder}")


if __name__ == "__main__":
    source_folder = '/media/manu/ST2000DM005-2U91/fire/data/pseudo_filtered/vis/正例'
    target_folder = '/media/manu/ST2000DM005-2U91/fire/data/pseudo_filtered/vis-10-09/正例'
    destination_folder = '/media/manu/ST2000DM005-2U91/fire/data/pseudo_filtered/vis-11-05/正例'

    make_dirs(destination_folder, reset=True)

    find_and_copy_unique_files(source_folder, target_folder, destination_folder)

import os
import shutil

from tqdm import tqdm

from utils_wrapper.utils import make_dirs


def get_files_without_extension(folder_path, extensions):
    """Returns a set of file prefixes (without extensions) in the given folder with the specified extensions."""
    return {os.path.splitext(filename)[0] for filename in os.listdir(folder_path) if filename.endswith(extensions)}


def copy_files(source_folder, destination_folder, file_prefixes, extensions):
    """Copies files with specified prefixes and any matching extension from source to destination."""
    for prefix in tqdm(file_prefixes, desc="Copying files", unit="file"):
        # Check each extension and copy the existing file
        for ext in extensions:
            source_file = os.path.join(source_folder, prefix + ext)
            destination_file = os.path.join(destination_folder, prefix + ext)
            if os.path.exists(source_file):
                shutil.copyfile(source_file, destination_file)
                # print(f"Copied {prefix + ext} to {destination_folder}")


def process_folders(img_folder_A, label_folder_A, img_folder_B, img_folder_C, label_folder_C, image_extensions):
    """Process the image and label folders to find and copy unique files."""
    # Get unique image prefixes
    image_prefixes_A = get_files_without_extension(img_folder_A, image_extensions)
    image_prefixes_B = get_files_without_extension(img_folder_B, image_extensions)
    unique_image_prefixes = image_prefixes_A - image_prefixes_B

    # Copy unique images
    copy_files(img_folder_A, img_folder_C, unique_image_prefixes, image_extensions)

    # Copy corresponding label files
    copy_files(label_folder_A, label_folder_C, unique_image_prefixes, (".txt",))


# Example usage with actual folder paths and image extensions
subset = 'valid'
img_folder_B = '/media/manu/ST2000DM005-2U91/fire/data/webs/fish_labeled_v0/Fire and Smoke20241231/valid/images'
root_A = '/media/manu/ST2000DM005-2U91/fire/data/webs/fish'
root_C = '/home/manu/tmp/fish_pick'
img_folder_A = os.path.join(root_A, subset, 'images')
label_folder_A = os.path.join(root_A, subset, 'labels')
img_folder_C = os.path.join(root_C, subset, 'images')
label_folder_C = os.path.join(root_C, subset, 'labels')

# Define accepted image extensions
image_extensions = ('.jpg', '.png', '.jpeg', '.bmp')

make_dirs(img_folder_C, reset=True)
make_dirs(label_folder_C, reset=True)

process_folders(img_folder_A, label_folder_A, img_folder_B, img_folder_C, label_folder_C, image_extensions)

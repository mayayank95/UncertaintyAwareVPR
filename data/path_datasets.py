import argparse
import os
from glob import glob

def generate_image_paths_file(dataset_folder: str):
    """
    Finds all images in the folder (recursively) and saves their relative paths 
    to a file named 'dataset_folder'_images_paths.txt.
    """
    dataset_folder = os.path.normpath(dataset_folder)
    output_file = dataset_folder + "_images_paths.txt"
    
    # Skip if file exists to avoid redundant processing
    if os.path.exists(output_file):
        print(f"Skipping: {output_file} (already exists)")
        return

    valid_exts = (".jpg", ".jpeg", ".png")
    
    # Recursive search to find images even in deep subfolders (like 37.7, 37.73)
    all_files = glob(os.path.join(dataset_folder, "**", "*"), recursive=True)
    
    # Filter for images and calculate relative path from the dataset_folder
    prefix_len = len(dataset_folder) + 1
    relative_paths = [
        p[prefix_len:] for p in all_files 
        if p.lower().endswith(valid_exts) and os.path.isfile(p)
    ]
    
    if not relative_paths:
        return # No images found in this branch

    relative_paths.sort()

    with open(output_file, "w") as f:
        f.write("\n".join(relative_paths))
    
    print(f"Created: {output_file} ({len(relative_paths)} images)")

def process_datasets(root_path: str):
    """
    Walks through the directory tree and triggers path generation for target folders.
    """
    root_path = os.path.abspath(root_path)
    if not os.path.exists(root_path):
        print(f"Error: Path '{root_path}' does not exist.")
        return
 
    for root, dirs, files in os.walk(root_path):
        for folder_name in dirs:
            full_path = os.path.join(root, folder_name)
            subfolders = [f.path for f in os.scandir(full_path) if f.is_dir()]
            for subfolder_name in subfolders:
                sub_full_path = os.path.join(root, folder_name, subfolder_name)
                if subfolder_name.endswith("train"):
                    generate_image_paths_file(sub_full_path)
                else:                    
                    sub_subfolders = [f.path for f in os.scandir(sub_full_path) if f.is_dir()]
                    for sub_subfolder_name in sub_subfolders:
                        sub_sub_full_path = os.path.join(root, folder_name, subfolder_name, sub_subfolder_name)
                        generate_image_paths_file(sub_sub_full_path)
                        

if __name__ == "__main__":
    p = argparse.ArgumentParser(description="Generate *_images_paths.txt files under dataset folders.")
    p.add_argument(
        "root",
        type=str,
        help="Datasets root directory (the tree your benchmarks live under)",
    )
    process_datasets(p.parse_args().root)
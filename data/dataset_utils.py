import os
import logging
from glob import glob
from PIL import ImageFile

# Ensure the training doesn't crash if an image is slightly corrupted or truncated
ImageFile.LOAD_TRUNCATED_IMAGES = True

def read_images_paths(dataset_folder, get_abs_path=True):
    """
    Finds image paths within 'dataset_folder' efficiently.
    Works for both Train and Test (Database/Queries).
    
    Parameters
    ----------
    dataset_folder : str, folder containing images.
    get_abs_path : bool, if True return absolute paths, otherwise relative.
    
    Returns
    -------
    images_paths : list[str], paths of images found.
    """
    if not os.path.exists(dataset_folder):
        raise FileNotFoundError(f"Folder {dataset_folder} does not exist")

    # Normalize folder path to handle trailing slashes and different OS styles
    dataset_folder = os.path.normpath(dataset_folder)
    file_with_paths = dataset_folder + "_images_paths.txt"
    
    # 1. FAST PATH: If the text file exists, read it directly (extremely fast for large datasets)
    if os.path.exists(file_with_paths):
        logging.debug(f"Reading paths from {file_with_paths}")
        with open(file_with_paths, "r") as file:
            images_paths = file.read().splitlines()
        
        if get_abs_path:
            # Use os.path.join for robust path construction
            images_paths = [os.path.join(dataset_folder, path) for path in images_paths]
            
        # Quick sanity check on the first image
        sample_path = images_paths[0] if get_abs_path else os.path.join(dataset_folder, images_paths[0])
        if not os.path.exists(sample_path):
            raise FileNotFoundError(f"Image {sample_path} not found. Check {file_with_paths}")
            
    # 2. SLOW PATH: Use glob if no text file is provided
    else:
        logging.debug(f"Searching images in {dataset_folder} with glob() (this may be slow for large folders)")
        # Search for all files recursively
        all_files = glob(os.path.join(dataset_folder, "**", "*"), recursive=True)
        
        valid_exts = (".jpg", ".jpeg", ".png")
        
        # Optimization: Filter and process paths in a single pass without expensive system calls
        if get_abs_path:
            images_paths = sorted([
                p for p in all_files 
                if p.lower().endswith(valid_exts) and os.path.isfile(p)
            ])
        else:
            # Use simple string slicing instead of os.path.abspath to save time
            prefix_len = len(dataset_folder) + 1
            images_paths = sorted([
                p[prefix_len:] for p in all_files 
                if p.lower().endswith(valid_exts) and os.path.isfile(p)
            ])
        
        if len(images_paths) == 0:
            raise FileNotFoundError(f"Directory {dataset_folder} contains no valid images {valid_exts}")
    
    return images_paths
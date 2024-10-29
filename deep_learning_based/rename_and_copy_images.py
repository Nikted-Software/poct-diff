import os
import shutil

def get_next_available_index(folder_path):
    """Find the next available index for naming files in the destination folder."""
    existing_files = os.listdir(folder_path)
    existing_indices = [int(f.split('.')[0]) for f in existing_files if f.split('.')[0].isdigit()]
    if existing_indices:
        return max(existing_indices) + 1
    else:
        return 1

def copy_and_rename_images(source_folder, destination_folder):
    """Copy and rename images from the source folder to the destination folder."""
    next_index = get_next_available_index(destination_folder)
    
    if not os.path.exists(destination_folder):
        os.makedirs(destination_folder)
    
    for filename in os.listdir(source_folder):
        source_path = os.path.join(source_folder, filename)
        if os.path.isfile(source_path):
            # Define the new filename and path
            new_filename = f"{next_index}.jpg"  # Change extension if needed
            destination_path = os.path.join(destination_folder, new_filename)
            
            # Copy and rename the file
            shutil.copy2(source_path, destination_path)
            print(f"Copied {filename} to {destination_path}")
            
            next_index += 1

# Define paths to source and destination folders
source_folder = '2'
destination_folder = 'images'

# Copy and rename images
copy_and_rename_images(source_folder, destination_folder)

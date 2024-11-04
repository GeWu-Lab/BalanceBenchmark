import os
import shutil
root_dir = "./experiments"
def cleanup_directories(root_path):
    """
    Recursively search through directories and remove those starting with 'train_2024'
    if they have empty checkpoint directories.
    
    Args:
        root_path (str): The root directory to start searching from
    """
    # Convert to absolute path to avoid any relative path issues
    root_path = os.path.abspath(root_path)
    backup_path = "/home/shaoxuan_xu/backup"
    # List to store directories to be removed
    to_remove = []
    dont_remove = []
    # Walk through all directories
    for dirpath, dirnames, filenames in os.walk(root_path, topdown=False):
        # Get the current directory name
        current_dir = os.path.basename(dirpath)
        
        # Check if directory starts with "train_2024"
        if current_dir.startswith("train_2024"):
            # Construct path to checkpoint directory
            checkpoint_path = os.path.join(dirpath, "checkpoints")
            # Check if checkpoint directory exists and is empty
            if os.path.exists(checkpoint_path):
                if not os.listdir(checkpoint_path):
                    to_remove.append(dirpath)
                else:
                    dont_remove.append(dirpath)
    to_remove.sort()
    dont_remove.sort()
    # print(to_remove)
    # print(dont_remove)
    for dir_path in to_remove:
        try:
            # Get original directory name
            original_name = os.path.basename(dir_path)
            name = dir_path.split("/")
            name = "/".join(name[-3:])
            # Create backup path with original name
            backup_path_full = os.path.join(backup_path, name)
            
            # Handle case where backup directory already exists
            if os.path.exists(backup_path_full):
                backup_path_full = os.path.join(backup_path, f"{name}")
                print(f"Backup directory already exists, creating: {backup_path_full}")
            
            # # Backup the directory
            # print(f"Backing up directory: {dir_path}")
            # print(f"To: {backup_path_full}")
            # shutil.copytree(dir_path, backup_path_full)
            
            # Remove the original directory
            print(f"Removing original directory: {dir_path}")
            shutil.rmtree(dir_path)
            
        except Exception as e:
            print(f"Error processing {dir_path}: {str(e)}")
    # Remove the identified directories
    # for dir_path in to_remove:
    #     try:
    #         print(f"Removing directory: {dir_path}")
    #         shutil.rmtree(dir_path)
    #     except Exception as e:
    #         print(f"Error removing {dir_path}: {str(e)}")

if __name__ == "__main__":
    # Get the directory path from user input
    root_directory = root_dir
    # Verify the directory exists
    if not os.path.exists(root_directory):
        print("Error: Specified directory does not exist!")
    else:
        # Execute the cleanup
        print("Starting directory cleanup...")
        cleanup_directories(root_directory)
        print("Cleanup completed!")
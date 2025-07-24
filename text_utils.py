import re
import shutil
import string
from typing import List, Dict
from pathlib import Path

def ensure_directory_exists(directory_path: str, debug: bool = False) -> bool:
    """
    Creates a directory if it doesn't exist.
    
    Args:
        directory_path (str): Path to the directory to create
        debug (bool): If True, prints debug information
        
    Returns:
        bool: True if directory exists or was created successfully, False otherwise
    """
    try:
        if debug:
            print(f"\n=== Ensuring Directory Exists ===")
            print(f"Directory path: {directory_path}")
        
        # Convert to Path object
        dir_path = Path(directory_path)
        
        # Check if directory already exists
        if dir_path.exists():
            if dir_path.is_dir():
                if debug:
                    print(f"Directory already exists: {directory_path}")
                return True
            else:
                if debug:
                    print(f"Path exists but is not a directory: {directory_path}")
                return False
        
        # Create directory and all parent directories
        dir_path.mkdir(parents=True, exist_ok=True)
        
        if debug:
            print(f"Successfully created directory: {directory_path}")
        
        return True
        
    except Exception as e:
        if debug:
            print(f"Error creating directory: {str(e)}")
        return False
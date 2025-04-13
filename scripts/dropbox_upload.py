#!/usr/bin/env python3
"""
Dropbox File Uploader

This script uploads files from a local folder to a Dropbox account.
It supports large files and shows progress using tqdm.

Usage as a script:
    python dropbox_upload.py --path /path/to/folder [--dest /dropbox/path]

Usage as a module:
    from dropbox_upload import upload_to_dropbox
    result = upload_to_dropbox(path="/path/to/folder", dest_path="/dropbox/path")
"""

import os
import sys
import argparse
import time
from pathlib import Path
from typing import Union, List, Tuple, Optional

import dropbox
from dropbox.files import WriteMode
from dropbox.exceptions import ApiError, AuthError
from tqdm import tqdm

# ===============================================
# Set your Dropbox API token here
DROPBOX_TOKEN = ""  
# ===============================================

# Error codes
ERROR_CODES = {
    'AUTH_ERROR': 1,
    'API_ERROR': 2,
    'FILE_ERROR': 3,
    'UPLOAD_ERROR': 4,
    'GENERAL_ERROR': 5
}

class ProgressTracker:
    """Track upload progress for tqdm."""
    
    def __init__(self, total_size: int, description: str):
        self.progress_bar = tqdm(total=total_size, unit='B', unit_scale=True, 
                                 desc=description, ncols=100)
        self.uploaded = 0
    
    def update(self, chunk_size: int):
        """Update progress bar with the size of the uploaded chunk."""
        self.uploaded += chunk_size
        self.progress_bar.update(chunk_size)
    
    def close(self):
        """Close the progress bar."""
        self.progress_bar.close()

def normalize_dropbox_path(path: str) -> str:
    """
    Normalize a path for Dropbox by ensuring it starts with a slash
    and doesn't have double slashes.
    
    Args:
        path: The path to normalize
        
    Returns:
        Normalized path
    """
    # Ensure path starts with a slash
    if not path.startswith('/'):
        path = '/' + path
    
    # Remove any double slashes
    while '//' in path:
        path = path.replace('//', '/')
        
    return path

def check_path(path: str) -> Tuple[bool, List[Path], str]:
    """
    Check if path exists and return list of files to upload.
    
    Args:
        path: Path to file or directory
        
    Returns:
        Tuple containing (success, list_of_files, absolute_source_path)
    """
    # Convert to absolute path to handle relative paths correctly
    abs_path = os.path.abspath(path)
    target_path = Path(abs_path)
    
    if not target_path.exists():
        print(f"ERROR: Path does not exist: {path}")
        return False, [], abs_path
    
    files_to_upload = []
    
    if target_path.is_file():
        files_to_upload.append(target_path)
    elif target_path.is_dir():
        for file_path in target_path.glob('**/*'):
            if file_path.is_file():
                files_to_upload.append(file_path)
    else:
        print(f"ERROR: Path is neither a file nor directory: {path}")
        return False, [], abs_path
    
    if not files_to_upload:
        print(f"WARNING: No files found in {path}")
    
    return True, files_to_upload, abs_path

def create_dropbox_dir_structure(dbx: dropbox.Dropbox, path: str) -> bool:
    """
    Create directory structure in Dropbox.
    
    Args:
        dbx: Dropbox client instance
        path: Directory path to create
        
    Returns:
        Boolean indicating success
    """
    if not path or path == '/':
        return True
    
    path = normalize_dropbox_path(path)
    
    # Split path into components
    components = path.split('/')
    components = [comp for comp in components if comp]  # Remove empty components
    
    current_path = ""
    for component in components:
        current_path += "/" + component
        try:
            dbx.files_create_folder_v2(current_path)
            print(f"Created directory: {current_path}")
        except ApiError as e:
            # Ignore if folder already exists (conflict)
            if not isinstance(e.error, dropbox.files.CreateFolderError) or \
               not e.error.is_path() or \
               not e.error.get_path().is_conflict():
                print(f"Warning: Could not create directory {current_path}: {e}")
                # Continue anyway
    
    return True

def extract_path_from_relative(relative_path: str) -> str:
    """
    Extract the correct path structure from a relative path.
    
    For example, "../results/imdb/folder" should become "/results/imdb/folder"
    
    Args:
        relative_path: Relative path to process
        
    Returns:
        Extracted path
    """
    # Split the path into components
    parts = relative_path.replace('\\', '/').split('/')
    
    # Skip the parent directory references
    result_parts = []
    for part in parts:
        if part == '..':
            continue
        if part == '.':
            continue
        result_parts.append(part)
    
    # Join the parts back together
    return '/' + '/'.join(result_parts)

def upload_single_file(dbx: dropbox.Dropbox, file_path: Path, 
                       source_dir: str, dest_path: str, 
                       original_path: str,
                       chunk_size: int = 4 * 1024 * 1024) -> bool:
    """
    Upload a single file to Dropbox.
    
    Args:
        dbx: Dropbox client instance
        file_path: Path to the file to upload
        source_dir: Base local path for determining relative paths
        dest_path: Destination path in Dropbox
        original_path: Original path provided by the user
        chunk_size: Size of chunks for large file upload (default: 4MB)
        
    Returns:
        Boolean indicating success
    """
    file_size = os.path.getsize(file_path)
    
    # Get the sub-path within the source directory
    sub_path = os.path.relpath(file_path, source_dir)
    
    # Prepare Dropbox path
    if dest_path:
        # If a destination path was specified, use it as the base
        if dest_path.endswith('/'):
            dropbox_path = dest_path + sub_path.replace('\\', '/')
        else:
            dropbox_path = dest_path + '/' + sub_path.replace('\\', '/')
    else:
        # If no destination path was specified, use the original path structure
        if original_path.startswith('.') or not original_path.startswith('/'):
            # For relative paths like "../results/..."
            base_path = extract_path_from_relative(original_path)
            # Remove any leading slashes from sub_path to avoid double slashes
            sub_path = sub_path.lstrip('\\/')
            dropbox_path = f"{base_path}/{sub_path}"
        else:
            # For absolute paths
            dropbox_path = f"{original_path}/{sub_path}"
    
    # Normalize the path for Dropbox
    dropbox_path = normalize_dropbox_path(dropbox_path)
    
    print(f"Uploading {file_path} to {dropbox_path} ({file_size} bytes)")
    
    try:
        # Create parent directories if needed
        parent_dir = os.path.dirname(dropbox_path)
        if parent_dir and parent_dir != '/':
            create_dropbox_dir_structure(dbx, parent_dir)
        
        # Use regular upload for smaller files
        if file_size <= chunk_size:
            with open(file_path, 'rb') as f:
                progress = ProgressTracker(file_size, f"Uploading {file_path.name}")
                try:
                    dbx.files_upload(f.read(), dropbox_path, 
                                     mode=WriteMode('overwrite'))
                    progress.update(file_size)  # Update to 100%
                finally:
                    progress.close()
            return True
        
        # Use chunked upload for larger files
        progress = ProgressTracker(file_size, f"Uploading {file_path.name}")
        
        try:
            with open(file_path, 'rb') as f:
                upload_session_start_result = dbx.files_upload_session_start(f.read(chunk_size))
                cursor = dropbox.files.UploadSessionCursor(
                    session_id=upload_session_start_result.session_id,
                    offset=f.tell()
                )
                progress.update(chunk_size)
                
                commit = dropbox.files.CommitInfo(
                    path=dropbox_path, 
                    mode=WriteMode('overwrite')
                )
                
                while f.tell() < file_size:
                    if (file_size - f.tell()) <= chunk_size:
                        # Last chunk
                        dbx.files_upload_session_finish(
                            f.read(chunk_size),
                            cursor,
                            commit
                        )
                        progress.update(min(chunk_size, file_size - (f.tell() - chunk_size)))
                    else:
                        # Upload intermediate chunk
                        dbx.files_upload_session_append_v2(
                            f.read(chunk_size),
                            cursor
                        )
                        cursor.offset = f.tell()
                        progress.update(chunk_size)
                        
                        # Avoid rate limiting
                        time.sleep(0.1)
        finally:
            progress.close()
        
        return True
    
    except ApiError as e:
        print(f"ERROR: API error uploading {file_path}: {e}")
        return False
    except Exception as e:
        print(f"ERROR: Error uploading {file_path}: {e}")
        return False

def upload_to_dropbox(path: str, dest_path: str = "", token: str = None) -> int:
    """
    Upload files to Dropbox.
    
    Args:
        path: Path to file or directory to upload
        dest_path: Destination path in Dropbox
        token: Optional Dropbox API token (uses DROPBOX_TOKEN if not provided)
        
    Returns:
        Error code (0 if successful)
    """
    try:
        # Use provided token or fall back to the fixed token
        dropbox_token = token if token else DROPBOX_TOKEN
        
        # Initialize Dropbox client
        dbx = dropbox.Dropbox(dropbox_token)
        
        # Check authentication
        try:
            account = dbx.users_get_current_account()
            print(f"Connected to Dropbox account: {account.name.display_name}")
        except AuthError:
            print("ERROR: Invalid Dropbox access token!")
            return ERROR_CODES['AUTH_ERROR']
        
        # Normalize the destination path if provided
        if dest_path:
            dest_path = normalize_dropbox_path(dest_path)
            
        # Get list of files to upload
        success, files_to_upload, abs_source_path = check_path(path)
        if not success:
            return ERROR_CODES['FILE_ERROR']
        
        if not files_to_upload:
            print(f"WARNING: No files found to upload in {path}")
            return 0
        
        # Upload files
        print(f"Found {len(files_to_upload)} files to upload")
        
        # Report source and destination paths
        print(f"Source path: {abs_source_path}")
        target_path = dest_path if dest_path else extract_path_from_relative(path)
        print(f"Target Dropbox path structure: {target_path}")
        
        failed_uploads = 0
        
        for file_path in files_to_upload:
            success = upload_single_file(
                dbx, 
                file_path, 
                abs_source_path, 
                dest_path,
                path  # Original path as provided by the user
            )
            if not success:
                failed_uploads += 1
        
        if failed_uploads:
            print(f"ERROR: Failed to upload {failed_uploads} out of {len(files_to_upload)} files")
            return ERROR_CODES['UPLOAD_ERROR']
        
        print(f"Successfully uploaded {len(files_to_upload)} files to Dropbox")
        return 0
    
    except AuthError:
        print("ERROR: Authentication error")
        return ERROR_CODES['AUTH_ERROR']
    except ApiError as e:
        print(f"ERROR: Dropbox API error: {e}")
        return ERROR_CODES['API_ERROR']
    except Exception as e:
        print(f"ERROR: Unexpected error: {e}")
        return ERROR_CODES['GENERAL_ERROR']

def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description='Upload files to Dropbox')
    parser.add_argument('--path', required=True, help='Path to file or directory to upload')
    parser.add_argument('--dest', default='', help='Destination path in Dropbox')
    
    return parser.parse_args()

if __name__ == "__main__":
    args = parse_args()
    result = upload_to_dropbox(
        path=args.path,
        dest_path=args.dest
    )
    sys.exit(result)
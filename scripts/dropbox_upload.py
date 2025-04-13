#!/usr/bin/env python3
"""
Dropbox File Uploader

This script uploads files to a Dropbox directory.
It supports large files, shows progress using tqdm, and refreshes tokens automatically.

Usage as a script:
    python dropbox_uploader.py --path /path/to/file_or_folder [--dest /dropbox/directory] [--file]

Usage as a module:
    from dropbox_uploader import upload_to_dropbox
    result = upload_to_dropbox(path="/path/to/file_or_folder", dest_path="/dropbox/directory", is_file=True)
"""

import os
import sys
import json
import argparse
import time
import requests
from pathlib import Path
from typing import List, Tuple, Dict, Optional

import dropbox
from dropbox.files import WriteMode
from dropbox.exceptions import ApiError, AuthError
from tqdm import tqdm

# ===============================================
# Dropbox API credentials
# ===============================================

# Set your credentials here - you'll need all these for automatic token refresh
DROPBOX_CREDENTIALS = {
    # Your app key (required for refresh)
    'app_key': '',
    
    # Your app secret (required for refresh)
    'app_secret': '',
    
    # Your refresh token (required for refresh)
    'refresh_token': '',
    
    # Your access token (will be refreshed automatically)
    'access_token': 'YOUR_ACCESS_TOKEN'
}

# Path to save refreshed tokens
CREDENTIALS_FILE = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'dropbox_credentials.json')

# Error codes
ERROR_CODES = {
    'AUTH_ERROR': 1,
    'API_ERROR': 2,
    'FILE_ERROR': 3,
    'UPLOAD_ERROR': 4,
    'GENERAL_ERROR': 5,
    'TOKEN_REFRESH_ERROR': 6
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

def load_credentials() -> Dict[str, str]:
    """
    Load Dropbox credentials from file if available, otherwise use defaults.
    
    Returns:
        Dictionary containing credentials
    """
    # Start with default credentials
    credentials = DROPBOX_CREDENTIALS.copy()
    
    # Try to load from file if it exists
    if os.path.exists(CREDENTIALS_FILE):
        try:
            with open(CREDENTIALS_FILE, 'r') as f:
                saved_credentials = json.load(f)
                # Update credentials with saved values
                credentials.update(saved_credentials)
                print("Loaded credentials from file")
        except Exception as e:
            print(f"Warning: Could not load credentials from file: {e}")
    
    return credentials

def save_credentials(credentials: Dict[str, str]) -> bool:
    """
    Save credentials to file.
    
    Args:
        credentials: Dictionary containing credentials
        
    Returns:
        Boolean indicating success
    """
    try:
        with open(CREDENTIALS_FILE, 'w') as f:
            json.dump(credentials, f)
        return True
    except Exception as e:
        print(f"Warning: Could not save credentials to file: {e}")
        return False

def refresh_access_token(credentials: Dict[str, str]) -> Dict[str, str]:
    """
    Refresh the Dropbox access token.
    
    Args:
        credentials: Dictionary containing credentials
        
    Returns:
        Updated credentials dictionary
    """
    print("Access token expired. Attempting to refresh...")
    
    if not credentials.get('app_key') or not credentials.get('app_secret') or not credentials.get('refresh_token'):
        raise ValueError("Cannot refresh token: Missing app_key, app_secret, or refresh_token")
    
    try:
        # Prepare the token refresh request
        url = 'https://api.dropboxapi.com/oauth2/token'
        data = {
            'grant_type': 'refresh_token',
            'refresh_token': credentials['refresh_token'],
            'client_id': credentials['app_key'],
            'client_secret': credentials['app_secret']
        }
        
        # Make the request
        response = requests.post(url, data=data)
        response.raise_for_status()  # Raise exception for HTTP errors
        
        # Parse the response
        token_data = response.json()
        
        # Update the credentials
        credentials['access_token'] = token_data['access_token']
        
        # Save updated credentials
        save_credentials(credentials)
        
        print("Successfully refreshed access token")
        return credentials
        
    except Exception as e:
        print(f"Error refreshing access token: {e}")
        raise

def get_dropbox_client(credentials: Optional[Dict[str, str]] = None) -> dropbox.Dropbox:
    """
    Get a Dropbox client with automatic token refresh.
    
    Args:
        credentials: Optional dictionary containing credentials (loads from file/defaults if None)
        
    Returns:
        Dropbox client instance
    """
    if credentials is None:
        credentials = load_credentials()
    
    # Create the Dropbox client
    dbx = dropbox.Dropbox(credentials['access_token'])
    
    # Test the client to see if the token is valid
    try:
        dbx.users_get_current_account()
        return dbx
    except AuthError:
        # Token invalid, try refreshing
        updated_credentials = refresh_access_token(credentials)
        return dropbox.Dropbox(updated_credentials['access_token'])

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
    
    # Remove trailing slash if present (except for root)
    if path != '/' and path.endswith('/'):
        path = path[:-1]
        
    return path

def extract_path_from_relative(relative_path: str) -> str:
    """
    Extract the correct path structure from a relative path.
    
    For example, "../results/imdb/folder" should become "/results/imdb/folder"
    
    Args:
        relative_path: Relative path to process
        
    Returns:
        Extracted path
    """
    # Replace backslashes with forward slashes
    path = relative_path.replace('\\', '/')
    
    # Split the path into components
    parts = path.split('/')
    
    # Skip the parent directory references
    result_parts = []
    for part in parts:
        if part == '..':
            continue
        if part == '.':
            continue
        result_parts.append(part)
    
    # Join the parts back together
    result = '/' + '/'.join(result_parts)
    
    return result

def get_files_to_upload(path: str, is_file: bool = False) -> Tuple[bool, List[Path], str]:
    """
    Get list of files to upload.
    
    Args:
        path: Path to file or directory
        is_file: Whether the path points to a file
        
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
    
    # Check if path exists and is of the expected type
    if is_file:
        if not target_path.is_file():
            print(f"ERROR: Path is not a file: {path}")
            return False, [], abs_path
        files_to_upload.append(target_path)
    else:
        if not target_path.is_dir():
            print(f"ERROR: Path is not a directory: {path}")
            return False, [], abs_path
        for file_path in target_path.glob('**/*'):
            if file_path.is_file():
                files_to_upload.append(file_path)
    
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

def upload_single_file(dbx: dropbox.Dropbox, file_path: Path, 
                       source_dir: str, dest_path: str, 
                       is_single_file: bool,
                       chunk_size: int = 4 * 1024 * 1024) -> bool:
    """
    Upload a single file to Dropbox.
    
    Args:
        dbx: Dropbox client instance
        file_path: Path to the file to upload
        source_dir: Base local path for determining relative paths
        dest_path: Destination directory in Dropbox
        is_single_file: Whether we're uploading a single file directly
        chunk_size: Size of chunks for large file upload (default: 4MB)
        
    Returns:
        Boolean indicating success
    """
    file_size = os.path.getsize(file_path)
    
    # Ensure destination is a directory path (ends with slash)
    dest_dir = dest_path if dest_path.endswith('/') else dest_path + '/'
    
    # Determine the target path in Dropbox
    if is_single_file:
        # Single file upload - just use the filename
        dropbox_path = dest_dir + os.path.basename(file_path)
    else:
        # Directory upload - preserve directory structure
        rel_path = os.path.relpath(file_path, source_dir)
        rel_path_normalized = rel_path.replace('\\', '/')
        dropbox_path = dest_dir + rel_path_normalized
    
    # Normalize the path for Dropbox
    dropbox_path = normalize_dropbox_path(dropbox_path)
    
    print(f"Uploading {file_path.name} to {dropbox_path} ({file_size} bytes)")
    
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

def upload_to_dropbox(path: str, dest_path: str = "", is_file: bool = False) -> int:
    """
    Upload files to Dropbox with automatic token refresh.
    
    Args:
        path: Path to file or directory to upload
        dest_path: Destination directory in Dropbox (optional, follows source structure if not specified)
        is_file: Whether the source path points to a file
        
    Returns:
        Error code (0 if successful)
    """
    try:
        # Get the Dropbox client (with potential token refresh)
        try:
            credentials = load_credentials()
            dbx = get_dropbox_client(credentials)
        except Exception as e:
            print(f"ERROR: Unable to authenticate with Dropbox: {e}")
            return ERROR_CODES['AUTH_ERROR']
        
        # Check authentication
        try:
            account = dbx.users_get_current_account()
            print(f"Connected to Dropbox account: {account.name.display_name}")
        except (AuthError, ApiError) as e:
            # Try one more refresh if initial authentication fails
            try:
                credentials = refresh_access_token(credentials)
                dbx = dropbox.Dropbox(credentials['access_token'])
                account = dbx.users_get_current_account()
                print(f"Connected to Dropbox account after token refresh: {account.name.display_name}")
            except Exception as refresh_err:
                print(f"ERROR: Authentication failed even after token refresh: {refresh_err}")
                return ERROR_CODES['TOKEN_REFRESH_ERROR']
        
        # Get list of files to upload
        success, files_to_upload, abs_source_path = get_files_to_upload(path, is_file)
        if not success:
            return ERROR_CODES['FILE_ERROR']
        
        if not files_to_upload:
            print(f"WARNING: No files found to upload in {path}")
            return 0
        
        # Upload files
        print(f"Found {len(files_to_upload)} files to upload")
        
        # Determine destination path
        if not dest_path:
            if is_file:
                # For single file, extract the directory structure
                dir_path = os.path.dirname(path)
                if dir_path:
                    dest_path = extract_path_from_relative(dir_path)
                else:
                    dest_path = "/"
            else:
                # For directory, use the same structure
                dest_path = extract_path_from_relative(path)
        
        # Normalize the destination path
        dest_path = normalize_dropbox_path(dest_path)
        
        # Report source and destination paths
        print(f"Source path: {abs_source_path}")
        print(f"Destination Dropbox directory: {dest_path}")
        
        # Create destination directory if it doesn't exist
        create_dropbox_dir_structure(dbx, dest_path)
        
        failed_uploads = 0
        
        for file_path in files_to_upload:
            # Retry up to 3 times with token refresh in case of AuthError
            retries = 3
            success = False
            
            while retries > 0 and not success:
                try:
                    success = upload_single_file(
                        dbx, 
                        file_path, 
                        abs_source_path, 
                        dest_path,
                        is_file
                    )
                    if not success:
                        failed_uploads += 1
                        break
                    
                except (AuthError, ApiError) as e:
                    # Check if it's an auth error
                    if "access token" in str(e).lower() or isinstance(e, AuthError):
                        print(f"Authentication error during upload: {e}")
                        # Try to refresh the token
                        try:
                            credentials = refresh_access_token(credentials)
                            dbx = dropbox.Dropbox(credentials['access_token'])
                            print("Token refreshed, retrying upload...")
                            retries -= 1
                        except Exception as refresh_err:
                            print(f"ERROR: Failed to refresh token: {refresh_err}")
                            return ERROR_CODES['TOKEN_REFRESH_ERROR']
                    else:
                        # Other API error
                        print(f"ERROR: API error during upload: {e}")
                        failed_uploads += 1
                        break
                
                except Exception as e:
                    print(f"ERROR: Unexpected error uploading {file_path}: {e}")
                    failed_uploads += 1
                    break
            
            # If we exited the retry loop without success
            if retries == 0 and not success:
                failed_uploads += 1
        
        if failed_uploads:
            print(f"ERROR: Failed to upload {failed_uploads} out of {len(files_to_upload)} files")
            return ERROR_CODES['UPLOAD_ERROR']
        
        print(f"Successfully uploaded {len(files_to_upload)} files to Dropbox")
        return 0
    
    except Exception as e:
        print(f"ERROR: Unexpected error: {e}")
        return ERROR_CODES['GENERAL_ERROR']

def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description='Upload files to Dropbox')
    parser.add_argument('--path', required=True, help='Path to file or directory to upload')
    parser.add_argument('--dest', default='', help='Destination directory in Dropbox (optional)')
    parser.add_argument('--file', action='store_true', help='Specify that the source path is a file')
    
    return parser.parse_args()

if __name__ == "__main__":
    args = parse_args()
    result = upload_to_dropbox(
        path=args.path,
        dest_path=args.dest,
        is_file=args.file
    )
    sys.exit(result)
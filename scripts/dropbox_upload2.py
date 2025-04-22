#!/usr/bin/env python3
"""
Dropbox File Uploader

This script uploads files to a Dropbox directory.
It supports large files, shows progress using tqdm, refreshes tokens automatically,
and checks for existing files, allowing optional overwrite.

Usage as a script:
    python dropbox_uploader.py --path /path/to/file_or_folder [--dest /dropbox/directory] [--file] [--overwrite]

Usage as a module:
    from dropbox_uploader import upload_to_dropbox
    result = upload_to_dropbox(path="/path/to/file_or_folder", dest_path="/dropbox/directory", is_file=True, overwrite=False)
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
from dropbox.files import WriteMode, FileMetadata, FolderMetadata # Added FileMetadata, FolderMetadata
from dropbox.exceptions import ApiError, AuthError
from dropbox.stone_validators import ValidationError # For specific ApiError check
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
                                 desc=description, ncols=100, leave=False) # Set leave=False
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
    print("Access token expired or invalid. Attempting to refresh...")

    if not credentials.get('app_key') or not credentials.get('app_secret') or not credentials.get('refresh_token'):
        raise ValueError("Cannot refresh token: Missing app_key, app_secret, or refresh_token in credentials")

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

    except requests.exceptions.RequestException as e:
        print(f"Error refreshing access token (Network/HTTP error): {e}")
        if e.response is not None:
            print(f"Response status: {e.response.status_code}")
            print(f"Response body: {e.response.text}")
        raise ConnectionError(f"Failed to refresh token: {e}") from e
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

    if not credentials.get('access_token'):
         raise ValueError("Missing access token in credentials.")

    # Create the Dropbox client
    dbx = dropbox.Dropbox(credentials['access_token'])

    # Test the client to see if the token is valid
    try:
        dbx.users_get_current_account()
        print("Initial token is valid.")
        return dbx
    except AuthError:
        # Token invalid, try refreshing
        print("Initial token invalid or expired.")
        try:
            updated_credentials = refresh_access_token(credentials)
            return dropbox.Dropbox(updated_credentials['access_token'])
        except Exception as refresh_err:
            print(f"ERROR: Failed to refresh token during client initialization: {refresh_err}")
            raise AuthError(request_id=None, error="Failed to refresh token.") # Re-raise as AuthError
    except ApiError as api_err:
        print(f"API Error during initial client check: {api_err}")
        # You might want to handle specific API errors differently here
        raise api_err

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

    # Join the parts back together, ensuring leading slash
    result = '/' + '/'.join(part for part in result_parts if part) # Filter empty parts

    # Handle edge case where input was '.' or './' or '../' resulting in just '/'
    if not result_parts and relative_path.strip() in ['.', './', '..', '../']:
         return "/" # Or maybe raise an error depending on desired behavior

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
    try:
        abs_path = os.path.abspath(path)
        target_path = Path(abs_path)
    except Exception as e:
        print(f"ERROR: Could not resolve path '{path}': {e}")
        return False, [], ""


    if not target_path.exists():
        print(f"ERROR: Path does not exist: {abs_path}")
        return False, [], abs_path

    files_to_upload = []

    # Check if path exists and is of the expected type
    if is_file:
        if not target_path.is_file():
            print(f"ERROR: Path is not a file: {abs_path}")
            return False, [], abs_path
        files_to_upload.append(target_path)
    else:
        if not target_path.is_dir():
            print(f"ERROR: Path is not a directory: {abs_path}")
            return False, [], abs_path
        # Use rglob for recursive search directly
        for file_path in target_path.rglob('*'):
            if file_path.is_file():
                files_to_upload.append(file_path)

    if not files_to_upload:
        if is_file: # Specific message if user specified --file but it wasn't found/valid
             pass # Error handled above
        else:
            print(f"WARNING: No files found in directory {abs_path}")

    return True, files_to_upload, abs_path

def create_dropbox_dir_structure(dbx: dropbox.Dropbox, path: str) -> bool:
    """
    Create directory structure in Dropbox. Ensures parent directories exist.

    Args:
        dbx: Dropbox client instance
        path: Directory path to ensure exists (normalized)

    Returns:
        Boolean indicating success (True even if dir already exists)
    """
    if not path or path == '/':
        return True

    try:
        # Check if the exact path exists first
        dbx.files_get_metadata(path)
        # print(f"Directory already exists: {path}")
        return True
    except ApiError as e:
        if e.error.is_path() and e.error.get_path().is_not_found():
            # Directory doesn't exist, proceed to create
            pass
        else:
            # Different API error, maybe permissions or invalid path format
            print(f"Warning: Error checking directory {path}: {e}")
            # Proceeding might still work if it's a parent dir issue resolved below
            pass # Continue to try creating

    # Create parent directories recursively if needed
    components = path.strip('/').split('/')
    current_path = ""
    for component in components:
        current_path += "/" + component
        try:
            dbx.files_create_folder_v2(current_path)
            print(f"Created directory: {current_path}")
        except ApiError as e:
            # Ignore if folder already exists (conflict)
            if isinstance(e.error, dropbox.files.CreateFolderError) and \
               e.error.is_path() and \
               e.error.get_path().is_conflict():
                 # print(f"Directory already exists: {current_path}") # Optional: uncomment for verbose logging
                 pass # Folder already exists, which is fine
            else:
                # Other error creating folder
                print(f"Warning: Could not create directory {current_path}: {e}")
                # Consider returning False here if creation is critical
                pass # Continue trying to upload anyway

    return True

def check_existing_file(dbx: dropbox.Dropbox, dropbox_path: str, local_file_size: int) -> Tuple[bool, bool]:
    """
    Check if a file exists on Dropbox and if its size matches.

    Args:
        dbx: Dropbox client instance.
        dropbox_path: The full path to the file on Dropbox.
        local_file_size: The size of the local file in bytes.

    Returns:
        Tuple (exists: bool, size_matches: bool)
    """
    try:
        metadata = dbx.files_get_metadata(dropbox_path)
        if isinstance(metadata, FileMetadata):
            # print(f"Found existing file: {dropbox_path}, Size: {metadata.size}") # Debug
            return True, metadata.size == local_file_size
        elif isinstance(metadata, FolderMetadata):
            # A folder exists with the same name, treat as not existing file
            print(f"Warning: A folder exists at the target file path: {dropbox_path}")
            return False, False
    except ApiError as e:
        if e.error.is_path() and e.error.get_path().is_not_found():
            # File does not exist
            return False, False
        else:
            # Other API error (permissions, bad path etc.)
            print(f"Warning: API error checking for existing file {dropbox_path}: {e}")
            # Be cautious: assume it doesn't exist or size doesn't match to avoid skipping incorrectly
            return False, False
    except Exception as e:
        print(f"Warning: Unexpected error checking for existing file {dropbox_path}: {e}")
        return False, False

    return False, False # Default case


def upload_single_file(dbx: dropbox.Dropbox, file_path: Path,
                       source_root_dir: str, dest_root_path: str,
                       is_single_file_upload: bool,
                       overwrite: bool,
                       chunk_size: int = 4 * 1024 * 1024) -> bool:
    """
    Upload a single file to Dropbox, checking for existence and size first.

    Args:
        dbx: Dropbox client instance
        file_path: Path to the local file to upload
        source_root_dir: Absolute base local path for determining relative paths
        dest_root_path: Destination root directory in Dropbox (normalized)
        is_single_file_upload: Whether the original command was for a single file
        overwrite: Whether to overwrite if file exists and size matches
        chunk_size: Size of chunks for large file upload (default: 4MB)

    Returns:
        Boolean indicating success (True if uploaded OR skipped correctly)
    """
    try:
        local_file_size = os.path.getsize(file_path)
    except OSError as e:
        print(f"ERROR: Cannot get size of local file {file_path}: {e}")
        return False

    # Determine the target path in Dropbox
    if is_single_file_upload:
        # Single file upload - place it directly in the dest_root_path
        # Ensure dest_root_path is treated as the *folder* it should go into
        target_filename = file_path.name
        # If dest_root_path was '/', make it empty so join works correctly
        base_dest = dest_root_path if dest_root_path != '/' else ''
        dropbox_path = f"{base_dest}/{target_filename}"
    else:
        # Directory upload - preserve directory structure relative to source_root_dir
        try:
            # Use Path objects for more robust relative path calculation
            rel_path = file_path.relative_to(source_root_dir)
            # Convert to forward slashes for Dropbox path
            rel_path_normalized = str(rel_path).replace('\\', '/')
            # If dest_root_path was '/', make it empty so join works correctly
            base_dest = dest_root_path if dest_root_path != '/' else ''
            dropbox_path = f"{base_dest}/{rel_path_normalized}"
        except ValueError as e:
             print(f"ERROR: Could not determine relative path for {file_path} within {source_root_dir}: {e}")
             return False


    # Normalize the final path
    dropbox_path = normalize_dropbox_path(dropbox_path)

    print(f"Processing: {file_path.name} -> {dropbox_path}")

    # --- Check if file exists on Dropbox ---
    try:
        exists, size_matches = check_existing_file(dbx, dropbox_path, local_file_size)

        if exists:
            if size_matches and not overwrite:
                print(f"Skipping: File already exists with the same size and overwrite=False.")
                return True # Skipped successfully
            elif size_matches and overwrite:
                print(f"Overwriting: File exists with the same size, but overwrite=True.")
            else: # Sizes don't match
                print(f"Overwriting: File exists but sizes differ (local: {local_file_size}, remote: different).")
        else:
            # File doesn't exist, proceed to upload
            pass # No message needed here, will print "Uploading..." below

    except AuthError:
        print("ERROR: Authentication error during file check. Needs token refresh.")
        # Signal to the caller to handle refresh
        raise # Re-raise AuthError to be caught in the main loop
    except ApiError as e:
        print(f"ERROR: API error during file check for {dropbox_path}: {e}")
        return False # Failed the check phase
    except Exception as e:
        print(f"ERROR: Unexpected error during file check for {dropbox_path}: {e}")
        return False # Failed the check phase


    # --- Proceed with Upload ---
    print(f"Uploading {file_path.name} to {dropbox_path} ({local_file_size} bytes)...")
    progress = None # Initialize progress tracker

    try:
        # Create parent directories if needed (do this just before upload)
        parent_dir = os.path.dirname(dropbox_path)
        if parent_dir and parent_dir != '/':
            if not create_dropbox_dir_structure(dbx, parent_dir):
                print(f"ERROR: Failed to create parent directory {parent_dir} for {dropbox_path}. Skipping file.")
                return False # Cannot proceed if parent dir creation failed

        # Use regular upload for smaller files
        if local_file_size <= chunk_size:
            with open(file_path, 'rb') as f:
                file_content = f.read() # Read content
            progress = ProgressTracker(local_file_size, f"Up: {file_path.name}")
            try:
                # Perform the upload
                dbx.files_upload(file_content, dropbox_path,
                                 mode=WriteMode('overwrite')) # Always overwrite if we reach here
                progress.update(local_file_size)  # Update to 100%
            finally:
                if progress: progress.close()
            print(f"Uploaded {file_path.name} successfully (small file).")
            return True

        # Use chunked upload for larger files
        progress = ProgressTracker(local_file_size, f"Up: {file_path.name}")
        try:
            with open(file_path, 'rb') as f:
                # Start session
                first_chunk = f.read(chunk_size)
                upload_session_start_result = dbx.files_upload_session_start(first_chunk)
                cursor = dropbox.files.UploadSessionCursor(
                    session_id=upload_session_start_result.session_id,
                    offset=f.tell() # Bytes read so far
                )
                progress.update(len(first_chunk))

                commit = dropbox.files.CommitInfo(
                    path=dropbox_path,
                    mode=WriteMode('overwrite') # Always overwrite if we reach here
                )

                # Upload remaining chunks
                while f.tell() < local_file_size:
                    next_chunk = f.read(chunk_size)
                    bytes_read_this_chunk = len(next_chunk)

                    if (local_file_size - f.tell()) == 0: # If this is the last chunk
                        # print(f"Finishing session for {file_path.name}, offset {cursor.offset}, chunk size {bytes_read_this_chunk}") # Debug
                        dbx.files_upload_session_finish(
                            next_chunk,
                            cursor,
                            commit
                        )
                    else:
                        # Append chunk
                        # print(f"Appending chunk for {file_path.name}, offset {cursor.offset}, chunk size {bytes_read_this_chunk}") # Debug
                        dbx.files_upload_session_append_v2(
                            next_chunk,
                            cursor
                        )
                        cursor.offset = f.tell() # Update offset for next call

                    progress.update(bytes_read_this_chunk)
                    time.sleep(0.05) # Small sleep potentially helps with very fast uploads / rate limits

        finally:
            if progress: progress.close()

        print(f"Uploaded {file_path.name} successfully (large file).")
        return True

    except ApiError as e:
        print(f"ERROR: API error during upload of {file_path.name}: {e}")
        if progress: progress.close() # Ensure progress bar is closed on error
        # Check if it's an auth error specifically
        if isinstance(e.error, dropbox.common.InvalidAccessToken) or \
           (hasattr(e, 'status_code') and e.status_code == 401):
             raise AuthError(request_id=e.request_id, error="Token likely expired during upload") # Trigger refresh
        return False
    except AuthError as e: # Catch AuthError raised during check or upload itself
         print(f"ERROR: Authentication error during upload of {file_path.name}: {e}")
         if progress: progress.close()
         raise # Re-raise to be caught by the main retry loop
    except Exception as e:
        print(f"ERROR: Unexpected error uploading {file_path.name}: {e}")
        import traceback
        traceback.print_exc() # Print stack trace for unexpected errors
        if progress: progress.close()
        return False

def upload_to_dropbox(path: str, dest_path: str = "", is_file: bool = False, overwrite: bool = False) -> int:
    """
    Upload files to Dropbox with automatic token refresh and existence check.

    Args:
        path: Path to file or directory to upload
        dest_path: Destination directory in Dropbox (optional, follows source structure if not specified)
        is_file: Whether the source path points to a file
        overwrite: Overwrite files in Dropbox if they exist and size matches

    Returns:
        Error code (0 if successful)
    """
    dbx = None
    credentials = {}
    files_uploaded_count = 0
    files_skipped_count = 0
    files_failed_count = 0

    try:
        # --- Authentication and Client Setup ---
        try:
            credentials = load_credentials()
            # Validate essential keys for refresh *before* trying to get client
            if not all(k in credentials and credentials[k] for k in ['app_key', 'app_secret', 'refresh_token', 'access_token']):
                 print("WARNING: Credentials file found but missing app_key, app_secret, refresh_token or initial access_token. Token refresh might fail.")
                 # Proceed cautiously, initial token might work

            dbx = get_dropbox_client(credentials)

            # Verify connection after potential refresh in get_dropbox_client
            account = dbx.users_get_current_account()
            print(f"Connected to Dropbox account: {account.name.display_name}")

        except AuthError as e:
             print(f"ERROR: Initial authentication failed: {e}")
             # Attempt one more refresh *explicitly* if get_dropbox_client failed due to auth
             try:
                 print("Attempting explicit token refresh...")
                 credentials = refresh_access_token(credentials)
                 dbx = dropbox.Dropbox(credentials['access_token'])
                 account = dbx.users_get_current_account()
                 print(f"Connected to Dropbox account after explicit refresh: {account.name.display_name}")
             except Exception as refresh_err:
                 print(f"ERROR: Authentication failed even after explicit token refresh: {refresh_err}")
                 return ERROR_CODES['TOKEN_REFRESH_ERROR']
        except (ApiError, ConnectionError, ValueError, Exception) as e: # Catch broader errors during setup
            print(f"ERROR: Unable to setup Dropbox client or authenticate: {e}")
            return ERROR_CODES['AUTH_ERROR']


        # --- Get Files and Determine Destination ---
        success, files_to_upload, abs_source_path = get_files_to_upload(path, is_file)
        if not success:
            return ERROR_CODES['FILE_ERROR']

        total_files = len(files_to_upload)
        if total_files == 0:
            print(f"No files found to upload in '{path}'.")
            return 0 # Not an error, just nothing to do

        print(f"Found {total_files} file(s) to process.")

        # Determine base destination path in Dropbox
        if not dest_path:
             # If no destination specified, try to mirror the source structure
             if is_file:
                 # Use the parent directory of the source file
                 source_parent = Path(abs_source_path).parent
                 # Extract path relative to CWD or some base if needed, or just use absolute structure
                 # Using extract_path_from_relative on the parent dir name seems reasonable
                 dest_path = extract_path_from_relative(str(source_parent.name)) # Might need adjustment based on desired structure
                 if not dest_path or dest_path == "/":
                     dest_path = "/" # Default to root if parent extraction is tricky
             else:
                 # Use the name of the source directory itself
                 dest_path = extract_path_from_relative(Path(abs_source_path).name)
        elif dest_path == '.': # Treat '.' as root
             dest_path = '/'

        # Normalize the final destination root path
        dest_root_path = normalize_dropbox_path(dest_path)

        print(f"Source base path: {abs_source_path}")
        print(f"Destination Dropbox root directory: {dest_root_path}")

        # Ensure the base destination directory exists (only needed if uploading files directly into it)
        # Individual file uploads will handle their specific parent dirs later
        # However, creating the root now can be helpful.
        if dest_root_path != '/':
             print(f"Ensuring base destination directory exists: {dest_root_path}")
             if not create_dropbox_dir_structure(dbx, dest_root_path):
                  print(f"ERROR: Could not ensure base destination directory {dest_root_path} exists. Aborting.")
                  return ERROR_CODES['API_ERROR']


        # --- Process Each File ---
        source_root_for_relpath = abs_source_path if not is_file else str(Path(abs_source_path).parent)

        for i, file_path in enumerate(files_to_upload):
            print(f"\n--- Processing file {i+1}/{total_files} ---")
            retries = 2 # Allow 1 retry after a potential token refresh
            upload_successful = False

            while retries > 0:
                try:
                    # Pass the correct source root directory for relative path calculation
                    # If it's a single file upload, the "root" is its parent directory
                    # If it's a directory upload, the "root" is the directory itself
                    current_source_root = source_root_for_relpath

                    # Call the upload function for the single file
                    result = upload_single_file(
                        dbx,
                        file_path,
                        current_source_root, # Base for relative path calc
                        dest_root_path,     # Root destination folder on Dropbox
                        is_file,            # Was the original command for one file?
                        overwrite           # Overwrite flag
                    )

                    if result:
                        upload_successful = True
                        # Check if it was skipped based on log message (crude but avoids complex return values)
                        # A more robust way would be for upload_single_file to return a status enum
                        # For now, assume True means success (uploaded or skipped correctly)
                        # We increment uploaded/skipped counts later based on final status
                    else:
                        # upload_single_file returned False, indicating a non-AuthError failure
                        print(f"Failed processing {file_path.name}. Will not retry.")
                        # No break here, let the loop finish naturally after retries=0
                        pass

                    # If successful (or skipped), break the retry loop
                    if upload_successful:
                         break

                except AuthError as e:
                    print(f"Authentication error during upload attempt for {file_path.name}: {e}. Retrying after token refresh...")
                    retries -= 1
                    if retries <= 0:
                        print("ERROR: Max retries reached after authentication error.")
                        break # Exit retry loop

                    # Attempt token refresh before next retry
                    try:
                        credentials = refresh_access_token(credentials)
                        dbx = dropbox.Dropbox(credentials['access_token'])
                        print("Token refreshed successfully. Retrying upload...")
                    except Exception as refresh_err:
                        print(f"ERROR: Failed to refresh token during retry: {refresh_err}")
                        # Mark as failed and stop retrying for this file
                        upload_successful = False # Ensure it's marked as failed
                        break # Exit retry loop, cannot continue without token

                except ApiError as e:
                     # Handle other API errors (like rate limiting, storage quota)
                     print(f"ERROR: API error during upload for {file_path.name}: {e}")
                     # Decide if retry makes sense based on error type (e.g., rate limit vs storage)
                     # For simplicity, we don't retry on general API errors here
                     retries = 0 # No retry for general API errors
                     break # Exit retry loop

                except Exception as e:
                    print(f"ERROR: Unexpected error processing {file_path.name}: {e}")
                    import traceback
                    traceback.print_exc()
                    retries = 0 # No retry for unexpected errors
                    break # Exit retry loop

            # --- Update Counts After Retries ---
            if upload_successful:
                 # We need to differentiate between actual uploads and skips.
                 # Re-check existence post-facto (less efficient, but avoids complex returns from upload_single_file)
                 # Or, modify upload_single_file to return a status code (e.g., UPLOADED, SKIPPED, FAILED)
                 # For simplicity now, assume 'success' means uploaded unless explicitly skipped.
                 # This isn't perfect. A better approach is needed for accurate counts.
                 # Let's assume `upload_single_file` prints "Skipping" - we'll count failures accurately.
                 # Assume success = Uploaded or Skipped correctly.
                 # files_uploaded_count += 1 # Can't be sure if uploaded or skipped without better return code
                 pass # Logged success/skip in upload_single_file
            else:
                files_failed_count += 1

        # --- Final Summary ---
        print("\n--- Upload Summary ---")
        # print(f"Files Processed: {total_files}") # Redundant with "Found N files"
        # Accurate failed count is possible
        print(f"Files Failed: {files_failed_count}")
        # Cannot accurately report uploaded vs skipped without modification to upload_single_file return value
        # Print placeholder:
        print(f"Files Succeeded (Uploaded or Skipped): {total_files - files_failed_count}")

        if files_failed_count > 0:
            print(f"\nERROR: {files_failed_count} file(s) failed to process.")
            return ERROR_CODES['UPLOAD_ERROR']
        else:
            print("\nAll files processed successfully.")
            return 0 # Success

    except Exception as e:
        print(f"\nFATAL ERROR: An unexpected error occurred in the main process: {e}")
        import traceback
        traceback.print_exc()
        return ERROR_CODES['GENERAL_ERROR']

def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description='Upload files or directories to Dropbox with existence check and optional overwrite.',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter # Show defaults
    )
    parser.add_argument('--path', required=True, help='Path to the local file or directory to upload.')
    parser.add_argument(
        '--dest', default='',
        help='Destination directory path in Dropbox. If omitted, attempts to mirror source structure in root. Use "/" for root.'
    )
    parser.add_argument(
        '--file', action='store_true',
        help='Specify that the source --path points to a single file, not a directory.'
    )
    parser.add_argument(
        '--overwrite', action='store_true',
        help='Overwrite files in Dropbox even if they exist and have the same size.'
             ' Files with different sizes are always overwritten.'
    )

    return parser.parse_args()

if __name__ == "__main__":
    args = parse_args()
    result = upload_to_dropbox(
        path=args.path,
        dest_path=args.dest,
        is_file=args.file,
        overwrite=args.overwrite
    )
    sys.exit(result)
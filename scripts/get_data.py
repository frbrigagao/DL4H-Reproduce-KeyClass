import os
import boto3
from botocore.exceptions import NoCredentialsError, ClientError
from tqdm import tqdm
from zipfile import ZipFile

# Replace these with your Backblaze B2 credentials
KEY_ID = '00366006aa4022e000000000a'
APPLICATION_KEY = 'K003yFGks8sxQ0VcMVSIzmJOYB5tCd0'
BUCKET_NAME = 'cs598-dl4h-project'
ENDPOINT_URL = 'https://s3.eu-central-003.backblazeb2.com'  # Adjust if using a different region

# Initialize the S3 client for Backblaze B2
s3 = boto3.client(
    service_name='s3',
    endpoint_url=ENDPOINT_URL,
    aws_access_key_id=KEY_ID,
    aws_secret_access_key=APPLICATION_KEY
)

# List of files to download (file_key, subfolder_name)
files_to_download = [
    ('agnews_data.zip', 'datasets/agnews'),
    ('imdb_data.zip', 'datasets/imdb'),
    ('dbpedia_data.zip', 'datasets/dbpedia'),
    ('amazon_data.zip', 'datasets/amazon'),
    ('models_agnews_v1.0.zip', 'models'),
    ('models_dbpedia_v1.0.zip', 'models'),
    ('models_imdb_v.10.zip', 'models'),
    ('models_amazon_v1.0.zip', 'models'),
    ('results_v1.0.zip', 'results')
    # Add more files and their respective subfolders as needed
]

def download_file(file_key, destination_path):
    try:
        # Get the file size for progress tracking
        response = s3.head_object(Bucket=BUCKET_NAME, Key=file_key)
        file_size = response['ContentLength']

        # Ensure the destination directory exists
        os.makedirs(os.path.dirname(destination_path), exist_ok=True)

        # Download the file with progress bar
        with tqdm(total=file_size, unit='B', unit_scale=True, desc=file_key) as pbar:
            with open(destination_path, 'wb') as f:
                def progress(bytes_transferred):
                    pbar.update(bytes_transferred)

                s3.download_fileobj(BUCKET_NAME, file_key, f, Callback=progress)

        print(f"Downloaded '{file_key}' to '{destination_path}' successfully.")
        return True

    except NoCredentialsError:
        print("Credentials not available. Check your AWS configuration.")
    except ClientError as e:
        print(f"Failed to download '{file_key}': {e}")
    return False

def extract_zip(file_path, extract_to):
    try:
        with ZipFile(file_path, 'r') as zip_ref:
            zip_ref.extractall(extract_to)
        print(f"Extracted '{file_path}' to '{extract_to}' successfully.")
    except Exception as e:
        print(f"Failed to extract '{file_path}': {e}")

if __name__ == "__main__":
    # Determine the path to the 'original_datasets' directory (one level up from the script's directory)
    base_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'original_data'))

    for file_key, subfolder_name in files_to_download:
        user_input = input(f"Do you want to download and extract '{file_key}' to '{base_dir}/{subfolder_name}'? (yes/no): ").strip().lower()
        if user_input == 'yes' or user_input == 'y':
            # Define the destination path for the downloaded zip file
            destination_path = os.path.join(base_dir, subfolder_name, file_key)
            if download_file(file_key, destination_path):
                # Extract the zip file into its subfolder
                extract_zip(destination_path, os.path.join(base_dir, subfolder_name))
                # Optionally, remove the zip file after extraction
                os.remove(destination_path)
        else:
            print(f"Skipped '{file_key}'.")

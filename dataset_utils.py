import os

try:
    import wget
except ImportError:
    os.system('pip install wget')

import shutil
import zipfile
import wget

datasets_dir = 'datasets'

# Create the dataset folder if it doesn't exist
if not os.path.exists(datasets_dir):
    os.makedirs(datasets_dir)


def download_and_extract_dataset_zip(dataset_zip_url, required_files):
    # Check if the required files exist
    if all(os.path.exists(file) for file in required_files):
        return  # All required files exist.

    # Extract the dataset name from the URL
    dataset_zip_filename = os.path.basename(dataset_zip_url)

    # Download the dataset file if it doesn't exist
    dataset_zip_filepath = os.path.join(datasets_dir, dataset_zip_filename)
    if not os.path.exists(dataset_zip_filepath):
        # Download the dataset file
        print(f'Downloading {dataset_zip_filename}... ', end='', flush=True)
        wget.download(dataset_zip_url, out=dataset_zip_filepath)
        print('Downloaded.')

    # Extract the dataset file
    print(f'Extracting {dataset_zip_filename}... ', end='', flush=True)
    dataset_url_filepath = os.path.basename(dataset_zip_url)
    if dataset_url_filepath.endswith('.zip'):
        with zipfile.ZipFile(dataset_zip_filepath, 'r') as zip_ref:
            zip_ref.extractall(datasets_dir)
    elif dataset_url_filepath.endswith('.tar.gz'):
        shutil.unpack_archive(dataset_zip_filepath, extract_dir=datasets_dir, format='gztar')
    elif dataset_url_filepath.endswith('.tar.bz2'):
        shutil.unpack_archive(dataset_zip_filepath, extract_dir=datasets_dir, format='bztar')
    else:
        print(f"Unsupported file format: {dataset_url_filepath}")
        return
    print('Extracted. \n')

    # Remove the dataset file
    os.remove(dataset_zip_filepath)
'''
Usage Example:

from dataset_utils import download_and_extract_dataset_zip, datasets_dir

dataset_foldername = 'ml-latest-small'
zip_file_path = os.path.join(datasets_dir, dataset_foldername + '.zip')
extracted_folder_path = zip_file_path.replace('.zip', '')
ratings_file = os.path.join(extracted_folder_path, 'ratings.csv')

required_files = [ratings_file, ]  # add more files to the list if any
dataset_zip_url = f'https://files.grouplens.org/datasets/movielens/{dataset_foldername}.zip'

download_and_extract_dataset_zip(dataset_zip_url, required_files)

ratings = pd.read_csv(ratings_file)  # Load the file
'''


# Later, write the code to fetch from github or kaggle

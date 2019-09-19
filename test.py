from zipfile import ZipFile
import pandas as pd
import subprocess
import os

def main():
    fetchdata()
    data = pd.read_csv('train.csv')
    print(data.head())
    print(data.info())
    print(data.describe())


def fetchdata():
    cli_command = "kaggle competitions download -c cat-in-the-dat -w"
    subprocess.run(cli_command, shell=True) # Get data from kaggle
    zip_files = [files for files in os.listdir() if files.endswith('.zip')]
    # Unzip files if not already unzipped
    unzipped_filenames = [files.replace('.zip','') for files in zip_files]
    if not all(x in os.listdir() for x in unzipped_filenames):
        for file in zip_files:
            with ZipFile(file, 'r') as zip:
                zip.extractall()


if __name__ == '__main__':
    main()
import os
import glob
import kagglehub
import pandas as pd

def load_raw_data():
    """
    Downloads the dataset from Kaggle via kagglehub if not present,
    locates the target CSV file, and returns it as a pandas DataFrame.
    """
    print("Checking/Downloading dataset from Kaggle...")
    # Downloads the latest version of the target agriculture dataset
    download_path = kagglehub.dataset_download("ramkrijal/agriculture-vegetables-fruits-time-series-prices")
    
    # Locate the CSV inside the downloaded cache path
    csv_files = glob.glob(os.path.join(download_path, "*.csv"))
    if not csv_files:
        raise FileNotFoundError(f"No CSV files found in downloaded path: {download_path}")
        
    # The target file is 'kalimati_tarkari_dataset.csv'
    target_csv = csv_files[0]
    print(f"Loading data from: {target_csv}")
    
    # Load and clean baseline configurations
    df = pd.read_csv(target_csv)
    return df
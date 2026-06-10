import pandas as pd
import os
from . import config

def load_data(file_path=config.DATA_FILE):
    """
    Loads the ASAP dataset from the TSV file.
    """
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"Data file not found at {file_path}. Please download 'training_set_rel3.tsv' and place it in the 'data/' directory.")
    
    # The dataset is tab-separated and may have encoding issues, trying 'ISO-8859-1' is common for this dataset
    try:
        df = pd.read_csv(file_path, sep='\t', encoding='ISO-8859-1')
    except Exception as e:
        print(f"Error loading data: {e}")
        raise

    # We are primarily interested in 'essay_set', 'essay', and 'domain1_score'
    columns_to_keep = ['essay_set', 'essay', 'domain1_score']
    if not all(col in df.columns for col in columns_to_keep):
         raise ValueError(f"Dataset missing required columns: {columns_to_keep}")
         
    return df[columns_to_keep]

if __name__ == "__main__":
    try:
        df = load_data()
        print(f"Data loaded successfully. Shape: {df.shape}")
        print(df.head())
    except Exception as e:
        print(e)

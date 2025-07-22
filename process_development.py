import numpy as np
from remote_data import data_loader
from config import PROCESSED_DATA_DIR
import os

def process_development_set():
    """Process development set"""
    try:
        # Upload data
        X_dev, y_dev = data_loader.load_features('development', return_coords=False)
        
        # save data processed
        output_dir = os.path.join(PROCESSED_DATA_DIR, 'dev')
        os.makedirs(output_dir, exist_ok=True)
        
        np.save(os.path.join(output_dir, 'features.npy'), X_dev)
        np.save(os.path.join(output_dir, 'labels.npy'), y_dev)
        
    except Exception as e:
        print(f"Error during processes: {str(e)}")

if __name__ == "__main__":
    process_development_set()
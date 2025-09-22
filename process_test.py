import os
import numpy as np
from remote_data import data_loader
from config import PROCESSED_DATA_DIR

def process_test_set():
    """Process the test set using remote data"""
    try:
        # Upload data from FigShare
        X_test, y_test = data_loader.load_features('test', return_coords=False)
        
        # Create output directory if it doesn't exist
        output_dir = os.path.join(PROCESSED_DATA_DIR, 'test')
        os.makedirs(output_dir, exist_ok=True)
        
        # Save processed data as numpy files
        np.save(os.path.join(output_dir, 'X_test.npy'), X_test)
        np.save(os.path.join(output_dir, 'y_test.npy'), y_test)

    except Exception as e:
        print(f"Processing error: {e}")
        raise

if __name__ == "__main__":
    process_test_set()
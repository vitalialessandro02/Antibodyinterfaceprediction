import numpy as np
from remote_data import data_loader
from config import PROCESSED_DATA_DIR
import os

def process_training_set():
    """Process the training set using remote data"""
    try:
       #Upload data
        X_train, y_train = data_loader.load_features('training', return_coords=False)
        
        # Save processed data 
        output_dir = os.path.join(PROCESSED_DATA_DIR, 'train')
        os.makedirs(output_dir, exist_ok=True)
        
        np.save(os.path.join(output_dir, 'features.npy'), X_train)
        np.save(os.path.join(output_dir, 'labels.npy'), y_train)
        
        
        
    except Exception as e:
        print(f"Error durante il processing: {str(e)}")
        raise

if __name__ == "__main__":
    process_training_set()
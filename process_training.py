import os
import numpy as np
from remote_data import data_loader
from config import PROCESSED_DATA_DIR
def process_training_set():
    """Processa il set di training usando dati remoti"""
    # Carica i dati direttamente da FigShare
    X_train, y_train = data_loader.load_features('training')
    
    # Salva solo i dati processati (non i raw)
    output_dir = os.path.join(PROCESSED_DATA_DIR, 'train')
    os.makedirs(output_dir, exist_ok=True)
    
    np.save(os.path.join(output_dir, 'features.npy'), X_train)
    np.save(os.path.join(output_dir, 'labels.npy'), y_train)
    
    print(f"Training set processed: {X_train.shape[0]} samples, {X_train.shape[1]} features")

if __name__ == "__main__":
    process_training_set()
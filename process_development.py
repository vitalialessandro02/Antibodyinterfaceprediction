import numpy as np
from remote_data import data_loader
from config import PROCESSED_DATA_DIR
import os

def process_development_set():
    """Processa il set di sviluppo usando dati in streaming"""
    try:
        # Carica i dati senza coordinate
        X_dev, y_dev = data_loader.load_features('development', return_coords=False)
        
        # Salva solo i risultati processati
        output_dir = os.path.join(PROCESSED_DATA_DIR, 'dev')
        os.makedirs(output_dir, exist_ok=True)
        
        np.save(os.path.join(output_dir, 'features.npy'), X_dev)
        np.save(os.path.join(output_dir, 'labels.npy'), y_dev)
        
        print(f"Development set processato: {X_dev.shape[0]} campioni, {X_dev.shape[1]} features")
        print(f"Dati salvati in: {output_dir}")
        
    except Exception as e:
        print(f"Errore durante il processing: {str(e)}")

if __name__ == "__main__":
    process_development_set()
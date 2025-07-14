import os
import numpy as np
from remote_data import data_loader
from config import PROCESSED_DATA_DIR

def process_test_set():
    """Processa il set di test usando dati remoti"""
    try:
        # Carica i dati direttamente da FigShare (senza coordinate)
        X_test, y_test = data_loader.load_features('test', return_coords=False)
        
        # Salva i dati processati
        output_dir = os.path.join(PROCESSED_DATA_DIR, 'test')
        os.makedirs(output_dir, exist_ok=True)
        
        np.save(os.path.join(output_dir, 'X_test.npy'), X_test)
        np.save(os.path.join(output_dir, 'y_test.npy'), y_test)
        
        print(f"Test set processato: {X_test.shape[0]} campioni, {X_test.shape[1]} features")
        print(f"Dati salvati in: {output_dir}")
        
        # Mostra la distribuzione delle etichette
        unique, counts = np.unique(y_test, return_counts=True)
        print(f"Distribuzione etichette: {dict(zip(unique, counts))}")
        
    except Exception as e:
        print(f"Errore durante il processing: {e}")
        raise

if __name__ == "__main__":
    process_test_set()
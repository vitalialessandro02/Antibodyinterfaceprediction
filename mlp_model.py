import os
import time

import logging.config  
import numpy as np
from sklearn.neural_network import MLPClassifier
from sklearn.model_selection import cross_val_score, StratifiedKFold
from sklearn.utils import resample
from joblib import dump
from config import (PROCESSED_DATA_DIR, MODEL_DIR, MLP_PARAMS, LOGGING, 
                   OPTIMAL_THRESHOLD, RESULTS_DIR)
import matplotlib.pyplot as plt


logging.config.dictConfig(LOGGING)
logger = logging.getLogger('training')

def train_mlp():
    """Train and evaluate MLP model using cross-validation"""
    try:
        X_train, y_train = load_processed_data()
        
        mlp = MLPClassifier(**MLP_PARAMS)
        
        # Final training
        start_time = time.time()
        mlp.fit(X_train, y_train)
        logger.info(f"MLP training completed in {time.time() - start_time:.2f}s")
        
        # Save model
        os.makedirs(MODEL_DIR, exist_ok=True)
        model_path = os.path.join(MODEL_DIR, 'trained_mlp_model.joblib')
        dump(mlp, model_path)
        
        return mlp
        
    except Exception as e:
        logger.error(f"MLP training error: {str(e)}")
        raise

def load_processed_data():
    """Load and balance training data from processed files"""
    try:
        X_train = np.load(os.path.join(PROCESSED_DATA_DIR, 'train', 'features.npy'))
        y_train = np.load(os.path.join(PROCESSED_DATA_DIR, 'train', 'labels.npy'))
        
        # Balance dataset
        mask_minority = y_train == 1
        X_minority = X_train[mask_minority]
        y_minority = y_train[mask_minority]
        X_majority = X_train[~mask_minority]
        y_majority = y_train[~mask_minority]
        
        n_samples = min(len(X_majority), 2 * len(X_minority))
        X_majority_down, y_majority_down = resample(
            X_majority, y_majority,
            replace=False,
            n_samples=n_samples,
            random_state=42
        )
        
        X_train_balanced = np.vstack([X_majority_down, X_minority])
        y_train_balanced = np.concatenate([y_majority_down, y_minority])
        
        return X_train_balanced, y_train_balanced
        
    except Exception as e:
        logger.error(f"Data loading error: {str(e)}")
        raise

if __name__ == "__main__":
    train_mlp()
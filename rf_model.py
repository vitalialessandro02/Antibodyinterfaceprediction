import os
import time
import logging.config  
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import cross_val_score, StratifiedKFold
from sklearn.utils import resample
from joblib import dump
from config import (PROCESSED_DATA_DIR, MODEL_DIR, RF_PARAMS, LOGGING, 
                   OPTIMAL_THRESHOLD, RESULTS_DIR)
import matplotlib.pyplot as plt


logging.config.dictConfig(LOGGING)
logger = logging.getLogger('training')

def train_rf():
    """Train and evaluate Random Forest model using cross-validation"""
    try:
        
        X_train, y_train = load_processed_data()
        rf = RandomForestClassifier(**RF_PARAMS)
       
        
        # Feature importance
        rf.fit(X_train, y_train)
        importances = rf.feature_importances_
        indices = np.argsort(importances)[::-1]
        
        # Plot feature importance
        plt.figure(figsize=(12, 6))
        plt.title("Feature Importances")
        plt.bar(range(20), importances[indices[:20]], align='center')
        plt.xticks(range(20), indices[:20])
        plt.xlim([-1, 20])
        plt.tight_layout()
        plt.savefig(os.path.join(RESULTS_DIR, 'rf_feature_importance.png'))
        plt.close()
        
        # Save model
        os.makedirs(MODEL_DIR, exist_ok=True)
        model_path = os.path.join(MODEL_DIR, 'trained_rf_model.joblib')
        dump(rf, model_path)
        
        return rf
        
    except Exception as e:
        logger.error(f"RF training error: {str(e)}")
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
    train_rf()
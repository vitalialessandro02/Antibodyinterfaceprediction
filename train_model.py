import os
import time
import logging
import logging.config
import numpy as np
from sklearn.svm import SVC
from joblib import dump
from sklearn.model_selection import cross_val_score, StratifiedKFold
from sklearn.utils import resample
from config import (PROCESSED_DATA_DIR, MODEL_DIR, SVM_PARAMS, LOGGING, 
                   OPTIMAL_THRESHOLD, PHYSICOCHEMICAL_PROPERTIES)
from remote_data import data_loader
from zernike_descriptors import ProteinSurface, compute_3dzd, extract_patch


logging.config.dictConfig(LOGGING)
logger = logging.getLogger('training')

class AntibodyDataset:
    def __init__(self):
        self.X = None
        self.y = None
        self.coords = None
        
    def load_from_remote(self, dataset_type):
     """Load dataset from remote source"""
     logger.info(f"Loading {dataset_type} dataset from remote source...")
    
    # Modified to handle both cases (with and without coordinates)
     try:
        result = data_loader.load_features(dataset_type)
        if len(result) == 3:
            X, y, coords = result
            self.coords = coords
        else:
            X, y = result
            self.coords = None  # Or generate dummy coordinates if needed
            
        self.X = X
        self.y = y
        
        logger.info(f"Loaded {len(y)} samples with {X.shape[1]} features")
        self._log_class_distribution()
     except Exception as e:
        logger.error(f"Error loading dataset: {str(e)}")
        raise
        
    def balance_dataset(self):
        """Balance the dataset using a combination of under and oversampling"""
        logger.info("Balancing dataset...")
        
        # Separate majority and minority classes
        mask_minority = self.y == 1
        X_minority = self.X[mask_minority]
        y_minority = self.y[mask_minority]
        X_majority = self.X[~mask_minority]
        y_majority = self.y[~mask_minority]
        
        # Undersample majority class
        n_majority = len(X_majority)
        n_minority = len(X_minority)
        undersample_factor = 2  # Reduce majority class to 2x minority
        n_samples = min(n_majority, n_minority * undersample_factor)
        
        X_majority_down, y_majority_down = resample(
            X_majority, y_majority,
            replace=False,
            n_samples=n_samples,
            random_state=42
        )
        
        # Combine with minority class
        self.X = np.vstack([X_majority_down, X_minority])
        self.y = np.concatenate([y_majority_down, y_minority])
        
        logger.info("Dataset balanced")
        self._log_class_distribution()
        
    def _log_class_distribution(self):
        """Log class distribution information"""
        unique, counts = np.unique(self.y, return_counts=True)
        dist = dict(zip(unique, counts))
        logger.info(f"Class distribution: {dist}")
        if len(counts) > 1:
            logger.info(f"Positive class ratio: {counts[1]/sum(counts):.2%}")

def train_model():
    """Training   SVM  model """
    try:
        # Upload and balanced data 
        X_train, y_train = load_processed_data()
        
        # Create  SVM model with 
        logger.info("Creation of SVM model")
        svm = SVC(**SVM_PARAMS)
        
        # Cross-validazione stratificated  
        logger.info("Cross-validazione 10-fold...")
        cv = StratifiedKFold(n_splits=10, shuffle=True, random_state=42)
        cv_scores = cross_val_score(
            svm, X_train, y_train, 
            cv=cv, 
            scoring='roc_auc',
            n_jobs=-1
        )
        
        logger.info(f"ROC AUC medio: {np.mean(cv_scores):.4f} ± {np.std(cv_scores):.4f}")
        
        # Final traing on full  il dataset
        logger.info("Final Training ")
        start_time = time.time()
        svm.fit(X_train, y_train)
        logger.info(f"Training  complete in {time.time() - start_time:.2f}s")
        
        # Save model
        os.makedirs(MODEL_DIR, exist_ok=True)
        model_path = os.path.join(MODEL_DIR, 'trained_svm_model.joblib')
        dump(svm, model_path)
        logger.info(f"Model save  in {model_path}")
        
        return svm
        
    except Exception as e:
        logger.error(f"Errore nell'addestramento: {str(e)}")
        raise
def load_processed_data():
    
    try:
        X_train = np.load(os.path.join(PROCESSED_DATA_DIR, 'train', 'features.npy'))
        y_train = np.load(os.path.join(PROCESSED_DATA_DIR, 'train', 'labels.npy'))
        
        logger.info(f"Upload {X_train.shape[0]} samples with {X_train.shape[1]} features")
        
        # Bilancedc data  (undersampling + oversampling)
        mask_minority = y_train == 1
        X_minority = X_train[mask_minority]
        y_minority = y_train[mask_minority]
        X_majority = X_train[~mask_minority]
        y_majority = y_train[~mask_minority]
        
        # Undersampling della classe maggioritaria
        n_samples = min(len(X_majority), 2 * len(X_minority))  # rapport 2:1 
        X_majority_down, y_majority_down = resample(
            X_majority, y_majority,
            replace=False,
            n_samples=n_samples,
            random_state=42
        )
        
        # Combina i dataset
        X_train_balanced = np.vstack([X_majority_down, X_minority])
        y_train_balanced = np.concatenate([y_majority_down, y_minority])
        
        logger.info(f"Dimensione dataset bilanciato: {len(y_train_balanced)}")
        logger.info(f"Distribuzione classi: {np.bincount(y_train_balanced.astype(int))}")
        
        return X_train_balanced, y_train_balanced
        
    except Exception as e:
        logger.error(f"Error during data uploading: {str(e)}")
        raise
if __name__ == "__main__":
    train_model()
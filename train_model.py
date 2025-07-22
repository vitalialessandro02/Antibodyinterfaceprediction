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
import warnings
from sklearn.exceptions import ConvergenceWarning


warnings.filterwarnings("ignore", category=ConvergenceWarning)
warnings.filterwarnings("ignore", category=UserWarning)

logging.config.dictConfig(LOGGING)
logger = logging.getLogger('training')

class AntibodyDataset:
    def __init__(self):
        """Initialize dataset container"""
        self.X = None
        self.y = None
        self.coords = None
        
    def load_from_remote(self, dataset_type):
        """Load dataset from remote source"""
        try:
            result = data_loader.load_features(dataset_type)
            if len(result) == 3:
                X, y, coords = result
                self.coords = coords
            else:
                X, y = result
                self.coords = None 
                
            self.X = X
            self.y = y
            self._log_class_distribution()
        except Exception as e:
            logger.error(f"Error loading dataset: {str(e)}")
            raise
            
    def balance_dataset(self):
        """Balance the dataset using a combination of under and oversampling"""
        mask_minority = self.y == 1
        X_minority = self.X[mask_minority]
        y_minority = self.y[mask_minority]
        X_majority = self.X[~mask_minority]
        y_majority = self.y[~mask_minority]
        
        n_majority = len(X_majority)
        n_minority = len(X_minority)
        undersample_factor = 2 
        n_samples = min(n_majority, n_minority * undersample_factor)
        
        X_majority_down, y_majority_down = resample(
            X_majority, y_majority,
            replace=False,
            n_samples=n_samples,
            random_state=42
        )
        
        self.X = np.vstack([X_majority_down, X_minority])
        self.y = np.concatenate([y_majority_down, y_minority])
        self._log_class_distribution()
        
    def _log_class_distribution(self):
        """Log class distribution information"""
        unique, counts = np.unique(self.y, return_counts=True)
        dist = dict(zip(unique, counts))
        if len(counts) > 1:
            logger.info(f"Positive class ratio: {counts[1]/sum(counts):.2%}")

def train_model():
    """Training and evaluate SVM model using cross-validation"""
    try:
        X_train, y_train = load_processed_data()
        
        # Modifica i parametri SVM per disabilitare l'output
        svm_params = SVM_PARAMS.copy()
        svm_params['verbose'] = False  # Disabilita output di LibSVM
        
        svm = SVC(**svm_params)
        cv = StratifiedKFold(n_splits=10, shuffle=True, random_state=42)
        
        # Disabilita output durante cross-validation
        cv_scores = cross_val_score(
            svm, X_train, y_train, 
            cv=cv, 
            scoring='roc_auc',
            n_jobs=-1,
            verbose=0  # Disabilita output di cross_val_score
        )
        
        # Training finale
        start_time = time.time()
        svm.fit(X_train, y_train)
        logger.info(f"Training completed in {time.time() - start_time:.2f}s")
        
        os.makedirs(MODEL_DIR, exist_ok=True)
        model_path = os.path.join(MODEL_DIR, 'trained_svm_model.joblib')
        dump(svm, model_path)
        
        
        return svm
        
    except Exception as e:
        logger.error(f"Training error: {str(e)}")
        raise

def load_processed_data():
    """Load and balance training data from processed files"""
    try:
        X_train = np.load(os.path.join(PROCESSED_DATA_DIR, 'train', 'features.npy'))
        y_train = np.load(os.path.join(PROCESSED_DATA_DIR, 'train', 'labels.npy'))
        
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
    train_model()
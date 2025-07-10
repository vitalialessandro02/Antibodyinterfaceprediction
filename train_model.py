import os
import time
import logging
import logging.config
import numpy as np
from sklearn.svm import SVC
from joblib import dump
from sklearn.model_selection import cross_val_score
from config import PROCESSED_DATA_DIR, MODEL_DIR, SVM_PARAMS, LOGGING, OPTIMAL_THRESHOLD
from remote_data import data_loader
# Configura il logging
logging.config.dictConfig(LOGGING)
logger = logging.getLogger('training')

class ProgressLogger:
    def __init__(self, total_iterations):
        self.start_time = time.time()
        self.total_iter = total_iterations
        self.last_logged = 0
        
    def log_progress(self, iteration):
        if time.time() - self.last_logged > 30:  # Log ogni 30 secondi
            elapsed = time.time() - self.start_time
            remaining = (elapsed / (iteration + 1)) * (self.total_iter - iteration - 1)
            logger.info(
                f"Progress: {iteration + 1}/{self.total_iter} "
                f"({(iteration + 1)/self.total_iter:.1%}) | "
                f"Elapsed: {elapsed:.0f}s | "
                f"Remaining: {remaining:.0f}s"
            )
            self.last_logged = time.time()

def train_model():
    # Carica i dati di training
    logger.info("Loading training data from remote source...")
    X_train, y_train = data_loader.load_features('training')
    logger.info(f"Loaded {X_train.shape[0]} samples with {X_train.shape[1]} features")
    
    # Distribuzione delle classi
    unique, counts = np.unique(y_train, return_counts=True)
    logger.info(f"Class distribution: {dict(zip(unique, counts))}")
    logger.info(f"Positive class ratio: {counts[1]/sum(counts):.2%}")

    # Crea il modello SVM con i parametri ottimali dal paper
    logger.info("Creating SVM model with optimal parameters from the paper")
    svm = SVC(**SVM_PARAMS)
    
    # Valutazione cross-validata
    logger.info("Performing cross-validation...")
    cv_scores = cross_val_score(svm, X_train, y_train, cv=5, scoring='roc_auc')
    logger.info(f"Cross-validation ROC AUC scores: {cv_scores}")
    logger.info(f"Mean ROC AUC: {np.mean(cv_scores):.4f} ± {np.std(cv_scores):.4f}")

    # Addestramento finale
    logger.info("Starting final training on full dataset...")
    svm.fit(X_train, y_train)
    logger.info("Training completed!")
    
    # Salva il modello
    os.makedirs(MODEL_DIR, exist_ok=True)
    model_path = os.path.join(MODEL_DIR, 'trained_svm_model.joblib')
    dump(svm, model_path)
    logger.info(f"Model saved to {model_path}")

if __name__ == "__main__":
    train_model()
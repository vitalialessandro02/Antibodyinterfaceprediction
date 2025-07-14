# test_model.py
import os
import numpy as np
from joblib import load
from sklearn.metrics import (roc_auc_score, average_precision_score, 
                           precision_score, recall_score, f1_score, 
                           accuracy_score, confusion_matrix)
from sklearn.metrics import roc_curve, precision_recall_curve
from sklearn.ensemble import IsolationForest
import matplotlib.pyplot as plt
import seaborn as sns
from config import MODEL_DIR, OPTIMAL_THRESHOLD, IF_CONTAMINATION, RESULTS_DIR
import time

from remote_data import data_loader

# Configura lo stile dei grafici
plt.style.use('seaborn-v0_8')
sns.set_palette('colorblind')

def save_metrics(metrics, filename):
    """Salva le metriche in un file"""
    with open(os.path.join(RESULTS_DIR, filename), 'w') as f:
        for k, v in metrics.items():
            f.write(f"{k}: {v:.4f}\n")

def plot_curves(y_true, y_score, set_name):
    """Crea e salva le curve ROC e PR"""
    # ROC Curve
    fpr, tpr, _ = roc_curve(y_true, y_score)
    roc_auc = roc_auc_score(y_true, y_score)
    
    plt.figure(figsize=(10, 8))
    plt.plot(fpr, tpr, label=f'ROC (AUC = {roc_auc:.3f})')
    plt.plot([0, 1], [0, 1], 'k--')
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title(f'ROC Curve - {set_name}')
    plt.legend()
    plt.savefig(os.path.join(RESULTS_DIR, f'roc_{set_name}.png'))
    plt.close()
    
    # PR Curve
    precision, recall, _ = precision_recall_curve(y_true, y_score)
    pr_auc = average_precision_score(y_true, y_score)
    
    plt.figure(figsize=(10, 8))
    plt.plot(recall, precision, label=f'PR (AUC = {pr_auc:.3f})')
    plt.xlabel('Recall')
    plt.ylabel('Precision')
    plt.title(f'PR Curve - {set_name}')
    plt.legend()
    plt.savefig(os.path.join(RESULTS_DIR, f'pr_{set_name}.png'))
    plt.close()

def evaluate_model(model, X, y, set_name):
    """Valuta il modello e applica il post-processing"""
    # Calcola le decisioni
    y_score = model.decision_function(X)
    y_pred = (y_score >= OPTIMAL_THRESHOLD).astype(int)
    
    # Metriche base
    metrics = {
        'ROC AUC': roc_auc_score(y, y_score),
        'PR AUC': average_precision_score(y, y_score),
        'Accuracy': accuracy_score(y, y_pred),
        'Precision': precision_score(y, y_pred),
        'Recall': recall_score(y, y_pred),
        'F1 Score': f1_score(y, y_pred)
    }
    
    # Salva le curve
    plot_curves(y, y_score, set_name)
    save_metrics(metrics, f'metrics_{set_name}.txt')
    
    print(f"\nPerformance sul {set_name} set:")
    print("="*50)
    for name, value in metrics.items():
        print(f"- {name}: {value:.4f}")
    
    return metrics

def test_model():
    """Testa il modello sui set di sviluppo e test"""
    try:
        # Carica il modello
        model_path = os.path.join(MODEL_DIR, 'trained_svm_model.joblib')
        if not os.path.exists(model_path):
            raise FileNotFoundError(f"Modello non trovato in {model_path}")
        
        svm = load(model_path)
        
        # Valuta sul set di sviluppo (se disponibile)
        try:
            print("\nValutazione sul set di sviluppo...")
            X_dev, y_dev = data_loader.load_features('development')
            evaluate_model(svm, X_dev, y_dev, "development")
        except Exception as e:
            print(f"\nAttenzione: impossibile valutare sul set di sviluppo: {e}")
        
        # Valuta sul set di test con più tentativi
        max_attempts = 3
        for attempt in range(max_attempts):
            try:
                print(f"\nTentativo {attempt + 1} di {max_attempts} - Valutazione sul set di test...")
                X_test, y_test = data_loader.load_features('test')
                evaluate_model(svm, X_test, y_test, "test")
                break
            except Exception as e:
                if attempt == max_attempts - 1:
                    raise
                print(f"Tentativo fallito: {e}\nRiprovo...")
                time.sleep(5 * (attempt + 1))
                
    except Exception as e:
        print(f"\nERRORE CRITICO durante il testing: {str(e)}")
        raise

if __name__ == "__main__":
    os.makedirs(RESULTS_DIR, exist_ok=True)
    test_model()
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


plt.style.use('seaborn-v0_8')
sns.set_palette('colorblind')

def save_metrics(metrics, filename):
    """Salva le metriche in un file"""
    with open(os.path.join(RESULTS_DIR, filename), 'w') as f:
        for k, v in metrics.items():
            f.write(f"{k}: {v:.4f}\n")

def plot_curves(y_true, y_score, set_name):
    """Crea e salva le curve ROC, PR e le metriche al variare della soglia"""
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
    
  
    thresholds = np.linspace(min(y_score), max(y_score), 100)
    accuracies = []
    f1_scores = []
    
    for thresh in thresholds:
        y_pred = (y_score >= thresh).astype(int)
        accuracies.append(accuracy_score(y_true, y_pred))
        f1_scores.append(f1_score(y_true, y_pred))
    
    # Grafic for Accuracy vs Threshold
    plt.figure(figsize=(10, 8))
    plt.plot(thresholds, accuracies, label='Accuracy', color='blue')
    plt.axvline(x=OPTIMAL_THRESHOLD, color='red', linestyle='--', 
                label=f'Optimal Threshold ({OPTIMAL_THRESHOLD:.3f})')
    plt.xlabel('Threshold')
    plt.ylabel('Accuracy')
    plt.title(f'Accuracy vs Decision Threshold - {set_name}')
    plt.legend()
    plt.savefig(os.path.join(RESULTS_DIR, f'accuracy_vs_threshold_{set_name}.png'))
    plt.close()
    
    # Grafic for F1 Score vs Threshold
    plt.figure(figsize=(10, 8))
    plt.plot(thresholds, f1_scores, label='F1 Score', color='green')
    plt.axvline(x=OPTIMAL_THRESHOLD, color='red', linestyle='--', 
                label=f'Optimal Threshold ({OPTIMAL_THRESHOLD:.3f})')
    plt.xlabel('Threshold')
    plt.ylabel('F1 Score')
    plt.title(f'F1 Score vs Decision Threshold - {set_name}')
    plt.legend()
    plt.savefig(os.path.join(RESULTS_DIR, f'f1_vs_threshold_{set_name}.png'))
    plt.close()

def evaluate_model(model, X, y, set_name):
    """evaluate the  modello and  apply  post-processing"""
    # Calcolate the decision
    y_score = model.decision_function(X)
    y_pred = (y_score >= OPTIMAL_THRESHOLD).astype(int)
    
    # base metrics
    metrics = {
        'ROC AUC': roc_auc_score(y, y_score),
        'PR AUC': average_precision_score(y, y_score),
        'Accuracy': accuracy_score(y, y_pred),
        'Precision': precision_score(y, y_pred),
        'Recall': recall_score(y, y_pred),
        'F1 Score': f1_score(y, y_pred)
    }
    
    # Save the   curve
    plot_curves(y, y_score, set_name)
    save_metrics(metrics, f'metrics_{set_name}.txt')
    
    print(f"\nPerformance on  {set_name} set:")
    print("="*50)
    for name, value in metrics.items():
        print(f"- {name}: {value:.4f}")
    
    return metrics

def test_model():
    """Testing   modello on development and testing  set """
    try:
        # Upload the method
        model_path = os.path.join(MODEL_DIR, 'trained_svm_model.joblib')
        if not os.path.exists(model_path):
            raise FileNotFoundError(f"Modello not found in {model_path}")
        
        svm = load(model_path)
        
       
        try:
            print("\nValutation of developmnet   set ...")
            X_dev, y_dev = data_loader.load_features('development')
            evaluate_model(svm, X_dev, y_dev, "development")
        except Exception as e:
            print(f"\nWarning : impossibile evaluate on development set : {e}")
        
        # With more 3 attemps
        max_attempts = 3
        for attempt in range(max_attempts):
            try:
                print(f"\nAttemps {attempt + 1} on {max_attempts} - Evaluete on test set...")
                X_test, y_test = data_loader.load_features('test')
                evaluate_model(svm, X_test, y_test, "test")
                break
            except Exception as e:
                if attempt == max_attempts - 1:
                    raise
                print(f"Faulty attemps: {e}\nRitry...")
                time.sleep(5 * (attempt + 1))
                
    except Exception as e:
        print(f"\nCritic ERROR  during testing: {str(e)}")
        raise

if __name__ == "__main__":
    os.makedirs(RESULTS_DIR, exist_ok=True)
    test_model()
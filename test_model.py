import os
import numpy as np
from joblib import load
from sklearn.metrics import (accuracy_score, roc_auc_score, average_precision_score, 
                           precision_score, recall_score, f1_score, confusion_matrix)
from sklearn.metrics import roc_curve, precision_recall_curve, auc
from sklearn.ensemble import IsolationForest
import matplotlib.pyplot as plt
import seaborn as sns
from remote_data import data_loader
from config import MODEL_DIR, OPTIMAL_THRESHOLD, IF_CONTAMINATION, BASE_DIR


# Crea directory per i risultati
RESULTS_DIR = os.path.join(BASE_DIR, 'results')
os.makedirs(RESULTS_DIR, exist_ok=True)

def save_metrics_to_file(metrics, set_name):
    """Salva le metriche in un file di testo"""
    filename = os.path.join(RESULTS_DIR, f'metrics_{set_name}.txt')
    with open(filename, 'w') as f:
        for key, value in metrics.items():
            f.write(f"{key}: {value:.4f}\n")

def plot_roc_curve(y_true, y_score, set_name, post_process=False):
    fpr, tpr, _ = roc_curve(y_true, y_score)
    roc_auc = auc(fpr, tpr)
    
    plt.figure(figsize=(10, 8))
    plt.plot(fpr, tpr, color='darkorange', lw=2, 
             label=f'ROC curve (AUC = {roc_auc:.3f})')
    plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate', fontsize=12)
    plt.ylabel('True Positive Rate', fontsize=12)
    title = f'ROC Curve - {set_name}'
    if post_process:
        title += ' (Post-processed)'
    plt.title(title, fontsize=14)
    plt.legend(loc="lower right", fontsize=12)
    plt.grid()
    
    filename = f'roc_curve_{set_name}'
    if post_process:
        filename += '_processed'
    plt.savefig(os.path.join(RESULTS_DIR, f'{filename}.png'), dpi=300, bbox_inches='tight')
    plt.close()

def plot_pr_curve(y_true, y_score, set_name, post_process=False):
    precision, recall, _ = precision_recall_curve(y_true, y_score)
    pr_auc = auc(recall, precision)
    
    plt.figure(figsize=(10, 8))
    plt.plot(recall, precision, color='blue', lw=2, 
             label=f'PR curve (AUC = {pr_auc:.3f})')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('Recall', fontsize=12)
    plt.ylabel('Precision', fontsize=12)
    title = f'Precision-Recall Curve - {set_name}'
    if post_process:
        title += ' (Post-processed)'
    plt.title(title, fontsize=14)
    plt.legend(loc="upper right", fontsize=12)
    plt.grid()
    
    filename = f'pr_curve_{set_name}'
    if post_process:
        filename += '_processed'
    plt.savefig(os.path.join(RESULTS_DIR, f'{filename}.png'), dpi=300, bbox_inches='tight')
    plt.close()

def plot_confusion_matrix(y_true, y_pred, set_name, post_process=False):
    cm = confusion_matrix(y_true, y_pred)
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                xticklabels=['Non-Interface', 'Interface'],
                yticklabels=['Non-Interface', 'Interface'])
    title = f'Confusion Matrix - {set_name}'
    if post_process:
        title += ' (Post-processed)'
    plt.title(title, fontsize=14)
    plt.xlabel('Predicted', fontsize=12)
    plt.ylabel('Actual', fontsize=12)
    
    filename = f'confusion_matrix_{set_name}'
    if post_process:
        filename += '_processed'
    plt.savefig(os.path.join(RESULTS_DIR, f'{filename}.png'), dpi=300, bbox_inches='tight')
    plt.close()

def evaluate_model(model, X, y, set_name):
    # Calcola le decisioni e le probabilità
    y_score = model.decision_function(X)
    
    # Applica la soglia ottimale
    y_pred = (y_score >= OPTIMAL_THRESHOLD).astype(int)
    
    # Post-processing con Isolation Forest
    np.random.seed(42)
    patch_coords = np.random.rand(len(X), 3) * 10  # Coordinate fittizie
    
    iso_forest = IsolationForest(contamination=IF_CONTAMINATION, random_state=42)
    inliers = iso_forest.fit_predict(patch_coords)
    y_score_processed = y_score.copy()
    y_score_processed[inliers == -1] = 0  # Elimina outlier
    
    # Metriche senza post-processing
    metrics = {
        'ROC AUC': roc_auc_score(y, y_score),
        'PR AUC': average_precision_score(y, y_score),
        'Accuracy': accuracy_score(y, y_pred),
        'Precision': precision_score(y, y_pred),
        'Recall': recall_score(y, y_pred),
        'F1 Score': f1_score(y, y_pred)
    }
    
    # Metriche con post-processing
    y_pred_processed = (y_score_processed >= OPTIMAL_THRESHOLD).astype(int)
    metrics_processed = {
        'ROC AUC': roc_auc_score(y, y_score_processed),
        'PR AUC': average_precision_score(y, y_score_processed),
        'Accuracy': accuracy_score(y, y_pred_processed),
        'Precision': precision_score(y, y_pred_processed),
        'Recall': recall_score(y, y_pred_processed),
        'F1 Score': f1_score(y, y_pred_processed)
    }
    
    # Plot e salvataggio risultati
    plot_roc_curve(y, y_score, set_name)
    plot_pr_curve(y, y_score, set_name)
    plot_confusion_matrix(y, y_pred, set_name)
    
    plot_roc_curve(y, y_score_processed, set_name, post_process=True)
    plot_pr_curve(y, y_score_processed, set_name, post_process=True)
    plot_confusion_matrix(y, y_pred_processed, set_name, post_process=True)
    
    save_metrics_to_file(metrics, set_name)
    save_metrics_to_file(metrics_processed, f"{set_name}_processed")
    
    # Stampa i risultati
    print(f"\nPerformance on {set_name} set:")
    print("="*50)
    print("Before post-processing:")
    for name, value in metrics.items():
        print(f"- {name}: {value:.4f}")
    
    print("\nAfter post-processing:")
    for name, value in metrics_processed.items():
        print(f"- {name}: {value:.4f}")

def test_model():
    """Testa il modello sui set di validazione e test"""
    # Carica il modello
    model_path = os.path.join(MODEL_DIR, 'trained_svm_model.joblib')
    svm = load(model_path)
    
    # Valuta sul set di sviluppo (se disponibile)
    try:
        X_dev, y_dev = data_loader.load_features('development')
        evaluate_model(svm, X_dev, y_dev, "development")
    except Exception as e:
        print(f"Could not load development set: {e}")
    
    # Valuta sul set di test
    X_test, y_test = data_loader.load_features('test')
    evaluate_model(svm, X_test, y_test, "test")

if __name__ == "__main__":
    test_model()
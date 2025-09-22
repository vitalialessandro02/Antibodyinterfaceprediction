import os
import numpy as np
from joblib import load
from sklearn.metrics import (roc_auc_score, average_precision_score, 
                           precision_score, recall_score, f1_score, 
                           accuracy_score)
from sklearn.metrics import roc_curve, precision_recall_curve
from config import MODEL_DIR, OPTIMAL_THRESHOLD, RESULTS_DIR
import matplotlib.pyplot as plt
import seaborn as sns
from remote_data import data_loader

plt.style.use('seaborn-v0_8')
sns.set_palette('colorblind')

def evaluate_mlp(model, X, y, set_name):
    """Evaluate MLP model performance"""
    # Get predicted probabilities
    y_score = model.predict_proba(X)[:, 1]
    y_pred = (y_score >= OPTIMAL_THRESHOLD).astype(int)
    
    # Compute metrics
    metrics = {
        'ROC AUC': roc_auc_score(y, y_score),
        'PR AUC': average_precision_score(y, y_score),
        'Accuracy': accuracy_score(y, y_pred),
        'Precision': precision_score(y, y_pred),
        'Recall': recall_score(y, y_pred),
        'F1 Score': f1_score(y, y_pred)
    }
    
    # Generate plots (same as original test_model.py)
    plot_curves(y, y_score, set_name)
    save_metrics(metrics, f'mlp_metrics_{set_name}.txt')
    
    print(f"\nMLP Performance on {set_name} set:")
    print("="*50)
    for name, value in metrics.items():
        print(f"- {name}: {value:.4f}")
    
    return metrics

def save_metrics(metrics, filename):
    """Save evaluation metrics to a text file"""
    with open(os.path.join(RESULTS_DIR, filename), 'w') as f:
        for k, v in metrics.items():
            f.write(f"{k}: {v:.4f}\n")

def plot_curves(y_true, y_score, set_name):
    """Generate and save ROC, PR curves and threshold-based metrics plots"""
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
    plt.savefig(os.path.join(RESULTS_DIR, f' MLP_roc_{set_name}.png'))
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
    plt.savefig(os.path.join(RESULTS_DIR, f'MLP_pr_{set_name}.png'))
    plt.close()
    
    # Threshold analysis
    thresholds = np.linspace(min(y_score), max(y_score), 100)
    accuracies = []
    f1_scores = []
    
    for thresh in thresholds:
        y_pred = (y_score >= thresh).astype(int)
        accuracies.append(accuracy_score(y_true, y_pred))
        f1_scores.append(f1_score(y_true, y_pred))
    
    # Accuracy vs Threshold plot
    plt.figure(figsize=(10, 8))
    plt.plot(thresholds, accuracies, label='Accuracy', color='blue')
    plt.axvline(x=OPTIMAL_THRESHOLD, color='red', linestyle='--', 
                label=f'Optimal Threshold ({OPTIMAL_THRESHOLD:.3f})')
    plt.xlabel('Threshold')
    plt.ylabel('Accuracy')
    plt.title(f'Accuracy vs Decision Threshold - {set_name}')
    plt.legend()
    plt.savefig(os.path.join(RESULTS_DIR, f'MLP_accuracy_vs_threshold_{set_name}.png'))
    plt.close()
    
    # F1 Score vs Threshold plot
    plt.figure(figsize=(10, 8))
    plt.plot(thresholds, f1_scores, label='F1 Score', color='green')
    plt.axvline(x=OPTIMAL_THRESHOLD, color='red', linestyle='--', 
                label=f'Optimal Threshold ({OPTIMAL_THRESHOLD:.3f})')
    plt.xlabel('Threshold')
    plt.ylabel('F1 Score')
    plt.title(f'F1 Score vs Decision Threshold - {set_name}')
    plt.legend()
    plt.savefig(os.path.join(RESULTS_DIR, f'MLP_f1_vs_threshold_{set_name}.png'))
    plt.close()

def test_mlp():
    """Main function to evaluate MLP on development and test sets"""
    try:
        # Load trained MLP model
        model_path = os.path.join(MODEL_DIR, 'trained_mlp_model.joblib')
        if not os.path.exists(model_path):
            raise FileNotFoundError(f"MLP model not found at {model_path}")
        
        mlp = load(model_path)
        
        # Evaluate on development set
        try:
            print("\nEvaluating MLP on development set...")
            X_dev, y_dev = data_loader.load_features('development')
            evaluate_mlp(mlp, X_dev, y_dev, "development")
        except Exception as e:
            print(f"\nWarning: Could not evaluate development set - {e}")
        
        # Evaluate on test set
        print("\nEvaluating MLP on test set...")
        X_test, y_test = data_loader.load_features('test')
        evaluate_mlp(mlp, X_test, y_test, "test")
                
    except Exception as e:
        print(f"\nCritical error during MLP testing: {str(e)}")
        raise

if __name__ == "__main__":
    os.makedirs(RESULTS_DIR, exist_ok=True)
    test_mlp()
import os

# Directory base del progetto
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

# URL dei dataset su FigShare - URL diretti al download
FIGSHARE_URLS = {
    'training': 'https://figshare.com/articles/dataset/Training_validation_and_testing_sample_data_used_in_Antibody_interface_prediction_with_3D_Zernike_descriptors_and_SVM_/5442229?file=11804663',
    'development': 'https://figshare.com/articles/dataset/Training_validation_and_testing_sample_data_used_in_Antibody_interface_prediction_with_3D_Zernike_descriptors_and_SVM_/5442229?file=11804657',
    'test': 'https://figshare.com/articles/dataset/Training_validation_and_testing_sample_data_used_in_Antibody_interface_prediction_with_3D_Zernike_descriptors_and_SVM_/5442229?file=11804666',
    'seq_descriptors': 'https://figshare.com/articles/dataset/Training_validation_and_testing_sample_data_used_in_Antibody_interface_prediction_with_3D_Zernike_descriptors_and_SVM_/5442229?file=11804651'
}

# Directory per la cache
CACHE_DIR = os.path.join(BASE_DIR, 'data_cache')
os.makedirs(CACHE_DIR, exist_ok=True)

# Directory di output
PROCESSED_DATA_DIR = os.path.join(BASE_DIR, 'data', 'processed')
os.makedirs(PROCESSED_DATA_DIR, exist_ok=True)
# Directory per i modelli salvati
MODEL_DIR = os.path.join(BASE_DIR, 'models')
os.makedirs(MODEL_DIR, exist_ok=True)

SVM_PARAMS = {
    'C': 540.2,           # Valore ottimale dal paper
    'kernel': 'rbf',      # Kernel usato nel paper
    'gamma': 7.983e-3,    # Valore gamma ottimale
    'class_weight': 'balanced',
    'probability': True,  # Necessario per ROC/PR curves
    'max_iter': 5000,     # Aumentato per convergenza
    'verbose': True
}

# Parametri dei descrittori
PATCH_RADIUS = 6.0  # Angstrom (come nel paper)
INTERFACE_DISTANCE = 4.5  # Angstrom (soglia per residui di interfaccia)
OPTIMAL_THRESHOLD = 0.6232  # Soglia ottimale dal paper
IF_CONTAMINATION = 0.18  # Parametro per Isolation Forest

# Aggiungi queste configurazioni
LOGGING = {
    'version': 1,
    'disable_existing_loggers': False,
    'handlers': {
        'console': {
            'class': 'logging.StreamHandler',
            'formatter': 'simple'
        },
    },
    'formatters': {
        'simple': {
            'format': '[%(asctime)s] %(levelname)s - %(message)s',
            'datefmt': '%H:%M:%S'
        }
    },
    'loggers': {
        'training': {
            'handlers': ['console'],
            'level': 'INFO',
        }
    }
}
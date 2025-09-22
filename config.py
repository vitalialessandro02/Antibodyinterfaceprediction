import os


# Project base directory 
BASE_DIR = os.path.dirname(os.path.abspath(__file__)) 

# Dataset URLs from FigShare
FIGSHARE_URLS = {
    'training': 'https://figshare.com/articles/dataset/Training_validation_and_testing_sample_data_used_in_Antibody_interface_prediction_with_3D_Zernike_descriptors_and_SVM_/5442229?file=11804663',
    'development': 'https://figshare.com/articles/dataset/Training_validation_and_testing_sample_data_used_in_Antibody_interface_prediction_with_3D_Zernike_descriptors_and_SVM_/5442229?file=11804657',
    'test': 'https://figshare.com/articles/dataset/Training_validation_and_testing_sample_data_used_in_Antibody_interface_prediction_with_3D_Zernike_descriptors_and_SVM_/5442229?file=11804666',
    'seq_descriptors': 'https://figshare.com/articles/dataset/Training_validation_and_testing_sample_data_used_in_Antibody_interface_prediction_with_3D_Zernike_descriptors_and_SVM_/5442229?file=11804651'
}

# Directory structure configuaration
CACHE_DIR = os.path.join(BASE_DIR, 'data_cache')
PROCESSED_DATA_DIR = os.path.join(BASE_DIR, 'data', 'processed')
MODEL_DIR = os.path.join(BASE_DIR, 'models')
PDB_DIR = os.path.join(BASE_DIR, 'data', 'pdb')
RESULTS_DIR = os.path.join(BASE_DIR, 'results')

for directory in [CACHE_DIR, PROCESSED_DATA_DIR, MODEL_DIR, PDB_DIR, RESULTS_DIR]:
    os.makedirs(directory, exist_ok=True)




# MLP Parameters
MLP_PARAMS = {
    'hidden_layer_sizes': (512, 256, 128, 64),
    'activation': 'relu',
    'solver': 'adam',
    'alpha': 0.0001,
    'batch_size': 256,
    'learning_rate': 'adaptive',
    'max_iter': 500,
    'early_stopping': True,
    'validation_fraction': 0.1,
    'random_state': 42
}

# Random Forest Parameters
RF_PARAMS = {
    'n_estimators': 500,
    'criterion': 'gini',
    'max_depth': None,
    'min_samples_split': 2,
    'min_samples_leaf': 1,
    'max_features': 'sqrt',
    'bootstrap': True,
    'class_weight': 'balanced',
    'random_state': 42,
    'n_jobs': -1
}


OPTIMAL_THRESHOLD = 0.6232 





# Logging configuration
LOGGING = {
    'version': 1,
    'disable_existing_loggers': False,
    'formatters': {
        'verbose': {
            'format': '[%(asctime)s] %(levelname)s [%(name)s:%(lineno)s] %(message)s',
            'datefmt': '%Y-%m-%d %H:%M:%S'
        },
        'simple': {
            'format': '[%(asctime)s] %(levelname)s - %(message)s',
            'datefmt': '%H:%M:%S'
        }
    },
    'handlers': {
        'console': {
            'class': 'logging.StreamHandler',
            'formatter': 'simple',
            'level': 'INFO'
        },
        'file': {
            'class': 'logging.FileHandler',
            'filename': os.path.join(BASE_DIR, 'training.log'),
            'formatter': 'verbose',
            'level': 'DEBUG'
        }
    },
    'loggers': {
        'training': {
            'handlers': ['console', 'file'],
            'level': 'DEBUG',
            'propagate': True
        },
        'data': {
            'handlers': ['console', 'file'],
            'level': 'INFO',
            'propagate': True
        }
    }
}
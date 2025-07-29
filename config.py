import os


# Project base directory - ora punta alla directory del progetto stesso
BASE_DIR = os.path.dirname(os.path.abspath(__file__))  # Modificato per usare la directory corrente

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


# Create directories if they don't exist
for directory in [CACHE_DIR, PROCESSED_DATA_DIR, MODEL_DIR, PDB_DIR, RESULTS_DIR]:
    os.makedirs(directory, exist_ok=True)


# Aggiungi queste configurazioni alla fine di config.py

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





# Feature extraction parameters
PATCH_RADIUS = 6.0        # Patch size in Angstrom
INTERFACE_DISTANCE = 4.5  # Distance threshold for interface residues 
OPTIMAL_THRESHOLD = 0.6232 
IF_CONTAMINATION = 0.18   # Parameter for Isolation Forest

# Resolution parameters
VOXEL_RESOLUTION = 64     
SOLVENT_PROBE_RADIUS = 1.4  

# 3D Zernike parameters
ZERNIKE_ORDER = 5         # Maximum order of descriptors
ZERNIKE_DIMENSION = 12    # Number of descriptors per function (for order=5)

# Selected physicochemical properties from AAindex
PHYSICOCHEMICAL_PROPERTIES = [
    'BIOV880101', 'CHAM820101', 'CHAM820102', 'CHOC760101', 
    'EISD840101', 'FASG760101', 'FASG760102', 'FASG760103',
    'FASG760104', 'FASG760105', 'GRAR740102', 'JANJ780101',
    'JANJ780102', 'JANJ780103', 'KARP850101', 'KARP850102',
    'KARP850103', 'LEVM780101', 'LEVM780102', 'LEVM780103'
]

# Sampling parameters for positive/negative examples
NON_INTERFACE_SAMPLING_DISTANCE = 1.8 
INTERFACE_SAMPLING_DISTANCE = 1.0














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
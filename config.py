import os

# Project base directory
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

# Dataset URLs from FigShare
FIGSHARE_URLS = {
    'training': 'https://figshare.com/articles/dataset/Training_validation_and_testing_sample_data_used_in_Antibody_interface_prediction_with_3D_Zernike_descriptors_and_SVM_/5442229?file=11804663',
    'development': 'https://figshare.com/articles/dataset/Training_validation_and_testing_sample_data_used_in_Antibody_interface_prediction_with_3D_Zernike_descriptors_and_SVM_/5442229?file=11804657',
    'test': 'https://figshare.com/articles/dataset/Training_validation_and_testing_sample_data_used_in_Antibody_interface_prediction_with_3D_Zernike_descriptors_and_SVM_/5442229?file=11804666',
    'seq_descriptors': 'https://figshare.com/articles/dataset/Training_validation_and_testing_sample_data_used_in_Antibody_interface_prediction_with_3D_Zernike_descriptors_and_SVM_/5442229?file=11804651'
}

# Directory structure
CACHE_DIR = os.path.join(BASE_DIR, 'data_cache')
PROCESSED_DATA_DIR = os.path.join(BASE_DIR, 'data', 'processed')
MODEL_DIR = os.path.join(BASE_DIR, 'models')
PDB_DIR = os.path.join(BASE_DIR, 'data', 'pdb')
RESULTS_DIR = os.path.join(BASE_DIR, 'results')

# Create directories if they don't exist
for directory in [CACHE_DIR, PROCESSED_DATA_DIR, MODEL_DIR, PDB_DIR, RESULTS_DIR]:
    os.makedirs(directory, exist_ok=True)

# SVM Parameters from the paper
SVM_PARAMS = {
    'C': 540.2,           # Optimal value from paper
    'kernel': 'rbf',      # Kernel used in paper
    'gamma': 7.983e-3,    # Optimal gamma value
    'class_weight': 'balanced',
    'probability': True,  # Needed for ROC/PR curves
    'max_iter': 10000,    # Increased for convergence
    'verbose': True,
    'random_state': 42    # For reproducibility
}

# Feature extraction parameters
PATCH_RADIUS = 6.0        # Ångstrom (as in paper)
INTERFACE_DISTANCE = 4.5  # Ångstrom (threshold for interface residues)
OPTIMAL_THRESHOLD = 0.6232  # Optimal threshold from paper
IF_CONTAMINATION = 0.18   # Parameter for Isolation Forest

# Resolution parameters
VOXEL_RESOLUTION = 64     # voxels per Å^3
SOLVENT_PROBE_RADIUS = 1.4  # Ångstrom

# 3D Zernike parameters
ZERNIKE_ORDER = 5         # As per the paper
ZERNIKE_DIMENSION = 12    # Number of descriptors per function (for order=5)

# Selected physicochemical properties from AAindex (as in the paper)
PHYSICOCHEMICAL_PROPERTIES = [
    'BIOV880101', 'CHAM820101', 'CHAM820102', 'CHOC760101', 
    'EISD840101', 'FASG760101', 'FASG760102', 'FASG760103',
    'FASG760104', 'FASG760105', 'GRAR740102', 'JANJ780101',
    'JANJ780102', 'JANJ780103', 'KARP850101', 'KARP850102',
    'KARP850103', 'LEVM780101', 'LEVM780102', 'LEVM780103'
]

# Sampling parameters
NON_INTERFACE_SAMPLING_DISTANCE = 1.8  # Å
INTERFACE_SAMPLING_DISTANCE = 1.0      # Å

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
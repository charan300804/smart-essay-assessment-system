import os

# Paths
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(BASE_DIR)
DATA_DIR = os.path.join(PROJECT_ROOT, 'data')
MODELS_DIR = os.path.join(PROJECT_ROOT, 'models')

# Dataset specific
DATA_FILE = os.path.join(DATA_DIR, 'training_set_rel3.tsv')
GLOVE_FILE = os.path.join(DATA_DIR, 'glove.6B.100d.txt')

# Hyperparameters
MAX_SEQUENCE_LENGTH = 500
EMBEDDING_DIM = 100
VOCAB_SIZE = 20000 
BATCH_SIZE = 64
EPOCHS = 50
LEARNING_RATE = 0.001
LSTM_UNITS = 300
DROPOUT_RATE = 0.5

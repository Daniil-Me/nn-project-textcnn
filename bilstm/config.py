# --- Paths ---
DATA_PATH = 'mr.p'

# --- Embedding parameters ---
EMBEDDING_DIM = 300
VOCAB_SIZE = 27644

# --- Model parameters ---
HIDDEN_SIZE = 128
NUM_LAYERS = 2
DROPOUT = 0.5
NUM_CLASSES = 2

# --- Training parameters ---
BATCH_SIZE = 64
NUM_EPOCHS = 20
LEARNING_RATE = 0.001
MAX_SEQ_LEN = 50

# --- Data splits ---
SPLITS_TRAIN = [0, 1, 2, 3, 4, 5, 6, 7]
SPLITS_VAL   = [8]
SPLITS_TEST  = [9]

# --- Reproducibility ---
SEED = 42
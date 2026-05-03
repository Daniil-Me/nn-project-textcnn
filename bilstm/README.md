# Bi-LSTM Sentiment Analysis

Baseline model for sentiment classification on the MR (Movie Reviews) dataset.
Classifies movie reviews as positive (1) or negative (0).

## Project Structure

```
bilstm/
├── config.py       # All hyperparameters and settings
├── model.py        # Bi-LSTM model architecture
├── train.py        # Training script with Early Stopping
├── evaluate.py     # Evaluation script (accuracy, F1, confusion matrix)
├── requirements.txt
└── README.md
```

## Dataset

MR (Movie Reviews) dataset — stored as `mr.p` (pickle file).  
Place `mr.p` in the same directory as the scripts before running.  
Processed dataset (mr.p) is available here:  
(https://drive.google.com/file/d/1ljozYFVPu9Hb8DJURsOqtdyB-b5uWBZZ/view?usp=sharing)  

| Parameter | Value |
|---|---|
| Total samples | 54,981 |
| Positive (y=1) | 32,322 |
| Negative (y=0) | 22,659 |
| Vocabulary size | 27,644 |
| Embeddings | GloVe (300 dimensions) |
| Train splits | 0-7 (80%) |
| Validation split | 8 (10%) |
| Test split | 9 (10%) |

## Model Architecture

```
Input (token indices)
    → Embedding layer (GloVe, 300-dim, frozen)
    → Bi-LSTM (hidden_size=128, num_layers=2, bidirectional)
    → Dropout (0.5)
    → Fully Connected (256 → 2)
    → Softmax → Output (0 or 1)
```

## Installation

```bash
pip install -r requirements.txt
```

## Usage

**Train the model:**
```bash
python train.py
```

**Evaluate on test set:**
```bash
python evaluate.py
```

## Hyperparameters

All hyperparameters are defined in `config.py`:

| Parameter | Value |
|---|---|
| HIDDEN_SIZE | 128 |
| NUM_LAYERS | 2 |
| DROPOUT | 0.5 |
| BATCH_SIZE | 64 |
| NUM_EPOCHS | 20 |
| LEARNING_RATE | 0.001 |
| MAX_SEQ_LEN | 50 |
| SEED | 42 |

## Output Files

After training and evaluation the following files are generated:

- `bilstm_model.pt` — saved model weights
- `training_curve.png` — train/val loss plot
- `results.txt` — test accuracy, F1 score, confusion matrix
- `confusion_matrix.png` — confusion matrix plot

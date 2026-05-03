import pickle
import numpy as np
import torch
from torch.utils.data import DataLoader
from sklearn.metrics import accuracy_score, f1_score, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns
from model import BiLSTM
from config import (DATA_PATH, BATCH_SIZE, MAX_SEQ_LEN, SPLITS_TEST, SEED)

# --- Reproducibility ---
torch.manual_seed(SEED)
np.random.seed(SEED)

# --- Load data ---
def load_data():
    with open(DATA_PATH, 'rb') as f:
        data = pickle.load(f, encoding='latin1')
    texts = data[0]
    embedding_matrix = data[1]
    word_to_idx = data[3]
    return texts, embedding_matrix, word_to_idx

# --- Text to indices ---
def text_to_indices(text, word_to_idx, max_len):
    words = text.split()
    indices = [word_to_idx.get(word, 0) for word in words]
    if len(indices) < max_len:
        indices += [0] * (max_len - len(indices))
    else:
        indices = indices[:max_len]
    return indices

# --- Dataset ---
from train import MovieDataset

# --- Evaluate ---
def evaluate():
    texts, embedding_matrix, word_to_idx = load_data()

    test_dataset = MovieDataset(texts, word_to_idx, SPLITS_TEST)
    test_loader  = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False)

    model = BiLSTM(embedding_matrix)
    model.load_state_dict(torch.load('bilstm_model.pt'))
    model.eval()

    all_predictions = []
    all_labels = []

    with torch.no_grad():
        for x_batch, y_batch in test_loader:
            output = model(x_batch)
            predictions = torch.argmax(output, dim=1)
            all_predictions.extend(predictions.numpy())
            all_labels.extend(y_batch.numpy())

    # --- Metrics ---
    accuracy = accuracy_score(all_labels, all_predictions)
    f1 = f1_score(all_labels, all_predictions, average='weighted')
    cm = confusion_matrix(all_labels, all_predictions)

    print(f"Test Accuracy: {accuracy * 100:.2f}%")
    print(f"F1 Score:      {f1:.4f}")
    print(f"Confusion Matrix:\n{cm}")

    # --- Save results ---
    with open('results.txt', 'w') as f:
        f.write(f"Test Accuracy: {accuracy * 100:.2f}%\n")
        f.write(f"F1 Score:      {f1:.4f}\n")
        f.write(f"Confusion Matrix:\n{cm}\n")
    print("Results saved to results.txt")

    # --- Plot confusion matrix ---
    plt.figure(figsize=(6, 5))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                xticklabels=['Negative', 'Positive'],
                yticklabels=['Negative', 'Positive'])
    plt.xlabel('Predicted')
    plt.ylabel('Actual')
    plt.title('Confusion Matrix')
    plt.savefig('confusion_matrix.png')
    print("Confusion matrix saved to confusion_matrix.png")

if __name__ == '__main__':
    evaluate()
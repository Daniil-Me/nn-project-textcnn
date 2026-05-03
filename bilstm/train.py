import pickle
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from model import BiLSTM
from config import (DATA_PATH, EMBEDDING_DIM, VOCAB_SIZE, HIDDEN_SIZE,
                    NUM_LAYERS, DROPOUT, NUM_CLASSES, BATCH_SIZE,
                    NUM_EPOCHS, LEARNING_RATE, MAX_SEQ_LEN,
                    SPLITS_TRAIN, SPLITS_VAL, SPLITS_TEST, SEED)
import matplotlib.pyplot as plt

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
class MovieDataset(Dataset):
    def __init__(self, texts, word_to_idx, splits):
        self.samples = [t for t in texts if t['split'] in splits]
        self.word_to_idx = word_to_idx

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        sample = self.samples[idx]
        indices = text_to_indices(sample['text'], self.word_to_idx, MAX_SEQ_LEN)
        x = torch.tensor(indices, dtype=torch.long)
        y = torch.tensor(sample['y'], dtype=torch.long)
        return x, y

# --- Training loop ---
def train():
    texts, embedding_matrix, word_to_idx = load_data()

    train_dataset = MovieDataset(texts, word_to_idx, SPLITS_TRAIN)
    val_dataset   = MovieDataset(texts, word_to_idx, SPLITS_VAL)

    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
    val_loader   = DataLoader(val_dataset,   batch_size=BATCH_SIZE, shuffle=False)

    model = BiLSTM(embedding_matrix)
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE)

    train_losses = []
    val_losses   = []
    val_accuracies = []

    # --- Early stopping setup ---
    best_val_loss = float('inf')
    patience = 3
    patience_counter = 0

    for epoch in range(NUM_EPOCHS):

        # --- Train ---
        model.train()
        total_train_loss = 0

        for x_batch, y_batch in train_loader:
            optimizer.zero_grad()
            output = model(x_batch)
            loss = criterion(output, y_batch)
            loss.backward()
            optimizer.step()
            total_train_loss += loss.item()

        avg_train_loss = total_train_loss / len(train_loader)
        train_losses.append(avg_train_loss)

        # --- Validate ---
        model.eval()
        total_val_loss = 0
        correct = 0
        total = 0

        with torch.no_grad():
            for x_batch, y_batch in val_loader:
                output = model(x_batch)
                loss = criterion(output, y_batch)
                total_val_loss += loss.item()
                predictions = torch.argmax(output, dim=1)
                correct += (predictions == y_batch).sum().item()
                total += y_batch.size(0)

        avg_val_loss = total_val_loss / len(val_loader)
        val_accuracy = correct / total * 100
        val_losses.append(avg_val_loss)
        val_accuracies.append(val_accuracy)

        print(f"Epoch {epoch+1}/{NUM_EPOCHS} | "
              f"Train Loss: {avg_train_loss:.4f} | "
              f"Val Loss: {avg_val_loss:.4f} | "
              f"Val Accuracy: {val_accuracy:.2f}%")

        # --- Early stopping check ---
        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            patience_counter = 0
            torch.save(model.state_dict(), 'bilstm_model.pt')
            print(f"  → Best model saved (val loss: {best_val_loss:.4f})")
        else:
            patience_counter += 1
            print(f"  → No improvement ({patience_counter}/{patience})")
            if patience_counter >= patience:
                print(f"Early stopping at epoch {epoch+1}")
                break

    # --- Save training curve ---
    plt.figure(figsize=(10, 5))
    plt.plot(train_losses, label='Train Loss')
    plt.plot(val_losses,   label='Val Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title('Training Curve')
    plt.legend()
    plt.savefig('training_curve.png')
    print("Training curve saved to training_curve.png")

if __name__ == '__main__':
    train()
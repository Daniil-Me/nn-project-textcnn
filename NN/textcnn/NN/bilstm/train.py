import torch
import torch.nn as nn
import numpy as np
from torch.utils.data import DataLoader, TensorDataset
from sklearn.metrics import f1_score, accuracy_score
from collections import Counter
from model import BiLSTM
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix

def get_class_weights(y_train, num_classes):
    counter = Counter(y_train.numpy())
    total = len(y_train)
    weights = [total / (num_classes * counter[i]) for i in range(num_classes)]
    return torch.tensor(weights, dtype=torch.float32)

def train_model(model, X_train, y_train, X_val, y_val,
                epochs=20, batch_size=64, lr=0.001, patience=3):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device)
    weights = get_class_weights(y_train, num_classes=6).to(device)
    criterion = nn.CrossEntropyLoss(weight=weights)
    optimizer = torch.optim.Adam(
        filter(lambda p: p.requires_grad, model.parameters()), lr=lr)
    train_loader = DataLoader(
        TensorDataset(X_train, y_train), batch_size=batch_size, shuffle=True)
    best_val_f1 = 0
    patience_counter = 0
    history = {"train_loss": [], "val_loss": [], "val_f1": []}
    for epoch in range(epochs):
        model.train()
        train_loss = 0
        for X_batch, y_batch in train_loader:
            X_batch, y_batch = X_batch.to(device), y_batch.to(device)
            optimizer.zero_grad()
            loss = criterion(model(X_batch), y_batch)
            loss.backward()
            optimizer.step()
            train_loss += loss.item()
        model.eval()
        with torch.no_grad():
            val_out = model(X_val.to(device))
            val_loss = criterion(val_out, y_val.to(device)).item()
            val_preds = val_out.argmax(dim=1).cpu().numpy()
            val_f1 = f1_score(y_val.numpy(), val_preds, average="macro")
        history["train_loss"].append(train_loss / len(train_loader))
        history["val_loss"].append(val_loss)
        history["val_f1"].append(val_f1)
        print(f"Ep {epoch+1:02d} | Train Loss: {train_loss/len(train_loader):.4f} | Val Loss: {val_loss:.4f} | Val F1: {val_f1:.4f}")
        if val_f1 > best_val_f1:
            best_val_f1 = val_f1
            torch.save(model.state_dict(), "best_model.pt")
            patience_counter = 0
        else:
            patience_counter += 1
            if patience_counter >= patience:
                print(f"Early stopping at epoch {epoch+1}")
                break
    model.load_state_dict(torch.load("best_model.pt", weights_only=False))
    return model, history

data = torch.load("../preprocessed_data.pt", weights_only=False)
X_train, y_train = data["X_train"], data["y_train"]
X_val, y_val     = data["X_val"],   data["y_val"]
X_test, y_test   = data["X_test"],  data["y_test"]
embedding_matrix = data["embedding_matrix"]
label_names      = data["label_names"]
VOCAB_SIZE = len(data["vocab"])

configs = [
    {"name": "Exp1_baseline",    "hidden_size": 128, "num_layers": 2, "dropout": 0.5, "lr": 0.001},
    {"name": "Exp2_large",       "hidden_size": 256, "num_layers": 2, "dropout": 0.5, "lr": 0.001},
    {"name": "Exp3_single_layer","hidden_size": 128, "num_layers": 1, "dropout": 0.5, "lr": 0.001},
]

results_txt = ""
for cfg in configs:
    print(f"\n{'='*50}\nRunning {cfg['name']}\n{'='*50}")
    model = BiLSTM(VOCAB_SIZE, 300, cfg["hidden_size"],
                   cfg["num_layers"], 6, cfg["dropout"], embedding_matrix)
    model, history = train_model(model, X_train, y_train, X_val, y_val,
                                  epochs=20, batch_size=64,
                                  lr=cfg["lr"], patience=3)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.eval()
    with torch.no_grad():
        preds = model(X_test.to(device)).argmax(dim=1).cpu().numpy()
    acc = accuracy_score(y_test.numpy(), preds)
    f1  = f1_score(y_test.numpy(), preds, average="macro")
    f1_per = f1_score(y_test.numpy(), preds, average=None)
    print(f"Accuracy: {acc*100:.2f}% | F1: {f1:.4f}")
    results_txt += f"{cfg['name']}: Accuracy={acc*100:.2f}% F1={f1:.4f}\n"
    for name, score in zip(label_names, f1_per):
        results_txt += f"  {name}: {score:.4f}\n"
    torch.save(model.state_dict(), f"results/{cfg['name']}_weights.pt")

    plt.figure(figsize=(6,4))
    plt.plot(history["train_loss"], label="Train Loss")
    plt.plot(history["val_loss"],   label="Val Loss")
    plt.title(f"BiLSTM {cfg['name']} — Training Curve")
    plt.xlabel("Epoch"); plt.ylabel("Loss"); plt.legend()
    plt.tight_layout()
    plt.savefig(f"results/{cfg['name']}_training_curve.png", dpi=150)
    plt.close()

    cm = confusion_matrix(y_test.numpy(), preds)
    plt.figure(figsize=(8,6))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues",
                xticklabels=label_names, yticklabels=label_names)
    plt.title(f"BiLSTM {cfg['name']} — Confusion Matrix")
    plt.tight_layout()
    plt.savefig(f"results/{cfg['name']}_confusion_matrix.png", dpi=150)
    plt.close()

with open("results/experiments_results.txt", "w") as f:
    f.write(results_txt)
print("\nAll done! Results saved.")

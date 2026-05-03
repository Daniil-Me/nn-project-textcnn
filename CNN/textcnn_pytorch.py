import matplotlib.pyplot as plt
from sklearn.metrics import classification_report, f1_score
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
import argparse
import pickle
import random
import time
from dataclasses import dataclass

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader


def set_seed(seed: int = 3435):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


class TextDataset(Dataset):
    def __init__(self, x, y):
        self.x = torch.tensor(x, dtype=torch.long)
        self.y = torch.tensor(y, dtype=torch.long)

    def __len__(self):
        return len(self.y)

    def __getitem__(self, idx):
        return self.x[idx], self.y[idx]


class TextCNN(nn.Module):
    def __init__(
        self,
        vocab_size,
        embed_dim,
        num_classes=2,
        kernel_sizes=(3, 4, 5),
        num_filters=64, # ЕКСПЕРИМЕНТ 5: Зменшено кількість фільтрів (аналог HIDDEN_SIZE)
        dropout=0.5,
        embeddings=None,
        static=False,
    ):
        super().__init__()

        self.embedding = nn.Embedding(vocab_size, embed_dim, padding_idx=0)

        if embeddings is not None:
            emb_tensor = torch.tensor(embeddings, dtype=torch.float32)
            self.embedding.weight.data.copy_(emb_tensor)

        if static:
            self.embedding.weight.requires_grad = False

        self.convs = nn.ModuleList(
            [
                nn.Conv1d(
                    in_channels=embed_dim,
                    out_channels=num_filters,
                    kernel_size=k
                )
                for k in kernel_sizes
            ]
        )

        self.dropout = nn.Dropout(dropout)
        self.classifier = nn.Linear(num_filters * len(kernel_sizes), num_classes)

    def forward(self, x):
        x = self.embedding(x)                
        x = x.transpose(1, 2)               

        conv_outputs = []
        for conv in self.convs:
            c = torch.relu(conv(x))         
            p = torch.max(c, dim=2).values  
            conv_outputs.append(p)

        x = torch.cat(conv_outputs, dim=1)  
        x = self.dropout(x)
        logits = self.classifier(x)
        return logits


def get_idx_from_sent(sent, word_idx_map, max_l=56, filter_h=5):
    x = []
    pad = filter_h - 1
    for _ in range(pad): x.append(0)
    for word in sent.split():
        if word in word_idx_map: x.append(word_idx_map[word])
    while len(x) < max_l + 2 * pad: x.append(0)
    return x


def make_idx_data_cv(revs, word_idx_map, cv, max_l=56, filter_h=5):
    train, test = [], []
    for rev in revs:
        sent = get_idx_from_sent(rev["text"], word_idx_map, max_l=max_l, filter_h=filter_h)
        sent.append(rev["y"])
        if rev["split"] == cv: test.append(sent)
        else: train.append(sent)
    return np.array(train, dtype="int64"), np.array(test, dtype="int64")


def split_train_val(train_data, val_ratio=0.1, seed=3435):
    rng = np.random.default_rng(seed)
    idx = np.arange(len(train_data))
    rng.shuffle(idx)
    train_data = train_data[idx]
    n_val = max(1, int(round(len(train_data) * val_ratio)))
    return train_data[n_val:], train_data[:n_val]


@dataclass
class Metrics:
    acc: float
    loss: float


def evaluate(model, loader, criterion, device):
    model.eval()
    total_loss, total_correct, total = 0.0, 0, 0
    with torch.no_grad():
        for xb, yb in loader:
            xb, yb = xb.to(device), yb.to(device)
            logits = model(xb)
            loss = criterion(logits, yb)
            total_loss += loss.item() * yb.size(0)
            total_correct += (logits.argmax(dim=1) == yb).sum().item()
            total += yb.size(0)
    return Metrics(acc=total_correct / total if total else 0.0,
                   loss=total_loss / total if total else 0.0)


def train_one_fold(
    train_data, test_data, embeddings, static, 
    batch_size=50, n_epochs=20, lr=0.0001, # ЕКСПЕРИМЕНТ 5: Дуже низький Learning Rate
    dropout=0.5, kernel_sizes=(3, 4, 5), 
    num_filters=64, # ЕКСПЕРИМЕНТ 5: Менший розмір шару
    seed=3435, fold_idx=0
):
    set_seed(seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    train_data, val_data = split_train_val(train_data, val_ratio=0.1, seed=seed)

    train_loader = DataLoader(TextDataset(train_data[:, :-1], train_data[:, -1]), batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(TextDataset(val_data[:, :-1], val_data[:, -1]), batch_size=batch_size, shuffle=False)
    test_loader = DataLoader(TextDataset(test_data[:, :-1], test_data[:, -1]), batch_size=batch_size, shuffle=False)

    vocab_size, embed_dim = embeddings.shape
    model = TextCNN(vocab_size, embed_dim, num_classes=2, kernel_sizes=kernel_sizes, 
                    num_filters=num_filters, dropout=dropout, embeddings=embeddings, static=static).to(device)

    # --- ЕКСПЕРИМЕНТ 5: Повернення до стандартної функції втрат ---
    criterion = nn.CrossEntropyLoss()
    # -------------------------------------------------------------

    optimizer = torch.optim.Adam(filter(lambda p: p.requires_grad, model.parameters()), lr=lr)

    history = {'train_acc': [], 'val_acc': [], 'train_loss': [], 'val_loss': []}
    
    patience, counter, best_val_loss = 3, 0, float('inf')
    best_model_state = None

    for epoch in range(1, n_epochs + 1):
        model.train()
        for xb, yb in train_loader:
            xb, yb = xb.to(device), yb.to(device)
            optimizer.zero_grad()
            loss = criterion(model(xb), yb)
            loss.backward()
            optimizer.step()
            if model.embedding.weight.requires_grad:
                with torch.no_grad(): model.embedding.weight[0].fill_(0)

        t_m = evaluate(model, train_loader, criterion, device)
        v_m = evaluate(model, val_loader, criterion, device)
        
        history['train_acc'].append(t_m.acc)
        history['val_acc'].append(v_m.acc)
        history['train_loss'].append(t_m.loss)
        history['val_loss'].append(v_m.loss)

        print(f"Fold {fold_idx} | Ep {epoch:02d} | Train Acc: {t_m.acc*100:.2f}% | Val Acc: {v_m.acc*100:.2f}% | Val Loss: {v_m.loss:.4f}")

        if v_m.loss < best_val_loss:
            best_val_loss = v_m.loss
            best_model_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
            counter = 0
        else:
            counter += 1
            if counter >= patience:
                print(f"Early Stopping! Зупинка на епосі {epoch}")
                break

    if best_model_state:
        model.load_state_dict(best_model_state)

    plt.figure(figsize=(12, 4))
    plt.subplot(1, 2, 1)
    plt.plot(history['train_loss'], label='Train Loss')
    plt.plot(history['val_loss'], label='Val Loss')
    plt.title(f'Loss (Exp 5 - Low LR, Fold {fold_idx})')
    plt.legend()
    plt.subplot(1, 2, 2)
    plt.plot(history['train_acc'], label='Train Acc')
    plt.plot(history['val_acc'], label='Val Acc')
    plt.title(f'Accuracy (Fold {fold_idx})')
    plt.legend()
    plt.savefig(f"results_fold_{fold_idx}.png")
    plt.close()

    model.eval()
    all_preds, all_y = [], []
    with torch.no_grad():
        for xb, yb in test_loader:
            preds = model(xb.to(device)).argmax(dim=1)
            all_preds.extend(preds.cpu().numpy())
            all_y.extend(yb.numpy())
    
    f1 = f1_score(all_y, all_preds, average='weighted')
    save_confusion_matrix(all_y, all_preds, fold_idx) 
    print(f"--- Fold {fold_idx} Test F1: {f1:.4f} ---")
    
    return f1


def save_confusion_matrix(y_true, y_pred, fold_idx):
    cm = confusion_matrix(y_true, y_pred)
    classes = ['Negative', 'Positive'] 
    fig, ax = plt.subplots(figsize=(8, 6))
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=classes)
    disp.plot(cmap=plt.cm.Blues, ax=ax)
    plt.title(f'Confusion Matrix (Exp 5) - Fold {fold_idx}')
    plt.savefig(f'confusion_matrix_fold_{fold_idx}.png')
    plt.close()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", choices=["static", "non-static"], default="non-static")
    parser.add_argument("--vectors", choices=["rand", "word2vec"], default="word2vec")
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--batch-size", type=int, default=50)
    parser.add_argument("--dropout", type=float, default=0.5)
    parser.add_argument("--num-filters", type=int, default=64) # ЕКСПЕРИМЕНТ 5
    parser.add_argument("--seed", type=int, default=3435)
    args = parser.parse_args()

    with open("mr.p", "rb") as f:
        revs, W, W2, word_idx_map, vocab = pickle.load(f, encoding='latin1')

    static = args.mode == "static"
    embeddings = W2 if args.vectors == "rand" else W

    print(f"EXPERIMENT 5: TextCNN-{args.mode} | Small Model & Low LR (0.0001)")
    
    results = []
    for fold in range(4):
        print(f"\n===== fold {fold} =====")
        train_data, test_data = make_idx_data_cv(revs, word_idx_map, fold)
        acc = train_one_fold(
            train_data=train_data, test_data=test_data, embeddings=embeddings,
            static=static, batch_size=args.batch_size, n_epochs=args.epochs,
            dropout=args.dropout, seed=args.seed + fold, fold_idx=fold,
            num_filters=args.num_filters, lr=0.0001
        )
        results.append(acc)

    print("\n===== final =====")
    print("fold accuracies:", [round(x, 4) for x in results])
    print("mean accuracy:", round(float(np.mean(results)), 4))


if __name__ == "__main__":
    main()
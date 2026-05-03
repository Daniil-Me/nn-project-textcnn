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
        num_filters=100,
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
        # x: (batch, seq_len)
        x = self.embedding(x)                # (batch, seq_len, embed_dim)
        x = x.transpose(1, 2)               # (batch, embed_dim, seq_len)

        conv_outputs = []
        for conv in self.convs:
            c = torch.relu(conv(x))         # (batch, num_filters, L_out)
            p = torch.max(c, dim=2).values  # Global Max Pooling
            conv_outputs.append(p)

        x = torch.cat(conv_outputs, dim=1)  # (batch, num_filters * len(kernel_sizes))
        x = self.dropout(x)
        logits = self.classifier(x)
        return logits


def get_idx_from_sent(sent, word_idx_map, max_l=56, filter_h=5):
    x = []
    pad = filter_h - 1

    for _ in range(pad):
        x.append(0)

    for word in sent.split():
        if word in word_idx_map:
            x.append(word_idx_map[word])

    while len(x) < max_l + 2 * pad:
        x.append(0)

    return x


def make_idx_data_cv(revs, word_idx_map, cv, max_l=56, filter_h=5):
    train, test = [], []

    for rev in revs:
        sent = get_idx_from_sent(rev["text"], word_idx_map, max_l=max_l, filter_h=filter_h)
        sent.append(rev["y"])
        if rev["split"] == cv:
            test.append(sent)
        else:
            train.append(sent)

    train = np.array(train, dtype="int64")
    test = np.array(test, dtype="int64")
    return train, test


def split_train_val(train_data, val_ratio=0.1, seed=3435):
    rng = np.random.default_rng(seed)
    idx = np.arange(len(train_data))
    rng.shuffle(idx)
    train_data = train_data[idx]

    n_val = int(round(len(train_data) * val_ratio))
    if n_val == 0:
        n_val = 1

    val = train_data[:n_val]
    train = train_data[n_val:]
    return train, val


@dataclass
class Metrics:
    acc: float
    loss: float


def evaluate(model, loader, criterion, device):
    model.eval()
    total_loss = 0.0
    total_correct = 0
    total = 0

    with torch.no_grad():
        for xb, yb in loader:
            xb, yb = xb.to(device), yb.to(device)
            logits = model(xb)
            loss = criterion(logits, yb)

            total_loss += loss.item() * yb.size(0)
            preds = logits.argmax(dim=1)
            total_correct += (preds == yb).sum().item()
            total += yb.size(0)

    return Metrics(
        acc=total_correct / total if total else 0.0,
        loss=total_loss / total if total else 0.0,
    )


def train_one_fold(
    train_data,
    test_data,
    embeddings,
    static,
    batch_size=50,
    n_epochs=25,
    lr=1e-3,
    dropout=0.5,
    kernel_sizes=(3, 4, 5),
    num_filters=100,
    seed=3435,
):
    set_seed(seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    train_data, val_data = split_train_val(train_data, val_ratio=0.1, seed=seed)

    x_train, y_train = train_data[:, :-1], train_data[:, -1]
    x_val, y_val = val_data[:, :-1], val_data[:, -1]
    x_test, y_test = test_data[:, :-1], test_data[:, -1]

    train_loader = DataLoader(TextDataset(x_train, y_train), batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(TextDataset(x_val, y_val), batch_size=batch_size, shuffle=False)
    test_loader = DataLoader(TextDataset(x_test, y_test), batch_size=batch_size, shuffle=False)

    vocab_size, embed_dim = embeddings.shape

    model = TextCNN(
        vocab_size=vocab_size,
        embed_dim=embed_dim,
        num_classes=2,
        kernel_sizes=kernel_sizes,
        num_filters=num_filters,
        dropout=dropout,
        embeddings=embeddings,
        static=static,
    ).to(device)

    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(
        filter(lambda p: p.requires_grad, model.parameters()),
        lr=lr
    )

    best_val_acc = -1.0
    best_test_acc = 0.0

    for epoch in range(1, n_epochs + 1):
        model.train()
        start = time.time()

        for xb, yb in train_loader:
            xb, yb = xb.to(device), yb.to(device)

            optimizer.zero_grad()
            logits = model(xb)
            loss = criterion(logits, yb)
            loss.backward()
            optimizer.step()

            if model.embedding.weight.requires_grad:
                with torch.no_grad():
                    model.embedding.weight[0].fill_(0)

        train_metrics = evaluate(model, train_loader, criterion, device)
        val_metrics = evaluate(model, val_loader, criterion, device)

        print(
            f"epoch: {epoch:02d}, "
            f"time: {time.time() - start:.2f}s, "
            f"train acc: {train_metrics.acc * 100:.2f}%, "
            f"val acc: {val_metrics.acc * 100:.2f}%"
        )

        if val_metrics.acc >= best_val_acc:
            best_val_acc = val_metrics.acc
            test_metrics = evaluate(model, test_loader, criterion, device)
            best_test_acc = test_metrics.acc

    return best_test_acc


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", choices=["static", "non-static"], default="non-static")
    parser.add_argument("--vectors", choices=["rand", "word2vec"], default="word2vec")
    parser.add_argument("--epochs", type=int, default=25)
    parser.add_argument("--batch-size", type=int, default=50)
    parser.add_argument("--dropout", type=float, default=0.5)
    parser.add_argument("--num-filters", type=int, default=100)
    parser.add_argument("--seed", type=int, default=3435)
    args = parser.parse_args()

    print("loading data...")
    with open("mr.p", "rb") as f:
        revs, W, W2, word_idx_map, vocab = pickle.load(f)
    print("data loaded!")

    static = args.mode == "static"
    embeddings = W2 if args.vectors == "rand" else W

    print(f"model architecture: TextCNN-{args.mode}")
    print(f"vectors: {args.vectors}")
    print("kernel sizes: [3, 4, 5]")
    print(f"dropout: {args.dropout}")
    print(f"batch size: {args.batch_size}")
    print(f"epochs: {args.epochs}")

    results = []

    for fold in range(10):
        print(f"\n===== fold {fold} =====")
        train_data, test_data = make_idx_data_cv(
            revs,
            word_idx_map,
            fold,
            max_l=56,
            filter_h=5
        )

        acc = train_one_fold(
            train_data=train_data,
            test_data=test_data,
            embeddings=embeddings,
            static=static,
            batch_size=args.batch_size,
            n_epochs=args.epochs,
            dropout=args.dropout,
            kernel_sizes=(3, 4, 5),
            num_filters=args.num_filters,
            seed=args.seed + fold,
        )

        print(f"cv: {fold}, perf: {acc:.4f}")
        results.append(acc)

    print("\n===== final =====")
    print("fold accuracies:", [round(x, 4) for x in results])
    print("mean accuracy:", round(float(np.mean(results)), 4))


if __name__ == "__main__":
    main()
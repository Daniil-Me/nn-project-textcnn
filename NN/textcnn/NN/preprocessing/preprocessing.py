import re
import numpy as np
import urllib.request
import zipfile
import os
from datasets import load_dataset
from collections import Counter
import torch

def clean_text(text):
    text = text.lower()
    text = re.sub(r"[^a-z0-9\s]", "", text)
    text = re.sub(r"\s+", " ", text).strip()
    return text

def build_vocab(dataset, min_freq=1):
    counter = Counter()
    for split in ["train", "validation", "test"]:
        for sample in dataset[split]:
            tokens = clean_text(sample["text"]).split()
            counter.update(tokens)
    vocab = {"<PAD>": 0, "<UNK>": 1}
    for word, freq in counter.items():
        if freq >= min_freq:
            vocab[word] = len(vocab)
    return vocab

def tokenize(text, vocab, max_len):
    tokens = clean_text(text).split()[:max_len]
    ids = [vocab.get(t, vocab["<UNK>"]) for t in tokens]
    ids += [vocab["<PAD>"]] * (max_len - len(ids))
    return ids

def prepare_split(dataset, split, vocab, max_len):
    X, y = [], []
    for sample in dataset[split]:
        X.append(tokenize(sample["text"], vocab, max_len))
        y.append(sample["label"])
    return torch.tensor(X, dtype=torch.long), torch.tensor(y, dtype=torch.long)

MAX_SEQ_LEN = 50
EMBED_DIM = 300

dataset = load_dataset("dair-ai/emotion")
label_names = dataset["train"].features["label"].names
vocab = build_vocab(dataset)

if not os.path.exists("glove.6B.300d.txt"):
    urllib.request.urlretrieve(
        "https://downloads.cs.stanford.edu/nlp/data/glove.6B.zip",
        "glove.6B.zip"
    )
    with zipfile.ZipFile("glove.6B.zip", "r") as z:
        z.extract("glove.6B.300d.txt")

glove = {}
with open("glove.6B.300d.txt", encoding="utf-8") as f:
    for line in f:
        parts = line.split()
        word = parts[0]
        if word in vocab:
            glove[word] = np.array(parts[1:], dtype=np.float32)

embedding_matrix = np.zeros((len(vocab), EMBED_DIM), dtype=np.float32)
for word, idx in vocab.items():
    if word in glove:
        embedding_matrix[idx] = glove[word]
    else:
        embedding_matrix[idx] = np.random.uniform(-0.25, 0.25, EMBED_DIM)

X_train, y_train = prepare_split(dataset, "train", vocab, MAX_SEQ_LEN)
X_val, y_val     = prepare_split(dataset, "validation", vocab, MAX_SEQ_LEN)
X_test, y_test   = prepare_split(dataset, "test", vocab, MAX_SEQ_LEN)

torch.save({
    "X_train": X_train, "y_train": y_train,
    "X_val": X_val,     "y_val": y_val,
    "X_test": X_test,   "y_test": y_test,
    "embedding_matrix": embedding_matrix,
    "vocab": vocab,
    "label_names": label_names
}, "preprocessed_data.pt")

print("Done! Saved preprocessed_data.pt")

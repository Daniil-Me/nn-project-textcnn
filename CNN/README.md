### CNN Sentiment Analysis

Baseline model for sentiment classification using **Convolutional Neural Networks (TextCNN)**.

The model classifies text samples as:

* **Positive (1)**
* **Negative (0)**

---

## Project Structure

```
cnn/
├── config.py        # Hyperparameters and settings
├── model.py         # CNN architecture (TextCNN)
├── train.py         # Training script (with Early Stopping)
├── evaluate.py      # Evaluation (accuracy, F1, confusion matrix)
├── requirements.txt
└── README.md
```

---

## Dataset

Dataset used: **MR (Movie Reviews)**

* Stored as `mr.p` (pickle file)
* Place `mr.p` in the root directory before running

Download processed dataset:
https://drive.google.com/file/d/1ljozYFVPu9Hb8DJURsOqtdyB-b5uWBZZ/view?usp=sharing

### Dataset Statistics

| Parameter       | Value        |
| --------------- | ------------ |
| Total samples   | 54,981       |
| Positive (y=1)  | 32,322       |
| Negative (y=0)  | 22,659       |
| Vocabulary size | 27,644       |
| Embeddings      | GloVe (300d) |

### Data Splits

* Train: 80% (splits 0–7)
* Validation: 10% (split 8)
* Test: 10% (split 9)

---

## Model Architecture

```
Input (token indices)
    → Embedding Layer (GloVe, 300-dim, frozen)
    → Convolution Layers (filters=100, kernels=[3, 4, 5])
    → ReLU Activation
    → Max-Over-Time Pooling
    → Concatenation
    → Dropout (0.5)
    → Fully Connected Layer (300 → 2)
    → Softmax Output
```

---

## Installation

```bash
pip install -r requirements.txt
```

---

## Usage

### Train the model

```bash
python train.py
```

### Evaluate the model

```bash
python evaluate.py
```

---

## Hyperparameters

Defined in `config.py`:

| Parameter     | Value     |
| ------------- | --------- |
| NUM_FILTERS   | 100       |
| KERNEL_SIZES  | [3, 4, 5] |
| DROPOUT       | 0.5       |
| BATCH_SIZE    | 64        |
| NUM_EPOCHS    | 20        |
| LEARNING_RATE | 0.001     |
| MAX_SEQ_LEN   | 50        |
| SEED          | 42        |

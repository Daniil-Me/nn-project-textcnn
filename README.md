# nn-project-textcnn
TextCNN project for emotion classification (preprocessing, CNN, BiLSTM comparison)

Dataset used:
GoEmotions  
Download from:  
https://www.kaggle.com/datasets/shivamb/go-emotions-google-emotions-dataset

---

# Project Overview

This project focuses on **emotion classification in text** using deep learning methods.
We compare two neural network architectures:

* **TextCNN (Convolutional Neural Networks)**
* **Bi-LSTM (Recurrent Neural Networks)**

The pipeline includes:

* Data preprocessing
* Model training
* Evaluation and comparison

---

# Project Structure

```
project/
├── preprocessing/
├── CNN/
├── bilstm/
├── results
└── README.md
```

---

# Data Preprocessing

## Overview

The original **GoEmotions dataset** contains multiple emotion labels per sentence.
We convert it into a **binary classification task**:

* Positive
* Negative

## Files

* `goemotions_to_polarity.py`
  Converts dataset into polarity format (positive/negative)

* `process_data.py`
  Tokenization, vocabulary creation, cross-validation splits

* `rt-polarity.pos`
  Positive samples

* `rt-polarity.neg`
  Negative samples

* `mr.p`
  Final processed dataset

## Preprocessing Steps

* Remove neutral and ambiguous samples
* Remove mixed-label samples
* Remove duplicates and empty texts
* Convert to rt-polarity format
* Generate numerical dataset

## Run Preprocessing

```bash
python goemotions_to_polarity.py
python2 process_data.py
```

---

# Dataset

| Parameter     | Value        |
| ------------- | ------------ |
| Total samples | 54,981       |
| Positive      | 32,322       |
| Negative      | 22,659       |

### Splits

* Train: 80%
* Validation: 10%
* Test: 10%

---

# CNN Model (TextCNN)

## Description

Baseline CNN model for text classification.

## Architecture

```
Input
 → Embedding (GloVe, frozen)
 → Conv layers (3,4,5)
 → ReLU
 → Max Pooling
 → Concatenation
 → Dropout
 → Fully Connected
 → Softmax
```

## Hyperparameters

| Parameter   | Value   |
| ----------- | ------- |
| Filters     | 100     |
| Kernels     | [3,4,5] |
| Dropout     | 0.5     |
| Batch size  | 64      |
| Epochs      | 20      |
| LR          | 0.001   |
| Max seq len | 50      |

---

# Bi-LSTM Model

## Description

Recurrent neural network capturing sequential dependencies.

## Architecture

```
Input
 → Embedding (GloVe)
 → Bi-LSTM
 → Dropout
 → Fully Connected
 → Softmax
```

## Hyperparameters

| Parameter   | Value |
| ----------- | ----- |
| Hidden size | 128   |
| Layers      | 2     |
| Dropout     | 0.5   |
| Batch size  | 64    |
| Epochs      | 20    |
| LR          | 0.001 |

---

# Training

```bash
python train.py
```

---

# Evaluation

```bash
python evaluate.py
```

Outputs:

* Accuracy
* F1 Score
* Confusion Matrix
* Training curves

---

# Results

| Model   | Accuracy   | F1 Score   |
| ------- | ---------- | ---------- |
| Bi-LSTM | 70.80%     | 0.7102     |
| TextCNN | **72.00%** | **0.7199** |

---

# Key Findings

* TextCNN performs better on short texts
* Early Stopping is crucial
* Simpler models generalize better

---

# Notes

* Python 2 is required for `process_data.py`
* Dataset must be placed in project root
* Pretrained embeddings improve performance

---

# Team Contributions

* Daniil Melnychuk — preprocessing, dataset
* Vladyslav Morhai — Bi-LSTM experiments
* Vladyslav Pasenko — visualization, report
* Viktor Mitchenko — TextCNN implementation

---

# Resources

* Original TextCNN paper: https://arxiv.org/abs/1408.5882
* GoEmotions dataset: https://www.kaggle.com/datasets/shivamb/go-emotions-google-emotions-dataset
* Project repository: https://github.com/Daniil-Me/nn-project-textcnn

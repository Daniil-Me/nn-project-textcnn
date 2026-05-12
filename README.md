# TextCNN Emotion Classification

## Overview
Emotion classification using TextCNN and Bi-LSTM on dair-ai/emotion dataset.
Comparison of CNN-based and RNN-based approaches for 6-class emotion detection.

## Final Results
| Model | Accuracy | F1 macro |
|---|---|---|
| TextCNN Exp1 [3,4,5] | 88.90% | 0.8525 |
| TextCNN Exp2 [2,3,4,5] | 88.65% | 0.8469 |
| TextCNN Exp3 [3] ablation | 89.20% | 0.8535 |
| BiLSTM Exp1 hidden=128 | 91.55% | 0.8851 |
| BiLSTM Exp2 hidden=256 | 91.45% | 0.8816 |
| BiLSTM Exp3 single layer | 88.20% | 0.8491 |

## Key Conclusions
1. BiLSTM outperforms TextCNN — sequential context is important for emotion detection
2. Kernel size variety in TextCNN has minimal impact on short texts
3. BiLSTM depth is critical — 2 layers vs 1 layer gives +4% F1
4. Class imbalance (joy 33.5% vs surprise 3.6%) affects per-class performance

## Project Structure
- preprocessing/ — dataset loading, cleaning, tokenization, GloVe embeddings
- textcnn/ — TextCNN model, training, results
- bilstm/ — BiLSTM model, training, results

## Requirements
pip install datasets torch matplotlib seaborn scikit-learn numpy


# BiLSTM Model

## Architecture
- Embedding layer: GloVe 300d (frozen)
- Bidirectional LSTM (2 layers, hidden=128)
- Dropout (0.5)
- Fully Connected → 6 classes

## Experiments
| Experiment | Hidden | Layers | Accuracy | F1 macro |
|---|---|---|---|---|
| Exp1 baseline | 128 | 2 | 91.55% | 0.8851 |
| Exp2 large | 256 | 2 | 91.45% | 0.8816 |
| Exp3 single layer | 128 | 1 | 88.20% | 0.8491 |

## Key Findings
- 2 layers significantly better than 1 layer (-4% F1) — depth matters
- Increasing hidden size from 128 to 256 gives no improvement
- BiLSTM outperforms TextCNN — sequential context helps for emotion detection

## Training
- Optimizer: Adam, LR=0.001
- Loss: CrossEntropyLoss with class weights
- Early stopping: patience=3
- Batch size: 64

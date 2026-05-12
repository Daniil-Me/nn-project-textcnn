# TextCNN Model

## Architecture
- Embedding layer: GloVe 300d (frozen)
- Parallel Conv1D layers with kernel sizes [3,4,5]
- Global Max Pooling after each conv
- Dropout (0.5)
- Fully Connected → 6 classes

## Experiments
| Experiment | Kernels | Filters | Accuracy | F1 macro |
|---|---|---|---|---|
| Exp1 baseline | [3,4,5] | 100 | 88.90% | 0.8525 |
| Exp2 large | [2,3,4,5] | 200 | 88.65% | 0.8469 |
| Exp3 ablation | [3] | 100 | 89.20% | 0.8535 |

## Key Findings
- Kernel size variety has minimal impact on this dataset
- Short sentences (avg 19.2 words) mean all window sizes capture similar patterns
- Dropout 0.5 effectively prevents overfitting

## Training
- Optimizer: Adam, LR=0.001
- Loss: CrossEntropyLoss with class weights (handles imbalance)
- Early stopping: patience=3
- Batch size: 64

# Preprocessing

## Dataset
- Source: dair-ai/emotion (HuggingFace)
- Classes: sadness, joy, love, anger, fear, surprise
- Train: 16,000 | Validation: 2,000 | Test: 2,000
- Class imbalance: joy 33.5%, sadness 29.2%, surprise 3.6%

## Steps
1. Load dataset from HuggingFace
2. Clean text: lowercase, remove special characters
3. Build vocabulary (17,098 words, min_freq=1)
4. Download GloVe 6B 300d embeddings
5. Build embedding matrix (92.9% coverage)
6. Tokenize and pad all sentences to MAX_SEQ_LEN=50
7. Save to preprocessed_data.pt

## Output
- preprocessed_data.pt — tokenized dataset + embedding matrix

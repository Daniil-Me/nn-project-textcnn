CNN Sentiment AnalysisBaseline model for sentiment classification on the MR (Movie Reviews) dataset using Convolutional Neural Networks.Classifies movie reviews as positive (1) or negative (0).Project Structurecnn/
├── config.py       # All hyperparameters and settings
├── model.py        # CNN model architecture (TextCNN)
├── train.py        # Training script with Early Stopping
├── evaluate.py     # Evaluation script (accuracy, F1, confusion matrix)
├── requirements.txt
└── README.md
DatasetMR (Movie Reviews) dataset — stored as mr.p (pickle file).Place mr.p in the same directory as the scripts before running.Processed dataset (mr.p) is available here:(https://drive.google.com/file/d/1ljozYFVPu9Hb8DJURsOqtdyB-b5uWBZZ/view?usp=sharing)ParameterValueTotal samples54,981Positive (y=1)32,322Negative (y=0)22,659Vocabulary size27,644EmbeddingsGloVe (300 dimensions)Train splits0-7 (80%)Validation split8 (10%)Test split9 (10%)Model ArchitectureInput (token indices)
    → Embedding layer (GloVe, 300-dim, frozen)
    → Convolutional Layers (filters=100, kernels=[3, 4, 5])
    → ReLU Activation & Max-Over-Time Pooling
    → Concatenation
    → Dropout (0.5)
    → Fully Connected (300 → 2)
    → Softmax → Output (0 or 1)
InstallationBashpip install -r requirements.txt


UsageTrain the model:

Bashpython train.py

Evaluate on test set: 

evaluate.py

Hyperparameters
All hyperparameters are defined in config.py:
Parameter,          Value
NUM_FILTERS,        100
KERNEL_SIZES,       "[3, 4, 5]"
DROPOUT,            0.5
BATCH_SIZE,         64
NUM_EPOCHS,         20
LEARNING_RATE,      0.001
MAX_SEQ_LEN,        50
SEED,               42
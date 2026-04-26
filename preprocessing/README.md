Data Preprocessing – TextCNN Project

Overview  
This part of the project focuses on preparing the dataset for the TextCNN model.  
The original dataset contains multiple emotion labels per sentence. We convert it into a binary classification task (positive vs negative) and transform it into the format required by the original TextCNN implementation.

Files Description

goemotions_to_polarity.py  
Main preprocessing script.  
It reads the original CSV dataset, cleans the text, and converts multi-label emotions into two classes: positive and negative.  
It removes neutral, ambiguous, and mixed samples (where both positive and negative labels are present), as well as empty and duplicate texts.  
Output files: rt-polarity.pos and rt-polarity.neg.  

rt-polarity.pos  
File containing positive sentences, one sentence per line.  

rt-polarity.neg  
File containing negative sentences, one sentence per line.  

process_data.py  
Script adapted from the original TextCNN implementation.  
It cleans and tokenizes text, builds vocabulary, creates cross-validation splits, and converts text into numerical format.  
Output file: mr.p.

mr.p  
Final processed dataset used for training the TextCNN model.

Dataset  
We use the GoEmotions dataset:  
https://www.kaggle.com/datasets/google-research/google-goemotions

Final dataset (Google Drive)  
Processed dataset (mr.p) is available here:  
(https://drive.google.com/file/d/1ljozYFVPu9Hb8DJURsOqtdyB-b5uWBZZ/view?usp=sharing)

How to run

Convert dataset to polarity format  
python goemotions_to_polarity.py  
Generate dataset for TextCNN (Python 2 required)  
python2 process_data.py  

Notes
The original TextCNN pipeline requires data in rt-polarity format.
Python 2 is required for process_data.py.
Preprocessing removes noisy and ambiguous samples to improve model quality.

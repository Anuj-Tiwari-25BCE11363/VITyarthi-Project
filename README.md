# VITyarthi-Project
***Handwritten Digit Recognition***

**Overview**

This project implements a machine learning model for recognizing handwritten digits using image data from Kaggle competition datasets. Built in Google Colab, it leverages TensorFlow (Keras API) to achieve nearly perfect classification accuracy.​

**Features**

Kaggle-style workflow with CSV submission generation

Train/validation/test data splits for rigorous evaluation

Regularization using dropout to prevent overfitting

Efficient data handling with panda

**Technologies & Tools Used**

Python 3.7+

TensorFlow & Keras

Google Colab 

Pandas 

Kaggle Datasets (test.csv and train.csv)

**Installation & Setup**

Upload Datasets to Google Drive

Obtain the following CSV files from your Kaggle and upload them to your Google Drive 

1. train.csv = https://www.kaggle.com/datasets/bhavikjikadara/handwritten-digit-recognition?select=train.csv
2. test.csv = https://www.kaggle.com/datasets/bhavikjikadara/handwritten-digit-recognition

Environment Setup

In Colab, run:

bash

!pip install tensorflow pandas scikit-learn

Mount Google Drive

python
from google.colab import drive

drive.mount('/content/drive')

Run the Notebook

Execute all cells, ensuring paths to datasets are correct.

**Instructions for Testing**
After training the model, predictions on test data are generated.

Results are saved as submission.csv in your working directory.

For evaluation, inspect model accuracy from console outputs, and double-check submission format for Kaggle.

**Screenshots**

<img width="1920" height="1080" alt="Screenshot 2025-11-24 094241" src="https://github.com/user-attachments/assets/74e6a443-ac28-4dcd-8a70-3c0b53080445" />
<img width="1920" height="1080" alt="Screenshot 2025-11-24 094256" src="https://github.com/user-attachments/assets/bd3ea7c6-0a6a-4af5-b5a7-7da22c85a51f" />
<img width="1920" height="1080" alt="Screenshot 2025-11-24 094312" src="https://github.com/user-attachments/assets/bdba47f3-5ff0-4975-b291-c5164655e1ba" />
<img width="1920" height="1080" alt="Screenshot 2025-11-24 094319" src="https://github.com/user-attachments/assets/69c16d35-16b3-4b7d-ac09-41e8409c618d" />
<img width="1920" height="1080" alt="Screenshot 2025-11-24 094413" src="https://github.com/user-attachments/assets/4d5b2a38-c45e-4a54-80e8-edb98263cdd0" />
<img width="1920" height="1080" alt="Screenshot 2025-11-24 094421" src="https://github.com/user-attachments/assets/c7290c45-7e4c-429e-ab1e-8a5ed8e055ba" />
<img width="1920" height="1080" alt="Screenshot 2025-11-24 094426" src="https://github.com/user-attachments/assets/f8c518ae-581f-45e4-b1bd-7df2e70f6d4e" />









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
<img width="1920" height="1080" alt="Screenshot 2025-11-24 094241" src="https://github.com/user-attachments/assets/2efcda59-890c-40a6-a8e9-e038eba2eca4" />
<img width="1920" height="1080" alt="Screenshot 2025-11-24 094256" src="https://github.com/user-attachments/assets/6e665242-26c9-438b-952b-e8b696805eea" />
<img width="1920" height="1080" alt="Screenshot 2025-11-24 094312" src="https://github.com/user-attachments/assets/b611648f-8b43-485a-8f7c-940ffb194bd9" />
<img width="1920" height="1080" alt="Screenshot 2025-11-24 094319" src="https://github.com/user-attachments/assets/eab08a1c-f2ba-45e9-84c7-cc8e9228d311" />
<img width="1920" height="1080" alt="Screenshot 2025-11-24 094413" src="https://github.com/user-attachments/assets/e041478e-c356-43ac-bd9b-ac0553388017" />
<img width="1920" height="1080" alt="Screenshot 2025-11-24 094421" src="https://github.com/user-attachments/assets/931e16d1-fbe4-4629-bea8-d85282c6a876" />
<img width="1920" height="1080" alt="Screenshot 2025-11-24 094426" src="https://github.com/user-attachments/assets/9d13ad96-530a-47ac-ab1b-3b0e6deb121d" />

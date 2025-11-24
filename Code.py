#VITyarthi project
#required libraries
!pip install tensorflow pandas scikit-learn
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from google.colab import drive
import pandas as pd      
drive.mount('/content/drive')
#Model details
model = keras.Sequential([
    layers.Conv2D(32, (3,3), activation="relu", input_shape=(28,28,1)),
    layers.MaxPooling2D((2,2)),
    layers.Conv2D(64, (3,3), activation="relu"),
    layers.MaxPooling2D((2,2)),
    layers.Flatten(),
    layers.Dense(64, activation="relu"),
    layers.Dropout(0.5),
    layers.Dense(10, activation="softmax")
])

model.compile(optimizer = "adam", loss = "categorical_crossentropy", metrics = ["accuracy"])


#datasets taken from kaggle
train_path = "/content/drive/MyDrive/VITYARTHI/train.csv"
test_path  = "/content/drive/MyDrive/VITYARTHI/test.csv"

train_df = pd.read_csv(train_path)
test_df  = pd.read_csv(test_path)

print(train_df.shape)
print(test_df.shape)

from tensorflow.keras.utils import to_categorical
from sklearn.model_selection import train_test_split
import pandas as pd

y = train_df["label"].values
X = train_df.drop(columns=["label"]).values


X = X.astype("float32") / 255.0
test_data = test_df.values.astype("float32") / 255.0

X = X.reshape(-1, 28, 28, 1)
test_data = test_data.reshape(-1, 28, 28, 1)

y_cat = to_categorical(y, num_classes=10)

X_train, X_val, y_train, y_val = train_test_split(
    X, y_cat, test_size=0.1, random_state=42, stratify=y
)

print(X_train.shape, X_val.shape, test_data.shape)

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.losses import CategoricalCrossentropy

model = Sequential()


model.add(Conv2D(32, (3, 3), activation="relu", input_shape=(28, 28, 1)))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Conv2D(64, (3, 3), activation="relu"))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Conv2D(128, (3, 3), activation="relu"))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Flatten())
model.add(Dense(128, activation="relu"))
model.add(Dropout(0.5))
model.add(Dense(10))

model.compile(
    optimizer=Adam(learning_rate=0.001),
    loss=CategoricalCrossentropy(from_logits=True),
    metrics=["accuracy"],
)

model.summary()

epochs = 15
batch_size = 128

history = model.fit(
    X_train, y_train,
    epochs=epochs,
    batch_size=batch_size,
    validation_data=(X_val, y_val),
    verbose=2
)

import numpy as np
pred_probs = model.predict(test_data)
pred_labels = np.argmax(pred_probs, axis=1)

print(pred_labels[:10])

submission = pd.DataFrame({
    "ImageId": np.arange(1, len(pred_labels) + 1),
    "Label": pred_labels
})

submission.to_csv("submission.csv", index=False)
submission.head()
                    

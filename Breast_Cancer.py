"""
*****************************************************************************************
This model was built by: Djaid Walid

__________________________________________________________________________________________________
                                   Contacts                                                      |
__________________________________________________________________________________________________
Github     | https://github.com/waliddj                                                          |
Linkedin   | www.linkedin.com/in/walid-djaid-375777229                                           |
Instagram  | https://www.instagram.com/d.w.science?igsh=MWlnMmNpOTM2OW0xaA%3D%3D&utm_source=qr   |
__________________________________________________________________________________________________

*****************************************************************************************
"""
import tensorflow as tf
from tensorflow.keras import Sequential
from tensorflow.keras.layers import Conv2D, MaxPool2D, Flatten, Dense
from tensorflow.keras.preprocessing.image import ImageDataGenerator
import pathlib
import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
# Load the data
data_dir = 'C:/Users/walid/.cache/kagglehub/datasets/obulisainaren/multi-cancer/versions/3/Multi Cancer/Multi Cancer/Breast Cancer/'
# split the train and test data derectories
train_dir = 'C:/Users/walid/.cache/kagglehub/datasets/obulisainaren/multi-cancer/versions/3/Multi Cancer/Multi Cancer/Breast Cancer/train'
test_dir = 'C:/Users/walid/.cache/kagglehub/datasets/obulisainaren/multi-cancer/versions/3/Multi Cancer/Multi Cancer/Breast Cancer/test'


IMAGE_SIZE = (224,224)
BATCH_SIZE= 32

train_data = tf.keras.preprocessing.image_dataset_from_directory(
    train_dir,
    image_size= IMAGE_SIZE,
    label_mode='binary',
    batch_size=BATCH_SIZE,
)
test_data = tf.keras.preprocessing.image_dataset_from_directory(
    test_dir,
    image_size= IMAGE_SIZE,
    label_mode='binary'
)

# Get the class names from the train directory
class_names = train_data.class_names

# Data preprocessing
# > **Note:** the data was already augmented by the dataset creators


# Build the model
model = Sequential([
    Conv2D(filters=10,kernel_size=3, input_shape=(224,224,3), activation='relu'),
    Conv2D(10,3,activation='relu'),
    MaxPool2D(),
    Flatten(),
    Dense(1, activation='sigmoid')
])

model.compile(
    loss = tf.keras.losses.BinaryCrossentropy(),
    optimizer = 'adam',
    metrics=['accuracy']
)

history = model.fit(train_data,
                    epochs=6,
                    steps_per_epoch=len(train_data),
                    validation_data = test_data,
                    validation_steps=len(test_data))
# Evaluate the model
pd.DataFrame(history.history). plot()
plt.xlabel('epoch')
plt.ylabel('loss')
plt.show()


# Save the model

model.save("C:/Users/walid/Desktop/Breast_Cancer.keras")



# ===================================================
# Experiments
# ===================================================

"""
# ==================================================
                      Model 1
# ===================================================
# Architecture
- ```Conv2D``` layer *(```input``` layer)* with an ```input shape``` of ```(224,224,3)```, 10 ```filters```, the ```kernel_size``` = ```3```, and a ```ReLU``` activation method.
- Another ```Conv2D``` layer with same parameters as the first one but without the ```input shape``` value.
- ```MaxPool2D``` layer with a default value ```(2,2)```.
- ```Flatten```layer followed by a ```Dense``` (```output```) layer with a ```Sigmoid``` activation method.
- Optimizer ```Adam``` with default value ```0.001```
- Number of ```epochs```: ```6```.
# Loss/Accuracy:
|Metric|Accuracy|Loss|
|------|-----|---|
|Train data| 98.57%| 0.0617|
|Test data| 94.20%|0.161|

# ==================================================
                      Model 2
# ===================================================
# Architecture
 - Addition of  2 ```Conv2D```.
# Loss/Accuracy:
|Metric|Accuracy|Loss|
|------|-----|---|
|Train data| %| 0.|
|Test data| %|0.|


# ==================================================
                      Model 3
# ===================================================
# Architecture
 - Addition of  a second ```MaxPool2D``` layer.
# Loss/Accuracy:
|Metric|Accuracy|Loss|
|------|-----|---|
|Train data| %| 0.|
|Test data| %|0.|

# ==================================================
                      Model 4
# ===================================================
# Architecture
 - Addition of  a ```data_augmentation``` layer: 
 ```tf.keras.layers.RandomFlip("horizontal_and_vertical")``` and ```tf.keras.layers.RandomRotation(0.2)```.
# Loss/Accuracy:
|Metric|Accuracy|Loss|
|------|-----|---|
|Train data| 89%| 0.289|
|Test data| 91%|0.253|

# ==================================================
                      Model 5
# ===================================================
# Architecture
- Same architecture as the Model 1
 - Addition of  a ```data_augmentation``` layer.
# Loss/Accuracy:
|Metric|Accuracy|Loss|
|------|-----|---|
|Train data| 88%| 0.295|
|Test data| 89%|0.3171|


"""

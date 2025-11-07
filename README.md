# Breast_Cancer_Diagnosis_v1
A Convolutional Neuron Network based model for breast cancer diagnosis.

# Dataset
---

The dataset used for this model is the [Breast cancer dataset](https://www.kaggle.com/datasets/djaidwalid/kidney-cancer-dataset/data)

*Citation:*: https://www.kaggle.com/datasets/djaidwalid/kidney-cancer-dataset/data

---
## Dataset structure:
This dataset is divided into two main directories `train` and `test` directory each divided into two other directories `breast_malignant` and `breast_benign` following this structure:
```
\train
  |
  |__\breast_malignant
      |
      |__4000 images
  |
  |__\breast_benign
      |
      |__4000 images

\test
  |
  |__\breast_malignant
      |
      |__1000 images
  |
  |__\breast_benign
      |  
      |__1000 images

```
## Dataset details:

|Path|	Subclass|Description|
|-----|-----------|--------------|
|breast_benign|	Benign|	Non-cancerous (healthy) breast tissues|
|breast_malignant|	Malignant|	TCancerous breast tissues|

*Source: Collected from the Breast Cancer dataset by Anas Elmasry on Kaggle.*

## Data augmentation:
the data was augmentend by the original author of the dataset using Kera's `ImageDataGeberator` *[1]* and The augmentations include:
-  Rotation: Up to 10 degrees.
-   Width & Height Shift: Up to 10% of the total image size.
-   Shearing & Zooming: 10% variation.
-   Horizontal Flip: Randomly flips images for additional diversity.
-   Brightness Adjustment: Ranges from 0.2 to 1.2 for varying light conditions.

The parameters used for augmentation:
```python
from keras.preprocessing.image import ImageDataGenerator

datagen = ImageDataGenerator(
    rotation_range=10,
    width_shift_range=0.1,
    height_shift_range=0.1,
    shear_range=0.1,
    zoom_range=0.1,
    horizontal_flip=True,
    fill_mode='nearest',
    brightness_range=[0.2, 1.2]
)

```
*For more details visit the [Breast cancer dataset]() kaggle page* 

# Code architecture:
```python
import tensorflow as tf
from tensorflow.keras import Sequential
from tensorflow.keras.layers import Conv2D, MaxPool2D, Flatten, Dense
from tensorflow.keras.preprocessing.image import ImageDataGenerator
import pathlib
import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
```
## Dataset
### Load the dataset from directory
```python
data_dir = 'C:/Users/walid/.cache/kagglehub/datasets/obulisainaren/multi-cancer/versions/3/Multi Cancer/Multi Cancer/Breast Cancer/'
```
### Split the training and test data directories
```python
train_dir = 'C:/Users/walid/.cache/kagglehub/datasets/obulisainaren/multi-cancer/versions/3/Multi Cancer/Multi Cancer/Breast Cancer/train'
test_dir = 'C:/Users/walid/.cache/kagglehub/datasets/obulisainaren/multi-cancer/versions/3/Multi Cancer/Multi Cancer/Breast Cancer/test'
```
### Split the training and test data using `tf.keras.preprocessing.image_dataset_from_directory`
```python
IMAGE_SIZE = (224,224) # Fix the image sizes of the train and test data
BATCH_SIZE= 32 # fix the data batch size

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
```
### Get the class names (labels) from the `train_data`
```python
class_names = train_data.class_names
```
## Build the CNN model:

---

### Model Architecture:
- ```Conv2D``` layer *(```input``` layer)* with an ```input shape``` of ```(224,224,3)```, 10 ```filters```, the ```kernel_size``` = ```3```, and a ```ReLU``` activation method.
- Another ```Conv2D``` layer with same parameters as the first one but without the ```input shape``` value.
- ```MaxPool2D``` layer with a default value ```(2,2)```.
- ```Flatten```layer followed by a ```Dense``` (```output```) layer with a ```Sigmoid``` activation method.
- Optimizer ```Adam``` with default value ```0.001```
- Number of ```epochs```: ```6```.
  
---

### Create the model
```python
model = Sequential([
    Conv2D(filters=10,kernel_size=3, input_shape=(224,224,3), activation='relu'),
    Conv2D(10,3,activation='relu'),
    MaxPool2D(),
    Flatten(),
    Dense(1, activation='sigmoid')
])

```
### Compile the model
```python
model.compile(
    loss = tf.keras.losses.BinaryCrossentropy(),
    optimizer = 'adam',
    metrics=['accuracy']
)
```
### Fit the model to the `train_data` and validation data (`test_data`) while keep tracking the model's history for evaluation:
```python
history = model.fit(train_data,
                    epochs=6,
                    steps_per_epoch=len(train_data),
                    validation_data = test_data,
                    validation_steps=len(test_data))
```

# Model evaluation
## Plot Loss&Accuracy
<img width="640" height="480" alt="loss_accuracy_1" src="https://github.com/user-attachments/assets/3c73294b-2e2f-438d-90fe-90637fae7953" />

|Metric|Accuracy|Loss|
|------|-----|---|
|Train data| 98.57%| 0.0617|
|Test data| 94.20%|0.161|

## Confusion matrix

<img width="640" height="480" alt="cm_1" src="https://github.com/user-attachments/assets/fc2d8421-7110-4487-83eb-c0384b2a6a3f" />

> The `train` and `test` data accuracies are high. However, the confusion matrix shows that the model is not accurate.

## Predictions:
<img width="420" height="420" alt="model_1_pred_1_keep" src="https://github.com/user-attachments/assets/a9244bd8-8756-42fc-b58a-493a8c9c51bc" />
<img width="420" height="420" alt="model_1_pred_0" src="https://github.com/user-attachments/assets/47e36682-bb6c-4029-8d0f-26909bf392e7" />
<img width="420" height="420" alt="model_1_pred_0_keep" src="https://github.com/user-attachments/assets/45a1aa77-eb38-466d-81b5-25dbda303647" />
<img width="420" height="420" alt="model_1_pred_4" src="https://github.com/user-attachments/assets/312ecbb5-d652-4acd-878d-ec3c171ca398" />



# Nex Step:
Build a more accurate model.


---

# Appendix (Experiments)
## Model 1
### Architecture
- ```Conv2D``` layer *(```input``` layer)* with an ```input shape``` of ```(224,224,3)```, 10 ```filters```, the ```kernel_size``` = ```3```, and a ```ReLU``` activation method.
- Another ```Conv2D``` layer with same parameters as the first one but without the ```input shape``` value.
- ```MaxPool2D``` layer with a default value ```(2,2)```.
- ```Flatten```layer followed by a ```Dense``` (```output```) layer with a ```Sigmoid``` activation method.
- Optimizer ```Adam``` with default value ```0.001```
- Number of ```epochs```: ```6```.
### Loss&Accuracy:
<img width="640" height="480" alt="loss_accuracy_1" src="https://github.com/user-attachments/assets/3c73294b-2e2f-438d-90fe-90637fae7953" />
|Metric|Accuracy|Loss|
|------|-----|---|
|Train data| 98.57%| 0.0617|
|Test data| 94.20%|0.161|

### Confusion matrix
<img width="640" height="480" alt="cm_1" src="https://github.com/user-attachments/assets/fc2d8421-7110-4487-83eb-c0384b2a6a3f" />

## Model 2
### Architecture modifications
 - Addition of  2 ```Conv2D```.
### Loss&Accuracy:
<img width="640" height="480" alt="loss_accuracy_2" src="https://github.com/user-attachments/assets/c2441f4a-594b-452b-aa90-e4f9a0e1ce55" />

|Metric|Accuracy|Loss|
|------|-----|---|
|Train data| 89.78%| 0.15|
|Test data|84.10 %|0.362|

### Confusion matrix
<img width="640" height="480" alt="cm_2" src="https://github.com/user-attachments/assets/c611f612-0bd9-4e9d-99b2-161706ae8229" />

## Model 3
### Architecture modifications
### Loss&Accuracy:
<img width="640" height="480" alt="loss_accuracy_3" src="https://github.com/user-attachments/assets/efc73c59-74b3-4ed9-ad73-231b3128a13b" />

|Metric|Accuracy|Loss|
|------|-----|---|
|Train data| 90.10%| 0.18|
|Test data|87.3 %|0.322|

### Confusion matrix
<img width="640" height="480" alt="cm_3" src="https://github.com/user-attachments/assets/2238b519-2602-4ab1-ab6a-5d50a9800277" />

## Model 4 
### Architecture modifications
 - Addition of  a ```data_augmentation``` layer: 
 ```tf.keras.layers.RandomFlip("horizontal_and_vertical")``` and ```tf.keras.layers.RandomRotation(0.2)```.
### Loss&Accuracy:
<img width="640" height="480" alt="loss_accuracy_4" src="https://github.com/user-attachments/assets/f1f656d3-da32-48e7-915b-85844684936e" />

|Metric|Accuracy|Loss|
|------|-----|---|
|Train data| 89%| 0.289|
|Test data| 91%|0.253|

### Confusion matrix
<img width="640" height="480" alt="cm_4" src="https://github.com/user-attachments/assets/43dfcd76-58be-4d87-a3d1-9105edf35f93" />

## Model 5
### Architecture modifications
- Same architecture as the Model 1
- Addition of  a ```data_augmentation``` layer.
### Loss&Accuracy:
<img width="640" height="480" alt="loss_accuracy_5" src="https://github.com/user-attachments/assets/c06128a6-5fc6-470d-915d-6079127e473c" />


|Metric|Accuracy|Loss|
|------|-----|---|
|Train data| 88%| 0.295|
|Test data| 89%|0.3171|

### Confusion matrix
<img width="640" height="480" alt="cm_5" src="https://github.com/user-attachments/assets/b5a008f0-699e-453f-9e47-c71ca28a3b12" />

## Conclusion
`Model 1` is the model with the best performance

---

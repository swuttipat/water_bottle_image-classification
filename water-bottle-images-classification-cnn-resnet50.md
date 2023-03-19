<a id='back_to_top'></a>
# Water Bottle Images Classification-CNN & ResNet50
"Classifying Water Bottle Images Based on Water Level Using Machine Learning"

**Created By**: Wuttipat S. <br>
**Created Date**: 2023-02-10 <br>
**Status**: <span style="color:green">Completed</span>

#### Update: 
- **Version7**
    - Reduce image input_size 128->64
- **Version6**
    - Perform error analysis, fixed code error
    - Bring the pre-trained model for performance comparing
- **Version5**
    - Train and run model with resampling set, which return accucary
        - 49% with GridSearchCV (Further tune)
- **Version4**
    - Add Data Resampling
    - Add load images included .*png* file
- **Version3**
    - Add confusion matrix plot

# Table of contents

### 1. [Introduction](#introduction)
- [Project Objective](#project_objective)
- [Dataset Description](#dataset_description)
- [About this directory](#about_this_directory) 

### 2. [My Tasks](#my_tasks)
### 3. [Importing Data from the Directory](#load_dataset) 
### 4. [Data Preprocessing](#data_preprocessing) 
1. [Data Augmentation & Data Resampling](#data_augmentation)
2. [Nomalizing images value](#nomalizing_images_value)
3. [Convert the labels into one-hot encoder array](#convert_the_labels_into_one_hot_encoder_array)

### 5. [Machine Learning Model](#machine_learning_model) 
* [CNN model](#cnn_model)
* [Modified pre-trained model(ResNet50)](#resnet50)

### 6. [Hyperparameter Tuning](#hyperparameter_tuning)
- [GridSearchCV](#gridsearchcv)

##  [Note](#note)

<br><br><br>

---
<a id='introduction'></a>
# Introduction

<a id='project_objective'></a>
## Project Objective: 
The main objective of this project is to develop a machine learning model that can accurately classify water bottle images based on their water levels. The model will be trained on a dataset of water bottle images, with each image being labeled as Full water level, Half water level, or Overflowing. The goal is to develop a model that can accurately classify a given water bottle image based on its water level.

<a id='dataset_description'></a>
## Dataset Description: 
The dataset consists of water bottle images that have been classified based on the level of water inside the bottle. There are three categories of images: Full water level, Half water level, and Overflowing. Each category contains a number of images of water bottles with the corresponding water level. The purpose of the dataset is to be used for an image classification problem, where a machine learning model is trained to classify the water level of a given water bottle image.

The dataset is intended to be used for training and testing a machine learning model for image classification. The model will be trained on the provided images, with each image being labeled as either Full water level, Half water level, or Overflowing. The goal of the model is to accurately classify a given water bottle image based on its water level.

The dataset consists of a number of water bottle images, each of which has been classified based on the water level inside the bottle. The images in the Full water level category show water bottles with the maximum possible amount of water inside, while the images in the Half water level category show water bottles with roughly half the maximum amount of water inside. The images in the Overflowing category show water bottles with more water inside than the maximum capacity of the bottle, resulting in water spilling out.

The dataset is likely to be useful for a variety of applications, such as developing automated systems for monitoring and managing water levels in containers or for use in a general image classification problem. The dataset may also be useful for research purposes, as it allows for the development and testing of machine learning models for image classification tasks.

<a id='about_this_directory'></a>
### About this directory
"This folder contains 308 images of water bottles with full water levels. The images show a variety of water bottle sizes and shapes, and are captured from a range of angles. The water bottles are made of plastic and are in good condition. These images could be useful for training a machine learning model to recognize full water levels in water bottles."
#### The dataset contains with 3 folder:
1. Full Water Level - 308 images of full water bottle
2. Half water lavel - 139 images of half water bottle
3. Overflowing - 39 images of overflowing bottle


---
<a id='my_tasks'></a>
## My Tasks - Image Classification Project


1. Create, train, and validate **CNN** model for water bottle images classification:
    - Load dataset of water bottle images, and split it into training, validation, and testing sets.
    - Design a convolutional neural network (CNN) architecture that is suitable for image classification, and implement it using a deep learning TensorFlow.
    - Train the CNN using the training set, and validate it using the validation set to check for overfitting.
    - Evaluate the trained model using the testing set, and report its accuracy and other relevant metrics.

2. Conduct the pre-trained model **ResNet50** into comparison with prior created model:
    - Load the pre-trained ResNet50 model into the notebook and modify it to suit our specific image classification problem.
    - Train the modified ResNet50 model using the same training set as the prior created model.
    - Evaluate the performance of the modified ResNet50 model on the validation and testing sets, and compare it to the prior created model.

3. Build the **GridSearchCV** searching through optimize parameter for the model:
    - Choose the relevant hyperparameters that can be tuned for the CNN models, such as the learning rate, batch size, and optimizer algorithm.
    - Implement a grid search to exhaustively search through the hyperparameter space and find the optimal combination of hyperparameters that yields the best performance on the validation set.
    - Train the CNN model with the optimized hyperparameters and evaluate its performance on the testing set.

---
<a id='load_dataset'></a>
## Importing Data from the Directory
I started by importing the data from the directory. Using the `OS` module in python to access the directory and its sub-directories. Then use the `OpenCV` library to read the image files and convert them into arrays that can be processed by the machine learning model.


```python
import os
import cv2
import numpy as np

import warnings
warnings.filterwarnings('ignore') # Hide all warnings


data = []
labels = []
input_size = 64
image_size = (input_size, input_size)

# Access the directory and sub-directories and so on
# directory = "water-bottle-dataset"
directory = "/kaggle/input/water-bottle-dataset"

# Extract all images file inside the folders and stored them into list
for sub_folder in os.listdir(directory):
    sub_folder_path = os.path.join(directory, sub_folder)
    for sub_sub_folder in os.listdir(sub_folder_path):
        sub_sub_folder_path = os.path.join(sub_folder_path, sub_sub_folder)
        for image_file in os.listdir(sub_sub_folder_path):
            if image_file.endswith(".jpeg") or image_file.endswith(".png"): # Check if the file ends with either '.jped' or '.png'
                image_path = os.path.join(sub_sub_folder_path, image_file)
                # Read the image using OpenCV
                image = cv2.imread(image_path) #the decoded images stored in **B G R** order.
                # Resize the image to a standard size
                image = cv2.resize(image, image_size)
                # Append the image to the data list
                data.append(image)
                # Append the label to the labels list
                labels.append(sub_folder)

# Convert the data and labels lists into numpy arrays
data = np.array(data)
labels = np.array(labels)

# Print the dimension of dataset
print(f'data shape:{data.shape}')
print(f'labels shape:{labels.shape}')
```

    data shape:(486, 64, 64, 3)
    labels shape:(486,)
    


```python
'''
See how many numbers of each labels
'''

import pandas as pd
import matplotlib.pyplot as plt
df = pd.DataFrame({"label":labels})
df.value_counts().plot(kind='bar')
plt.xticks(rotation = 0) # Rotates X-Axis Ticks by 45-degrees
plt.show()
```


    
![png](water-bottle-images-classification-cnn-resnet50_files/water-bottle-images-classification-cnn-resnet50_11_0.png)
    


---
<a id='data_preprocessing'></a>
## Cleaning and Data Preprocessing
Now that we have imported the data, and need to clean and preprocess the data so that it can be used to train the machine learning model. The following preprocessing steps will be performed:

1. **Generate augmented data**. The augmented data is concatenated with the original data to increase the size of the training data.
2. **Resampling** is the process of randomly adding or removing data from the dataset to balance the classes. There are two main resampling techniques:

    - Undersampling: Undersampling involves randomly removing data from the majority class so that the number of samples in the majority class is the same as the number of samples in the minority class.

    - Oversampling: Oversampling involves randomly replicating data from the minority class so that the number of samples in the minority class is the same as the number of samples in the majority class.
    
    In this notebook I will use *oversampling*.
2. **Normalizing** the pixel values to a range between 0 and 1
3. **Converting** the labels into one-hot encoded arrays

<a id='data_augmentation'></a>
#### 1. Data Augmentation & Data Resampling

* In the begining of developing the model I generate images by multiplte them for original dateset, the accuracy given is above 80%. But I realize that not answer I look for, since the majority label of dataset is "Full Water Level". My model are overfitting with the training data, In addition the test set also engoving with "Full Water Level", thus it typical to return high accuracy score.
* Next step I bring a **Data Resampling** to fix the overfitting problem. Trainning and test set are equally labels generated.


```python

# Generate augmented data

from keras.preprocessing.image import ImageDataGenerator

# Load the data
X = data # array of preprocessed data
y = labels # array of labels
n_gen = 40

# Create data generator
datagen = ImageDataGenerator(
        rotation_range=0, #0
        width_shift_range=0.2,
        height_shift_range=0.2,
        shear_range=0.2,
        zoom_range=0.2,
        horizontal_flip=True,
        fill_mode='nearest')

# Fit the data generator on the data
datagen.fit(X)

# Generate augmented data
X_augmented, y_augmented = [], []

'''
1st Option multiple dataset with same ratio
'''
# # Non resampling
# for X_batch, y_batch in datagen.flow(X, y, batch_size=32):
#     X_augmented.append(X_batch)
#     y_augmented.append(y_batch)
#     if len(X_augmented) >= 100: # Setting generated augmented data
#         break

'''
2nd Option resampling with equaly labels ratio
'''
# With resampling
for X_batch, y_batch in datagen.flow(X[:308], y[:308], batch_size=32):
    X_augmented.append(X_batch)
    y_augmented.append(y_batch)
    if len(X_augmented) >= n_gen: # Setting generated augmented data
        break
        
for X_batch, y_batch in datagen.flow(X[308:447], y[308:447], batch_size=32):
    X_augmented.append(X_batch)
    y_augmented.append(y_batch)
    if len(X_augmented) >= n_gen*2.3: # Setting generated augmented data
        break
        
for X_batch, y_batch in datagen.flow(X[447:], y[447:], batch_size=32):
    X_augmented.append(X_batch)
    y_augmented.append(y_batch)
    if len(X_augmented) >= n_gen*4.2: # Setting generated augmented data
        break

# Concatenate augmented data with original data
data = np.concatenate((X, np.concatenate(X_augmented)))
labels = np.concatenate((y, np.concatenate(y_augmented)))

print(f"data augmented shape : {data.shape}")
print(f"labels augmented shape : {labels.shape}")

import pandas as pd
df = pd.DataFrame({"label":labels})
df.value_counts()

```

    data augmented shape : (4654, 64, 64, 3)
    labels augmented shape : (4654,)
    




    label            
    Half water level     1593
    Full  Water level    1540
    Overflowing          1521
    dtype: int64




```python
'''
See how many numbers of each labels. 
After I regenerated data.
'''

import pandas as pd
import matplotlib.pyplot as plt
df = pd.DataFrame({"label":labels})
df.value_counts().plot(kind='bar')
plt.xticks(rotation = 0) # Rotates X-Axis Ticks by 45-degrees
plt.show()
```


    
![png](water-bottle-images-classification-cnn-resnet50_files/water-bottle-images-classification-cnn-resnet50_15_0.png)
    


#### Train and Test Split

* Although mostly neural network previde the train-test split function for itself.
* I want to see a result more visualize by plot a *Confusion matrix* from *Predicted of test* and *True labels of test*. 


```python
from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(data, labels, test_size=0.2, random_state=42)

data = X_train # Split training data
labels = y_train # Split training labels

X_test = X_test # Test data
y_test = y_test # Test labels
```


```python
import pandas as pd

print(f'data shape:{data.shape}')
print(f'labels shape:{labels.shape}')
df = pd.DataFrame({"label":labels})
print(df.value_counts())
print("")
print(f'test_date shape:{X_test.shape}')
print(f'test_labels shape:{y_test.shape}')
df = pd.DataFrame({"test_labels":y_test})
print(df.value_counts())
```

    data shape:(3723, 64, 64, 3)
    labels shape:(3723,)
    label            
    Half water level     1286
    Overflowing          1221
    Full  Water level    1216
    dtype: int64
    
    test_date shape:(931, 64, 64, 3)
    test_labels shape:(931,)
    test_labels      
    Full  Water level    324
    Half water level     307
    Overflowing          300
    dtype: int64
    

<a id='nomalizing_images_value'></a>
#### 2. Nomalizing images value


```python
# Normalize the pixel values to a range between 0 and 1
data = data / 255.0
X_test = X_test / 225.0
```

<a id='convert_the_labels_into_one_hot_encoder_array'></a>
#### 3. Convert the labels into one-hot encoder array
Since model create prediction output as (n, 3) dimension array. Converting labels into same type is require for calcuate the model's accuracy and loss.


```python
labels = labels
# Convert the labels into one-hot encoded arrays
labels_one_hot = np.zeros((labels.shape[0], 3))

for i, label in enumerate(labels):
    if label == "Full  Water level":
        labels_one_hot[i, 0] = 1
    elif label == "Half water level":
        labels_one_hot[i, 1] = 1
    else:
        labels_one_hot[i, 2] = 1
        
```


```python
# Show converted output
print(labels_one_hot[0])
```

    [0. 1. 0.]
    


```python
'''
Show a sample of images from the dataset
'''

import matplotlib.pyplot as plt

# Load the data
data = data

# choose 20 random indices
indices = np.random.randint(0, len(data), 20)

# Get 20 sample images
sample_images = data[indices]

# Plot the images
fig = plt.figure(figsize=(10,10))
for i, img in enumerate(sample_images):
    plt.subplot(4, 5, i+1)
    plt.imshow(img)
    plt.axis('off')
    plt.title(labels[indices[i]])
    
plt.show()
```


    
![png](water-bottle-images-classification-cnn-resnet50_files/water-bottle-images-classification-cnn-resnet50_24_0.png)
    


#### Generate augmented images files (Optional)


```python
'''
# Save augmented images to specific directory --- Uncomment to use
# create new directory to save augmented images
import os

# Check existing directory, if not: crate new directory
if not os.path.exists("augmented_images"):
    os.makedirs("augmented_images")

augmented_data = data
labels = labels
# loop through each image in the augmented data
for i, image in enumerate(augmented_data):
    # convert the image back to its original form
    image = (image).astype("uint8")
    
    # save the image to the new directory
    cv2.imwrite(f"augmented_images/augmented_{labels[i]}_{i}.jpeg", image)
'''
```




    '\n# Save augmented images to specific directory --- Uncomment to use\n# create new directory to save augmented images\nimport os\n\n# Check existing directory, if not: crate new directory\nif not os.path.exists("augmented_images"):\n    os.makedirs("augmented_images")\n\naugmented_data = data\nlabels = labels\n# loop through each image in the augmented data\nfor i, image in enumerate(augmented_data):\n    # convert the image back to its original form\n    image = (image).astype("uint8")\n    \n    # save the image to the new directory\n    cv2.imwrite(f"augmented_images/augmented_{labels[i]}_{i}.jpeg", image)\n'



---
<a id='machine_learning_model'></a>
## Machine Learning Model
Finally, we will build, train, and evaluate machine learning models for the image classification problem. I will use the `Keras` and `TensorFlow` library in Python to build and train the models.

Here is a list of layers available in TensorFlow along with a brief explanation about each:

- **Dense Layer**: A dense layer is a fully connected layer where every input node is connected to every output node. It is the most basic layer in TensorFlow and is used for constructing deep neural networks.

- **Convolutional Layer**: A convolutional layer is used for image classification tasks. It uses filters to extract features from the input data.

- **Dropout Layer**: A dropout layer is used to prevent overfitting by randomly dropping out neurons during training.

- **Batch Normalization** Layer: A batch normalization layer is used to normalize the inputs to a deep neural network. This helps to improve the training process and prevent overfitting.

- **Pooling Layer**: A pooling layer is used to reduce the dimensionality of the input data. It is commonly used in image classification tasks to reduce the size of the input image.

- **Flatten Layer**: A flatten layer is used to convert the input data from a high-dimensional array to a one-dimensional array. This is used in image classification tasks to prepare the input data for the fully connected layer.

<a id='cnn_model'></a>
### CNN model


```python
def run_custom_model(batch_size, epochs):
    
    import tensorflow as tf
    from tensorflow import keras
    from tensorflow.keras import layers
    from tensorflow.keras.optimizers import Adam, SGD

    # set seed value for randomization
    # np.random.seed(42)
    tf.random.set_seed(42)

    # Build the model using a Convolutional Neural Network
    model = keras.Sequential([
        keras.layers.Conv2D(32, (3,3), activation='relu', input_shape=(input_size,input_size,3)),
        keras.layers.Conv2D(32, (3,3), activation='relu'),
        keras.layers.MaxPooling2D(2,2),
        keras.layers.Dropout(0.2),

        keras.layers.Conv2D(64, (3,3), activation='relu'),
        keras.layers.Conv2D(64, (3,3), activation='relu'),
        keras.layers.MaxPooling2D(2,2),
        keras.layers.Dropout(0.2),

        keras.layers.Conv2D(256, (3,3), activation='relu'),
        keras.layers.Conv2D(256, (3,3), activation='relu'),
        keras.layers.MaxPooling2D(2,2),
        keras.layers.Dropout(0.2),

        keras.layers.Flatten(),
        keras.layers.Dense(1024, activation='relu'),
        keras.layers.Dropout(0.5),
        keras.layers.Dense(3, activation='softmax')
    ])


    # Compile the model
    model.compile(optimizer=Adam(), loss='categorical_crossentropy', metrics=['accuracy'])

    # See an overview of the model architecture and to debug issues related to the model layers.
    model.summary()

###########################################################################################

    import time
    start_time = time.time() #To show the training time

    # Train the model

    # set an early stopping mechanism
    # set patience to be tolerant against random validation loss increases
    early_stopping = tf.keras.callbacks.EarlyStopping(patience=5)

    # history = model.fit(data, labels_one_hot, batch_size=32, epochs=10, validation_split=0.2)
    history = model.fit(x=data,
                        y=labels_one_hot,
                        batch_size=batch_size,
                        epochs=epochs,
                        validation_split=0.2,)
    #                     callbacks=[early_stopping])

    # Evaluate the model
    print("Test accuracy: ", max(history.history['val_accuracy']))

    # Assign the trained model
    self_train_model = history

    end_time = time.time() # To show the training time 
    training_time = end_time - start_time
    print("Training time:", training_time, "seconds")

    self_train_model_time = training_time
    
    return self_train_model, self_train_model_time
```

<a id='resnet50'></a>
### Modified ResNet50


```python
from tensorflow.keras.applications.resnet50 import ResNet50
from tensorflow.keras.layers import Dense, Flatten
from tensorflow.keras.models import Model
import tensorflow as tf

import time
start_time = time.time() #To show the training time

X=data
y=labels_one_hot

# set seed value for randomization
tf.random.set_seed(42)

# Load pre-trained ResNet50 model
resnet = ResNet50(include_top=False, input_shape=(input_size, input_size, 3))

# Freeze layers in ResNet50 model
for layer in resnet.layers:
    layer.trainable = False

# Add new classification layers
x = Flatten()(resnet.output)
x = Dense(128, activation='relu')(x)
x = Dense(3, activation='softmax')(x)

# Create new model
model = Model(inputs=resnet.input, outputs=x)

# Compile the model
model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

# Train the model
history = model.fit(X, y, epochs=100, batch_size=256, validation_split=0.2)

# Evaluate the model
print("Test accuracy: ", max(history.history['val_accuracy']))

# Assign the trained model
pre_train_model = history

end_time = time.time() # To show the training time 
training_time = end_time - start_time
print("Training time:", training_time, "seconds")

pre_train_model_time = training_time
```

    Downloading data from https://storage.googleapis.com/tensorflow/keras-applications/resnet/resnet50_weights_tf_dim_ordering_tf_kernels_notop.h5
    94773248/94765736 [==============================] - 4s 0us/step
    94781440/94765736 [==============================] - 4s 0us/step
    Epoch 1/100
    12/12 [==============================] - 12s 206ms/step - loss: 1.1322 - accuracy: 0.3700 - val_loss: 1.0495 - val_accuracy: 0.4416
    Epoch 2/100
    12/12 [==============================] - 1s 63ms/step - loss: 1.0297 - accuracy: 0.4708 - val_loss: 1.0110 - val_accuracy: 0.4966
    Epoch 3/100
    12/12 [==============================] - 1s 63ms/step - loss: 0.9948 - accuracy: 0.5044 - val_loss: 0.9800 - val_accuracy: 0.4993
    Epoch 4/100
    12/12 [==============================] - 1s 62ms/step - loss: 0.9750 - accuracy: 0.5302 - val_loss: 0.9688 - val_accuracy: 0.5705
    Epoch 5/100
    12/12 [==============================] - 1s 62ms/step - loss: 0.9532 - accuracy: 0.5554 - val_loss: 0.9873 - val_accuracy: 0.5114
    Epoch 6/100
    12/12 [==============================] - 1s 62ms/step - loss: 0.9394 - accuracy: 0.5527 - val_loss: 0.9350 - val_accuracy: 0.5369
    Epoch 7/100
    12/12 [==============================] - 1s 65ms/step - loss: 0.9162 - accuracy: 0.5860 - val_loss: 0.9243 - val_accuracy: 0.5691
    Epoch 8/100
    12/12 [==============================] - 1s 65ms/step - loss: 0.9020 - accuracy: 0.5934 - val_loss: 0.9303 - val_accuracy: 0.5638
    Epoch 9/100
    12/12 [==============================] - 1s 63ms/step - loss: 0.8797 - accuracy: 0.6169 - val_loss: 0.9035 - val_accuracy: 0.5799
    Epoch 10/100
    12/12 [==============================] - 1s 63ms/step - loss: 0.8732 - accuracy: 0.6095 - val_loss: 0.9273 - val_accuracy: 0.5705
    Epoch 11/100
    12/12 [==============================] - 1s 62ms/step - loss: 0.8641 - accuracy: 0.6232 - val_loss: 0.8879 - val_accuracy: 0.5879
    Epoch 12/100
    12/12 [==============================] - 1s 62ms/step - loss: 0.8504 - accuracy: 0.6212 - val_loss: 0.9474 - val_accuracy: 0.5329
    Epoch 13/100
    12/12 [==============================] - 1s 65ms/step - loss: 0.8667 - accuracy: 0.6058 - val_loss: 0.8943 - val_accuracy: 0.5732
    Epoch 14/100
    12/12 [==============================] - 1s 63ms/step - loss: 0.8400 - accuracy: 0.6256 - val_loss: 0.8705 - val_accuracy: 0.6067
    Epoch 15/100
    12/12 [==============================] - 1s 63ms/step - loss: 0.8221 - accuracy: 0.6370 - val_loss: 0.8624 - val_accuracy: 0.6121
    Epoch 16/100
    12/12 [==============================] - 1s 73ms/step - loss: 0.8170 - accuracy: 0.6400 - val_loss: 0.8667 - val_accuracy: 0.6174
    Epoch 17/100
    12/12 [==============================] - 1s 68ms/step - loss: 0.8041 - accuracy: 0.6461 - val_loss: 0.8871 - val_accuracy: 0.6013
    Epoch 18/100
    12/12 [==============================] - 1s 63ms/step - loss: 0.8097 - accuracy: 0.6380 - val_loss: 0.8531 - val_accuracy: 0.6054
    Epoch 19/100
    12/12 [==============================] - 1s 63ms/step - loss: 0.8103 - accuracy: 0.6373 - val_loss: 0.9006 - val_accuracy: 0.5919
    Epoch 20/100
    12/12 [==============================] - 1s 61ms/step - loss: 0.7941 - accuracy: 0.6481 - val_loss: 0.8463 - val_accuracy: 0.6148
    Epoch 21/100
    12/12 [==============================] - 1s 62ms/step - loss: 0.7764 - accuracy: 0.6655 - val_loss: 0.8605 - val_accuracy: 0.6255
    Epoch 22/100
    12/12 [==============================] - 1s 62ms/step - loss: 0.7715 - accuracy: 0.6692 - val_loss: 0.8611 - val_accuracy: 0.6054
    Epoch 23/100
    12/12 [==============================] - 1s 62ms/step - loss: 0.7760 - accuracy: 0.6561 - val_loss: 0.8508 - val_accuracy: 0.6242
    Epoch 24/100
    12/12 [==============================] - 1s 62ms/step - loss: 0.7784 - accuracy: 0.6565 - val_loss: 0.8510 - val_accuracy: 0.6215
    Epoch 25/100
    12/12 [==============================] - 1s 62ms/step - loss: 0.7639 - accuracy: 0.6739 - val_loss: 0.8610 - val_accuracy: 0.6161
    Epoch 26/100
    12/12 [==============================] - 1s 62ms/step - loss: 0.7631 - accuracy: 0.6662 - val_loss: 0.8474 - val_accuracy: 0.6107
    Epoch 27/100
    12/12 [==============================] - 1s 74ms/step - loss: 0.7534 - accuracy: 0.6696 - val_loss: 0.8546 - val_accuracy: 0.6322
    Epoch 28/100
    12/12 [==============================] - 1s 63ms/step - loss: 0.7589 - accuracy: 0.6659 - val_loss: 0.8524 - val_accuracy: 0.6081
    Epoch 29/100
    12/12 [==============================] - 1s 62ms/step - loss: 0.7518 - accuracy: 0.6692 - val_loss: 0.8348 - val_accuracy: 0.6322
    Epoch 30/100
    12/12 [==============================] - 1s 62ms/step - loss: 0.7603 - accuracy: 0.6612 - val_loss: 0.8747 - val_accuracy: 0.6148
    Epoch 31/100
    12/12 [==============================] - 1s 62ms/step - loss: 0.7567 - accuracy: 0.6655 - val_loss: 0.8729 - val_accuracy: 0.6121
    Epoch 32/100
    12/12 [==============================] - 1s 63ms/step - loss: 0.7440 - accuracy: 0.6793 - val_loss: 0.8358 - val_accuracy: 0.6255
    Epoch 33/100
    12/12 [==============================] - 1s 62ms/step - loss: 0.7393 - accuracy: 0.6830 - val_loss: 0.8242 - val_accuracy: 0.6349
    Epoch 34/100
    12/12 [==============================] - 1s 62ms/step - loss: 0.7342 - accuracy: 0.6810 - val_loss: 0.8631 - val_accuracy: 0.6201
    Epoch 35/100
    12/12 [==============================] - 1s 63ms/step - loss: 0.7329 - accuracy: 0.6854 - val_loss: 0.8309 - val_accuracy: 0.6282
    Epoch 36/100
    12/12 [==============================] - 1s 62ms/step - loss: 0.7285 - accuracy: 0.6860 - val_loss: 0.8440 - val_accuracy: 0.6309
    Epoch 37/100
    12/12 [==============================] - 1s 61ms/step - loss: 0.7134 - accuracy: 0.6934 - val_loss: 0.8296 - val_accuracy: 0.6295
    Epoch 38/100
    12/12 [==============================] - 1s 63ms/step - loss: 0.7166 - accuracy: 0.6938 - val_loss: 0.8181 - val_accuracy: 0.6349
    Epoch 39/100
    12/12 [==============================] - 1s 63ms/step - loss: 0.7226 - accuracy: 0.6880 - val_loss: 0.8449 - val_accuracy: 0.6322
    Epoch 40/100
    12/12 [==============================] - 1s 63ms/step - loss: 0.7360 - accuracy: 0.6699 - val_loss: 0.8527 - val_accuracy: 0.6161
    Epoch 41/100
    12/12 [==============================] - 1s 62ms/step - loss: 0.7239 - accuracy: 0.6870 - val_loss: 0.8142 - val_accuracy: 0.6389
    Epoch 42/100
    12/12 [==============================] - 1s 68ms/step - loss: 0.7153 - accuracy: 0.6847 - val_loss: 0.8252 - val_accuracy: 0.6376
    Epoch 43/100
    12/12 [==============================] - 1s 62ms/step - loss: 0.7156 - accuracy: 0.6823 - val_loss: 0.8415 - val_accuracy: 0.6295
    Epoch 44/100
    12/12 [==============================] - 1s 63ms/step - loss: 0.7074 - accuracy: 0.6938 - val_loss: 0.8137 - val_accuracy: 0.6362
    Epoch 45/100
    12/12 [==============================] - 1s 77ms/step - loss: 0.7088 - accuracy: 0.6887 - val_loss: 0.8228 - val_accuracy: 0.6309
    Epoch 46/100
    12/12 [==============================] - 1s 62ms/step - loss: 0.7004 - accuracy: 0.7048 - val_loss: 0.8157 - val_accuracy: 0.6523
    Epoch 47/100
    12/12 [==============================] - 1s 62ms/step - loss: 0.6909 - accuracy: 0.7052 - val_loss: 0.8140 - val_accuracy: 0.6483
    Epoch 48/100
    12/12 [==============================] - 1s 64ms/step - loss: 0.6913 - accuracy: 0.6988 - val_loss: 0.8381 - val_accuracy: 0.6295
    Epoch 49/100
    12/12 [==============================] - 1s 63ms/step - loss: 0.7035 - accuracy: 0.7048 - val_loss: 0.8341 - val_accuracy: 0.6430
    Epoch 50/100
    12/12 [==============================] - 1s 77ms/step - loss: 0.6865 - accuracy: 0.7079 - val_loss: 0.8313 - val_accuracy: 0.6349
    Epoch 51/100
    12/12 [==============================] - 1s 62ms/step - loss: 0.7000 - accuracy: 0.6998 - val_loss: 0.8224 - val_accuracy: 0.6336
    Epoch 52/100
    12/12 [==============================] - 1s 63ms/step - loss: 0.6858 - accuracy: 0.7099 - val_loss: 0.8055 - val_accuracy: 0.6470
    Epoch 53/100
    12/12 [==============================] - 1s 62ms/step - loss: 0.6827 - accuracy: 0.7021 - val_loss: 0.8476 - val_accuracy: 0.6309
    Epoch 54/100
    12/12 [==============================] - 1s 63ms/step - loss: 0.6816 - accuracy: 0.7095 - val_loss: 0.8009 - val_accuracy: 0.6483
    Epoch 55/100
    12/12 [==============================] - 1s 64ms/step - loss: 0.6789 - accuracy: 0.7105 - val_loss: 0.8149 - val_accuracy: 0.6430
    Epoch 56/100
    12/12 [==============================] - 1s 66ms/step - loss: 0.7155 - accuracy: 0.6780 - val_loss: 0.8519 - val_accuracy: 0.6242
    Epoch 57/100
    12/12 [==============================] - 1s 69ms/step - loss: 0.6962 - accuracy: 0.6914 - val_loss: 0.8011 - val_accuracy: 0.6510
    Epoch 58/100
    12/12 [==============================] - 1s 71ms/step - loss: 0.6776 - accuracy: 0.7015 - val_loss: 0.8457 - val_accuracy: 0.6282
    Epoch 59/100
    12/12 [==============================] - 1s 65ms/step - loss: 0.6847 - accuracy: 0.7072 - val_loss: 0.8182 - val_accuracy: 0.6510
    Epoch 60/100
    12/12 [==============================] - 1s 64ms/step - loss: 0.6639 - accuracy: 0.7250 - val_loss: 0.8469 - val_accuracy: 0.6268
    Epoch 61/100
    12/12 [==============================] - 1s 62ms/step - loss: 0.6684 - accuracy: 0.7129 - val_loss: 0.7971 - val_accuracy: 0.6577
    Epoch 62/100
    12/12 [==============================] - 1s 63ms/step - loss: 0.6589 - accuracy: 0.7139 - val_loss: 0.8044 - val_accuracy: 0.6523
    Epoch 63/100
    12/12 [==============================] - 1s 62ms/step - loss: 0.6658 - accuracy: 0.7176 - val_loss: 0.8088 - val_accuracy: 0.6497
    Epoch 64/100
    12/12 [==============================] - 1s 62ms/step - loss: 0.6738 - accuracy: 0.7112 - val_loss: 0.8211 - val_accuracy: 0.6430
    Epoch 65/100
    12/12 [==============================] - 1s 62ms/step - loss: 0.6611 - accuracy: 0.7203 - val_loss: 0.8017 - val_accuracy: 0.6470
    Epoch 66/100
    12/12 [==============================] - 1s 62ms/step - loss: 0.6570 - accuracy: 0.7203 - val_loss: 0.8287 - val_accuracy: 0.6510
    Epoch 67/100
    12/12 [==============================] - 1s 62ms/step - loss: 0.6574 - accuracy: 0.7136 - val_loss: 0.7985 - val_accuracy: 0.6658
    Epoch 68/100
    12/12 [==============================] - 1s 62ms/step - loss: 0.6521 - accuracy: 0.7260 - val_loss: 0.7996 - val_accuracy: 0.6456
    Epoch 69/100
    12/12 [==============================] - 1s 62ms/step - loss: 0.6524 - accuracy: 0.7246 - val_loss: 0.8195 - val_accuracy: 0.6523
    Epoch 70/100
    12/12 [==============================] - 1s 63ms/step - loss: 0.6484 - accuracy: 0.7310 - val_loss: 0.8150 - val_accuracy: 0.6564
    Epoch 71/100
    12/12 [==============================] - 1s 68ms/step - loss: 0.6500 - accuracy: 0.7293 - val_loss: 0.7939 - val_accuracy: 0.6644
    Epoch 72/100
    12/12 [==============================] - 1s 62ms/step - loss: 0.6526 - accuracy: 0.7199 - val_loss: 0.7950 - val_accuracy: 0.6671
    Epoch 73/100
    12/12 [==============================] - 1s 63ms/step - loss: 0.6670 - accuracy: 0.7142 - val_loss: 0.8215 - val_accuracy: 0.6510
    Epoch 74/100
    12/12 [==============================] - 1s 62ms/step - loss: 0.6687 - accuracy: 0.7069 - val_loss: 0.8307 - val_accuracy: 0.6389
    Epoch 75/100
    12/12 [==============================] - 1s 64ms/step - loss: 0.6507 - accuracy: 0.7344 - val_loss: 0.8138 - val_accuracy: 0.6430
    Epoch 76/100
    12/12 [==============================] - 1s 62ms/step - loss: 0.6467 - accuracy: 0.7257 - val_loss: 0.7928 - val_accuracy: 0.6711
    Epoch 77/100
    12/12 [==============================] - 1s 62ms/step - loss: 0.6462 - accuracy: 0.7277 - val_loss: 0.8393 - val_accuracy: 0.6523
    Epoch 78/100
    12/12 [==============================] - 1s 62ms/step - loss: 0.6626 - accuracy: 0.7179 - val_loss: 0.8378 - val_accuracy: 0.6456
    Epoch 79/100
    12/12 [==============================] - 1s 62ms/step - loss: 0.6479 - accuracy: 0.7203 - val_loss: 0.7889 - val_accuracy: 0.6564
    Epoch 80/100
    12/12 [==============================] - 1s 63ms/step - loss: 0.6468 - accuracy: 0.7243 - val_loss: 0.7879 - val_accuracy: 0.6685
    Epoch 81/100
    12/12 [==============================] - 1s 62ms/step - loss: 0.6558 - accuracy: 0.7213 - val_loss: 0.8010 - val_accuracy: 0.6523
    Epoch 82/100
    12/12 [==============================] - 1s 62ms/step - loss: 0.6387 - accuracy: 0.7280 - val_loss: 0.8210 - val_accuracy: 0.6631
    Epoch 83/100
    12/12 [==============================] - 1s 62ms/step - loss: 0.6642 - accuracy: 0.7213 - val_loss: 0.7914 - val_accuracy: 0.6550
    Epoch 84/100
    12/12 [==============================] - 1s 62ms/step - loss: 0.6320 - accuracy: 0.7391 - val_loss: 0.7799 - val_accuracy: 0.6752
    Epoch 85/100
    12/12 [==============================] - 1s 66ms/step - loss: 0.6344 - accuracy: 0.7277 - val_loss: 0.7901 - val_accuracy: 0.6752
    Epoch 86/100
    12/12 [==============================] - 1s 66ms/step - loss: 0.6307 - accuracy: 0.7337 - val_loss: 0.8268 - val_accuracy: 0.6349
    Epoch 87/100
    12/12 [==============================] - 1s 62ms/step - loss: 0.6391 - accuracy: 0.7330 - val_loss: 0.8242 - val_accuracy: 0.6362
    Epoch 88/100
    12/12 [==============================] - 1s 62ms/step - loss: 0.6472 - accuracy: 0.7226 - val_loss: 0.8408 - val_accuracy: 0.6443
    Epoch 89/100
    12/12 [==============================] - 1s 62ms/step - loss: 0.6481 - accuracy: 0.7213 - val_loss: 0.8564 - val_accuracy: 0.6483
    Epoch 90/100
    12/12 [==============================] - 1s 62ms/step - loss: 0.6592 - accuracy: 0.7095 - val_loss: 0.8203 - val_accuracy: 0.6389
    Epoch 91/100
    12/12 [==============================] - 1s 62ms/step - loss: 0.6268 - accuracy: 0.7374 - val_loss: 0.7844 - val_accuracy: 0.6765
    Epoch 92/100
    12/12 [==============================] - 1s 62ms/step - loss: 0.6212 - accuracy: 0.7414 - val_loss: 0.7976 - val_accuracy: 0.6617
    Epoch 93/100
    12/12 [==============================] - 1s 62ms/step - loss: 0.6188 - accuracy: 0.7411 - val_loss: 0.8121 - val_accuracy: 0.6510
    Epoch 94/100
    12/12 [==============================] - 1s 62ms/step - loss: 0.6250 - accuracy: 0.7340 - val_loss: 0.7873 - val_accuracy: 0.6779
    Epoch 95/100
    12/12 [==============================] - 1s 62ms/step - loss: 0.6077 - accuracy: 0.7435 - val_loss: 0.8039 - val_accuracy: 0.6617
    Epoch 96/100
    12/12 [==============================] - 1s 62ms/step - loss: 0.6157 - accuracy: 0.7394 - val_loss: 0.7949 - val_accuracy: 0.6725
    Epoch 97/100
    12/12 [==============================] - 1s 63ms/step - loss: 0.6170 - accuracy: 0.7404 - val_loss: 0.7788 - val_accuracy: 0.6819
    Epoch 98/100
    12/12 [==============================] - 1s 62ms/step - loss: 0.6243 - accuracy: 0.7364 - val_loss: 0.7771 - val_accuracy: 0.6725
    Epoch 99/100
    12/12 [==============================] - 1s 62ms/step - loss: 0.6059 - accuracy: 0.7408 - val_loss: 0.8495 - val_accuracy: 0.6591
    Epoch 100/100
    12/12 [==============================] - 1s 69ms/step - loss: 0.6294 - accuracy: 0.7307 - val_loss: 0.7784 - val_accuracy: 0.6725
    Test accuracy:  0.6818792223930359
    Training time: 94.45049977302551 seconds
    

##### Plot evalution results


```python
def plot_model_loss_and_acc(model, name):
    import matplotlib.pyplot as plt
    
    # Assign model to variable 'history'
    history = model
    
    # Set Figure size
    plt.figure(figsize=(10,5))
    
    # Plot the training and validation loss
    plt.subplot(1,2,1)
    plt.plot(history.history['loss'], label='training loss')
    plt.plot(history.history['val_loss'], label='validation loss')
    plt.xlabel('epochs')
    plt.ylabel('loss')
    plt.legend()
    plt.ylim(0,1.1)



    # Plot the training and validation accuracy
    plt.subplot(1,2,2)
    plt.plot(history.history['accuracy'], label='training accuracy')
    plt.plot(history.history['val_accuracy'], label='validation accuracy')
    plt.xlabel('epochs')
    plt.ylabel('accuracy')
    plt.legend()
    plt.ylim(0,1.1)
    
    plt.suptitle(name)
    plt.show()
```


```python

# Run CNN model
self_train_model, self_train_model_time = run_custom_model(256,100)
```

    Model: "sequential"
    _________________________________________________________________
    Layer (type)                 Output Shape              Param #   
    =================================================================
    conv2d (Conv2D)              (None, 62, 62, 32)        896       
    _________________________________________________________________
    conv2d_1 (Conv2D)            (None, 60, 60, 32)        9248      
    _________________________________________________________________
    max_pooling2d (MaxPooling2D) (None, 30, 30, 32)        0         
    _________________________________________________________________
    dropout (Dropout)            (None, 30, 30, 32)        0         
    _________________________________________________________________
    conv2d_2 (Conv2D)            (None, 28, 28, 64)        18496     
    _________________________________________________________________
    conv2d_3 (Conv2D)            (None, 26, 26, 64)        36928     
    _________________________________________________________________
    max_pooling2d_1 (MaxPooling2 (None, 13, 13, 64)        0         
    _________________________________________________________________
    dropout_1 (Dropout)          (None, 13, 13, 64)        0         
    _________________________________________________________________
    conv2d_4 (Conv2D)            (None, 11, 11, 256)       147712    
    _________________________________________________________________
    conv2d_5 (Conv2D)            (None, 9, 9, 256)         590080    
    _________________________________________________________________
    max_pooling2d_2 (MaxPooling2 (None, 4, 4, 256)         0         
    _________________________________________________________________
    dropout_2 (Dropout)          (None, 4, 4, 256)         0         
    _________________________________________________________________
    flatten_1 (Flatten)          (None, 4096)              0         
    _________________________________________________________________
    dense_2 (Dense)              (None, 1024)              4195328   
    _________________________________________________________________
    dropout_3 (Dropout)          (None, 1024)              0         
    _________________________________________________________________
    dense_3 (Dense)              (None, 3)                 3075      
    =================================================================
    Total params: 5,001,763
    Trainable params: 5,001,763
    Non-trainable params: 0
    _________________________________________________________________
    Epoch 1/100
    12/12 [==============================] - 3s 135ms/step - loss: 1.0971 - accuracy: 0.3586 - val_loss: 1.0773 - val_accuracy: 0.4872
    Epoch 2/100
    12/12 [==============================] - 1s 47ms/step - loss: 0.9933 - accuracy: 0.4896 - val_loss: 0.9651 - val_accuracy: 0.4805
    Epoch 3/100
    12/12 [==============================] - 1s 47ms/step - loss: 1.0390 - accuracy: 0.4594 - val_loss: 1.0436 - val_accuracy: 0.4242
    Epoch 4/100
    12/12 [==============================] - 1s 49ms/step - loss: 0.9641 - accuracy: 0.5064 - val_loss: 0.9457 - val_accuracy: 0.5007
    Epoch 5/100
    12/12 [==============================] - 1s 48ms/step - loss: 0.9228 - accuracy: 0.5171 - val_loss: 0.8830 - val_accuracy: 0.5329
    Epoch 6/100
    12/12 [==============================] - 1s 49ms/step - loss: 0.8540 - accuracy: 0.5772 - val_loss: 0.8808 - val_accuracy: 0.5597
    Epoch 7/100
    12/12 [==============================] - 1s 47ms/step - loss: 0.7893 - accuracy: 0.6138 - val_loss: 0.8326 - val_accuracy: 0.5758
    Epoch 8/100
    12/12 [==============================] - 1s 48ms/step - loss: 0.7353 - accuracy: 0.6494 - val_loss: 0.7234 - val_accuracy: 0.6537
    Epoch 9/100
    12/12 [==============================] - 1s 49ms/step - loss: 0.7560 - accuracy: 0.6283 - val_loss: 0.7853 - val_accuracy: 0.6255
    Epoch 10/100
    12/12 [==============================] - 1s 47ms/step - loss: 0.6923 - accuracy: 0.6964 - val_loss: 0.7515 - val_accuracy: 0.6698
    Epoch 11/100
    12/12 [==============================] - 1s 47ms/step - loss: 0.6570 - accuracy: 0.7025 - val_loss: 0.7263 - val_accuracy: 0.6805
    Epoch 12/100
    12/12 [==============================] - 1s 47ms/step - loss: 0.6293 - accuracy: 0.7310 - val_loss: 0.7183 - val_accuracy: 0.6658
    Epoch 13/100
    12/12 [==============================] - 1s 50ms/step - loss: 0.6040 - accuracy: 0.7340 - val_loss: 0.6253 - val_accuracy: 0.7289
    Epoch 14/100
    12/12 [==============================] - 1s 49ms/step - loss: 0.5490 - accuracy: 0.7693 - val_loss: 0.5538 - val_accuracy: 0.7517
    Epoch 15/100
    12/12 [==============================] - 1s 48ms/step - loss: 0.5783 - accuracy: 0.7582 - val_loss: 0.6121 - val_accuracy: 0.7423
    Epoch 16/100
    12/12 [==============================] - 1s 47ms/step - loss: 0.5100 - accuracy: 0.7874 - val_loss: 0.5832 - val_accuracy: 0.7584
    Epoch 17/100
    12/12 [==============================] - 1s 47ms/step - loss: 0.4893 - accuracy: 0.7945 - val_loss: 0.5152 - val_accuracy: 0.7852
    Epoch 18/100
    12/12 [==============================] - 1s 47ms/step - loss: 0.4542 - accuracy: 0.8109 - val_loss: 0.5612 - val_accuracy: 0.7570
    Epoch 19/100
    12/12 [==============================] - 1s 47ms/step - loss: 0.4361 - accuracy: 0.8227 - val_loss: 0.5954 - val_accuracy: 0.7597
    Epoch 20/100
    12/12 [==============================] - 1s 47ms/step - loss: 0.4402 - accuracy: 0.8244 - val_loss: 0.5716 - val_accuracy: 0.7329
    Epoch 21/100
    12/12 [==============================] - 1s 49ms/step - loss: 0.4368 - accuracy: 0.8287 - val_loss: 0.5355 - val_accuracy: 0.7745
    Epoch 22/100
    12/12 [==============================] - 1s 47ms/step - loss: 0.3864 - accuracy: 0.8475 - val_loss: 0.4822 - val_accuracy: 0.8000
    Epoch 23/100
    12/12 [==============================] - 1s 47ms/step - loss: 0.3610 - accuracy: 0.8606 - val_loss: 0.5049 - val_accuracy: 0.8094
    Epoch 24/100
    12/12 [==============================] - 1s 47ms/step - loss: 0.3838 - accuracy: 0.8496 - val_loss: 0.5323 - val_accuracy: 0.8027
    Epoch 25/100
    12/12 [==============================] - 1s 47ms/step - loss: 0.3476 - accuracy: 0.8734 - val_loss: 0.4125 - val_accuracy: 0.8349
    Epoch 26/100
    12/12 [==============================] - 1s 47ms/step - loss: 0.3235 - accuracy: 0.8741 - val_loss: 0.4978 - val_accuracy: 0.7933
    Epoch 27/100
    12/12 [==============================] - 1s 47ms/step - loss: 0.3000 - accuracy: 0.8848 - val_loss: 0.4099 - val_accuracy: 0.8322
    Epoch 28/100
    12/12 [==============================] - 1s 47ms/step - loss: 0.2921 - accuracy: 0.8939 - val_loss: 0.3785 - val_accuracy: 0.8470
    Epoch 29/100
    12/12 [==============================] - 1s 47ms/step - loss: 0.2510 - accuracy: 0.9056 - val_loss: 0.3760 - val_accuracy: 0.8497
    Epoch 30/100
    12/12 [==============================] - 1s 47ms/step - loss: 0.2160 - accuracy: 0.9147 - val_loss: 0.3396 - val_accuracy: 0.8765
    Epoch 31/100
    12/12 [==============================] - 1s 47ms/step - loss: 0.1953 - accuracy: 0.9285 - val_loss: 0.4152 - val_accuracy: 0.8456
    Epoch 32/100
    12/12 [==============================] - 1s 51ms/step - loss: 0.1777 - accuracy: 0.9349 - val_loss: 0.3432 - val_accuracy: 0.8805
    Epoch 33/100
    12/12 [==============================] - 1s 49ms/step - loss: 0.2176 - accuracy: 0.9234 - val_loss: 0.4392 - val_accuracy: 0.8456
    Epoch 34/100
    12/12 [==============================] - 1s 47ms/step - loss: 0.2207 - accuracy: 0.9161 - val_loss: 0.5288 - val_accuracy: 0.8107
    Epoch 35/100
    12/12 [==============================] - 1s 48ms/step - loss: 0.1832 - accuracy: 0.9285 - val_loss: 0.3893 - val_accuracy: 0.8604
    Epoch 36/100
    12/12 [==============================] - 1s 47ms/step - loss: 0.1426 - accuracy: 0.9459 - val_loss: 0.3306 - val_accuracy: 0.8792
    Epoch 37/100
    12/12 [==============================] - 1s 47ms/step - loss: 0.1485 - accuracy: 0.9433 - val_loss: 0.3467 - val_accuracy: 0.8779
    Epoch 38/100
    12/12 [==============================] - 1s 48ms/step - loss: 0.1256 - accuracy: 0.9574 - val_loss: 0.3354 - val_accuracy: 0.9074
    Epoch 39/100
    12/12 [==============================] - 1s 47ms/step - loss: 0.1273 - accuracy: 0.9543 - val_loss: 0.3353 - val_accuracy: 0.8886
    Epoch 40/100
    12/12 [==============================] - 1s 48ms/step - loss: 0.1178 - accuracy: 0.9520 - val_loss: 0.3333 - val_accuracy: 0.9060
    Epoch 41/100
    12/12 [==============================] - 1s 47ms/step - loss: 0.1001 - accuracy: 0.9641 - val_loss: 0.4500 - val_accuracy: 0.8685
    Epoch 42/100
    12/12 [==============================] - 1s 49ms/step - loss: 0.0843 - accuracy: 0.9715 - val_loss: 0.4896 - val_accuracy: 0.8953
    Epoch 43/100
    12/12 [==============================] - 1s 48ms/step - loss: 0.0765 - accuracy: 0.9718 - val_loss: 0.4875 - val_accuracy: 0.8644
    Epoch 44/100
    12/12 [==============================] - 1s 47ms/step - loss: 0.0716 - accuracy: 0.9728 - val_loss: 0.4979 - val_accuracy: 0.8711
    Epoch 45/100
    12/12 [==============================] - 1s 48ms/step - loss: 0.0915 - accuracy: 0.9664 - val_loss: 0.5638 - val_accuracy: 0.8604
    Epoch 46/100
    12/12 [==============================] - 1s 47ms/step - loss: 0.1093 - accuracy: 0.9597 - val_loss: 0.3134 - val_accuracy: 0.9047
    Epoch 47/100
    12/12 [==============================] - 1s 47ms/step - loss: 0.0930 - accuracy: 0.9654 - val_loss: 0.3148 - val_accuracy: 0.9007
    Epoch 48/100
    12/12 [==============================] - 1s 47ms/step - loss: 0.0618 - accuracy: 0.9795 - val_loss: 0.3163 - val_accuracy: 0.9034
    Epoch 49/100
    12/12 [==============================] - 1s 47ms/step - loss: 0.0809 - accuracy: 0.9715 - val_loss: 0.4656 - val_accuracy: 0.8899
    Epoch 50/100
    12/12 [==============================] - 1s 54ms/step - loss: 0.1130 - accuracy: 0.9567 - val_loss: 0.3270 - val_accuracy: 0.8926
    Epoch 51/100
    12/12 [==============================] - 1s 54ms/step - loss: 0.0838 - accuracy: 0.9698 - val_loss: 0.5072 - val_accuracy: 0.8698
    Epoch 52/100
    12/12 [==============================] - 1s 52ms/step - loss: 0.0931 - accuracy: 0.9661 - val_loss: 0.3379 - val_accuracy: 0.8940
    Epoch 53/100
    12/12 [==============================] - 1s 47ms/step - loss: 0.0536 - accuracy: 0.9839 - val_loss: 0.5055 - val_accuracy: 0.8872
    Epoch 54/100
    12/12 [==============================] - 1s 47ms/step - loss: 0.0515 - accuracy: 0.9782 - val_loss: 0.3463 - val_accuracy: 0.9074
    Epoch 55/100
    12/12 [==============================] - 1s 48ms/step - loss: 0.0526 - accuracy: 0.9819 - val_loss: 0.3229 - val_accuracy: 0.9154
    Epoch 56/100
    12/12 [==============================] - 1s 48ms/step - loss: 0.0477 - accuracy: 0.9815 - val_loss: 0.2908 - val_accuracy: 0.9302
    Epoch 57/100
    12/12 [==============================] - 1s 48ms/step - loss: 0.0359 - accuracy: 0.9859 - val_loss: 0.3652 - val_accuracy: 0.9195
    Epoch 58/100
    12/12 [==============================] - 1s 48ms/step - loss: 0.0275 - accuracy: 0.9903 - val_loss: 0.3356 - val_accuracy: 0.9221
    Epoch 59/100
    12/12 [==============================] - 1s 48ms/step - loss: 0.0260 - accuracy: 0.9913 - val_loss: 0.2917 - val_accuracy: 0.9315
    Epoch 60/100
    12/12 [==============================] - 1s 47ms/step - loss: 0.0817 - accuracy: 0.9778 - val_loss: 1.0122 - val_accuracy: 0.8268
    Epoch 61/100
    12/12 [==============================] - 1s 47ms/step - loss: 0.4269 - accuracy: 0.8677 - val_loss: 0.5378 - val_accuracy: 0.8121
    Epoch 62/100
    12/12 [==============================] - 1s 48ms/step - loss: 0.2406 - accuracy: 0.9117 - val_loss: 0.4155 - val_accuracy: 0.8564
    Epoch 63/100
    12/12 [==============================] - 1s 47ms/step - loss: 0.1368 - accuracy: 0.9503 - val_loss: 0.4765 - val_accuracy: 0.8550
    Epoch 64/100
    12/12 [==============================] - 1s 47ms/step - loss: 0.0985 - accuracy: 0.9668 - val_loss: 0.3741 - val_accuracy: 0.8940
    Epoch 65/100
    12/12 [==============================] - 1s 47ms/step - loss: 0.0862 - accuracy: 0.9694 - val_loss: 0.5717 - val_accuracy: 0.8725
    Epoch 66/100
    12/12 [==============================] - 1s 49ms/step - loss: 0.0953 - accuracy: 0.9661 - val_loss: 0.3853 - val_accuracy: 0.8966
    Epoch 67/100
    12/12 [==============================] - 1s 48ms/step - loss: 0.0598 - accuracy: 0.9805 - val_loss: 0.3415 - val_accuracy: 0.9141
    Epoch 68/100
    12/12 [==============================] - 1s 47ms/step - loss: 0.0409 - accuracy: 0.9859 - val_loss: 0.3762 - val_accuracy: 0.9101
    Epoch 69/100
    12/12 [==============================] - 1s 47ms/step - loss: 0.0264 - accuracy: 0.9913 - val_loss: 0.3507 - val_accuracy: 0.9047
    Epoch 70/100
    12/12 [==============================] - 1s 47ms/step - loss: 0.0269 - accuracy: 0.9906 - val_loss: 0.4454 - val_accuracy: 0.9020
    Epoch 71/100
    12/12 [==============================] - 1s 52ms/step - loss: 0.0406 - accuracy: 0.9866 - val_loss: 0.4394 - val_accuracy: 0.9060
    Epoch 72/100
    12/12 [==============================] - 1s 48ms/step - loss: 0.0616 - accuracy: 0.9825 - val_loss: 0.4643 - val_accuracy: 0.9007
    Epoch 73/100
    12/12 [==============================] - 1s 47ms/step - loss: 0.0433 - accuracy: 0.9846 - val_loss: 0.4257 - val_accuracy: 0.9034
    Epoch 74/100
    12/12 [==============================] - 1s 48ms/step - loss: 0.0446 - accuracy: 0.9835 - val_loss: 0.4937 - val_accuracy: 0.8886
    Epoch 75/100
    12/12 [==============================] - 1s 47ms/step - loss: 0.0539 - accuracy: 0.9802 - val_loss: 0.6038 - val_accuracy: 0.8631
    Epoch 76/100
    12/12 [==============================] - 1s 48ms/step - loss: 0.0498 - accuracy: 0.9825 - val_loss: 0.4526 - val_accuracy: 0.8953
    Epoch 77/100
    12/12 [==============================] - 1s 47ms/step - loss: 0.0413 - accuracy: 0.9849 - val_loss: 0.3887 - val_accuracy: 0.8993
    Epoch 78/100
    12/12 [==============================] - 1s 47ms/step - loss: 0.0312 - accuracy: 0.9886 - val_loss: 0.4016 - val_accuracy: 0.9195
    Epoch 79/100
    12/12 [==============================] - 1s 48ms/step - loss: 0.0260 - accuracy: 0.9913 - val_loss: 0.3844 - val_accuracy: 0.9195
    Epoch 80/100
    12/12 [==============================] - 1s 48ms/step - loss: 0.0194 - accuracy: 0.9936 - val_loss: 0.4759 - val_accuracy: 0.9087
    Epoch 81/100
    12/12 [==============================] - 1s 47ms/step - loss: 0.0279 - accuracy: 0.9899 - val_loss: 0.5561 - val_accuracy: 0.9074
    Epoch 82/100
    12/12 [==============================] - 1s 48ms/step - loss: 0.0156 - accuracy: 0.9956 - val_loss: 0.3895 - val_accuracy: 0.9275
    Epoch 83/100
    12/12 [==============================] - 1s 48ms/step - loss: 0.0129 - accuracy: 0.9956 - val_loss: 0.4353 - val_accuracy: 0.9168
    Epoch 84/100
    12/12 [==============================] - 1s 47ms/step - loss: 0.0101 - accuracy: 0.9976 - val_loss: 0.5251 - val_accuracy: 0.9128
    Epoch 85/100
    12/12 [==============================] - 1s 47ms/step - loss: 0.0098 - accuracy: 0.9976 - val_loss: 0.4291 - val_accuracy: 0.9262
    Epoch 86/100
    12/12 [==============================] - 1s 47ms/step - loss: 0.0107 - accuracy: 0.9956 - val_loss: 0.4588 - val_accuracy: 0.9235
    Epoch 87/100
    12/12 [==============================] - 1s 47ms/step - loss: 0.0124 - accuracy: 0.9956 - val_loss: 0.4550 - val_accuracy: 0.9195
    Epoch 88/100
    12/12 [==============================] - 1s 47ms/step - loss: 0.0104 - accuracy: 0.9970 - val_loss: 0.4789 - val_accuracy: 0.9154
    Epoch 89/100
    12/12 [==============================] - 1s 48ms/step - loss: 0.0115 - accuracy: 0.9970 - val_loss: 0.5356 - val_accuracy: 0.9114
    Epoch 90/100
    12/12 [==============================] - 1s 55ms/step - loss: 0.0310 - accuracy: 0.9893 - val_loss: 0.5309 - val_accuracy: 0.9007
    Epoch 91/100
    12/12 [==============================] - 1s 48ms/step - loss: 0.0475 - accuracy: 0.9839 - val_loss: 0.5498 - val_accuracy: 0.8792
    Epoch 92/100
    12/12 [==============================] - 1s 48ms/step - loss: 0.0322 - accuracy: 0.9862 - val_loss: 0.4737 - val_accuracy: 0.9154
    Epoch 93/100
    12/12 [==============================] - 1s 48ms/step - loss: 0.0181 - accuracy: 0.9933 - val_loss: 0.4878 - val_accuracy: 0.9101
    Epoch 94/100
    12/12 [==============================] - 1s 48ms/step - loss: 0.0152 - accuracy: 0.9956 - val_loss: 0.5834 - val_accuracy: 0.9060
    Epoch 95/100
    12/12 [==============================] - 1s 49ms/step - loss: 0.0186 - accuracy: 0.9940 - val_loss: 0.5569 - val_accuracy: 0.9114
    Epoch 96/100
    12/12 [==============================] - 1s 47ms/step - loss: 0.0246 - accuracy: 0.9923 - val_loss: 0.3993 - val_accuracy: 0.9329
    Epoch 97/100
    12/12 [==============================] - 1s 47ms/step - loss: 0.0147 - accuracy: 0.9953 - val_loss: 0.4169 - val_accuracy: 0.9195
    Epoch 98/100
    12/12 [==============================] - 1s 47ms/step - loss: 0.0143 - accuracy: 0.9933 - val_loss: 0.4381 - val_accuracy: 0.9275
    Epoch 99/100
    12/12 [==============================] - 1s 47ms/step - loss: 0.0116 - accuracy: 0.9953 - val_loss: 0.5013 - val_accuracy: 0.9087
    Epoch 100/100
    12/12 [==============================] - 1s 47ms/step - loss: 0.0151 - accuracy: 0.9940 - val_loss: 0.3964 - val_accuracy: 0.9248
    Test accuracy:  0.9328858852386475
    Training time: 60.310054063797 seconds
    


```python
plot_model_loss_and_acc(self_train_model, 'Self Train CNNs')
plot_model_loss_and_acc(pre_train_model, 'With Pre-trained Model(Resnet50)')
```


    
![png](water-bottle-images-classification-cnn-resnet50_files/water-bottle-images-classification-cnn-resnet50_35_0.png)
    



    
![png](water-bottle-images-classification-cnn-resnet50_files/water-bottle-images-classification-cnn-resnet50_35_1.png)
    


#### Plot confusion matrix


```python
'''
Convert np.ndarray(n,3) into List of predicted labels
'''
def output_converter(model_output):

    import numpy as np

    output = model_output

    # assume that 'output' is a numpy array of shape (n, 3)
    output_labels = ['Full  Water level', 'Half water level', 'Overflowing']
    predictions = np.argmax(output, axis=1)
    predicted_labels = [output_labels[p] for p in predictions]

    return predicted_labels
```


```python
'''
Plot a Heatmap-Crosstab table out of predicted labels and True labels
'''
def plot_hm_ct(y_true, y_pred): 
    import pandas as pd
    import seaborn as sns
    import matplotlib.pyplot as plt

    # create a DataFrame from y_true and y_pred
    df = pd.DataFrame({'y_true': y_true, 'y_pred': y_pred})

    # create cross-tabulation matrix
    ctab = pd.crosstab(df['y_true'], df['y_pred'])

    # create heatmap using seaborn
    sns.heatmap(ctab, annot=True, cmap='Blues', fmt='d')

    # add labels and title
    plt.xlabel('Predicted label')
    plt.ylabel('True label')
    plt.title('Confusion Matrix')

    # show the plot
    plt.show()
```


```python
'''
Generate confusion matrix from trained model
'''
def generate_cf(model, name):
    
    import pandas as pd
    import seaborn as sns
    import matplotlib.pyplot as plt
    
    # Assign model to variable 'history'
    history = model
    
    # Load output data
    y_pred = output_converter(history.model.predict(X_test))
    y_true = y_test

    # Plot the confusion matrix
    # create a DataFrame from y_true and y_pred
    df = pd.DataFrame({'y_true': y_true, 'y_pred': y_pred})

    # create cross-tabulation matrix
    ctab = pd.crosstab(df['y_true'], df['y_pred'])

    # create heatmap using seaborn
    sns.heatmap(ctab, annot=True, cmap='Blues', fmt='d')

    # add labels and title
    plt.xlabel('Predicted label')
    plt.ylabel('True label')
    plt.title('{} Confusion Matrix'.format(name))

    # show the plot
    plt.show()

    # Calculate accuracy score
    from sklearn.metrics import accuracy_score
    accuracy = accuracy_score(y_true, y_pred)
    print("{} accuracy score: {}".format(name, accuracy))
```


```python
generate_cf(self_train_model, 'Self Train CNNs')
print("")
print("")
print("")
generate_cf(pre_train_model, 'With Pre-trained Model (Resnet50)')
```


    
![png](water-bottle-images-classification-cnn-resnet50_files/water-bottle-images-classification-cnn-resnet50_40_0.png)
    


    Self Train CNNs accuracy score: 0.9527389903329753
    
    
    
    


    
![png](water-bottle-images-classification-cnn-resnet50_files/water-bottle-images-classification-cnn-resnet50_40_2.png)
    


    With Pre-trained Model (Resnet50) accuracy score: 0.7056928034371643
    


```python
'''
Preformance Comparision
'''
import pandas as pd
import matplotlib.pyplot as plt

df = pd.DataFrame({'Model': ['CNN', 'Modified ResNet50'],
                   'Accuracy': [max(self_train_model.history['accuracy']), max(pre_train_model.history['accuracy'])],
                   'Time(s)': [self_train_model_time, pre_train_model_time]})

# create a figure and axis object
fig, ax = plt.subplots()

# set the bar width
bar_width = 0.35

# create a bar plot for the first column on the primary y-axis
bar1 = ax.bar(df.index, df['Accuracy'], color='b', width=bar_width, label='Accuracy')
ax.set_ylabel('Accuracy')

# create a bar plot for the second column on the secondary y-axis
ax2 = ax.twinx()
bar2 = ax2.bar(df.index + bar_width, df['Time(s)'], color='r', width=bar_width, label='Time(s)')
ax2.set_ylabel('Time(s)')

# set the title and x-axis label
ax.set_title('Bar Chart with Two Columns')
ax.set_xlabel('Index')

# set the x-axis ticks and labels
ax.set_xticks(df.index + bar_width / 2)
ax.set_xticklabels(df['Model'])

# add the legend
handles, labels = [], []
for ax in [ax, ax2]:
    for h, l in zip(*ax.get_legend_handles_labels()):
        handles.append(h)
        labels.append(l)
ax.legend(handles, labels, loc='best')

# display the plot
plt.show()

```


    
![png](water-bottle-images-classification-cnn-resnet50_files/water-bottle-images-classification-cnn-resnet50_41_0.png)
    


---
<a id='hyperparameter_tuning'></a>
## 5. Hyperparameter Tuning

<a id='gridsearchcv'></a>
**GridSearchCV** is a technique used in machine learning to tune hyperparameters for a model. It allows us to define a grid of hyperparameters to test, and then it will search over all possible combinations of these hyperparameters to find the best combination for our model.


```python
from sklearn.model_selection import GridSearchCV
from keras.wrappers.scikit_learn import KerasClassifier
from keras.models import Sequential
from keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout
import tensorflow as tf
from keras.callbacks import EarlyStopping


import warnings
warnings.filterwarnings('ignore') # Hide all warnings

import time
start_time = time.time() #To show the training time

tf.random.set_seed(42)
batch_size = [128 ,256]
epochs = [50,100]
optimizer = ['adam']
# optimizer = ['adam', 'rmsprop']
# cv = 5 # None mean default (K-fold=5)
cv = [(slice(None), slice(None))]


# Design Model Layers
def create_model(optimizer):
    model = Sequential()
    model.add(Conv2D(32, (3,3), activation='relu', input_shape=(input_size, input_size, 3)))
    model.add(Conv2D(32, (3, 3),activation='relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Dropout(0.2))

    model.add(Conv2D(64, (3, 3), activation='relu',padding='same'))
    model.add(Conv2D(64, (3, 3), activation='relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Dropout(0.2))

    model.add(Conv2D(256, (3, 3), activation='relu',padding='same'))
    model.add(Conv2D(256, (3, 3),activation='relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Dropout(0.2))

    model.add(Flatten())
    model.add(Dense(1024, activation='relu'))
    model.add(Dropout(0.5))
    model.add(Dense(3, activation='softmax'))
    model.compile(loss='categorical_crossentropy', optimizer=optimizer, metrics=['accuracy'])
    return model


model = KerasClassifier(build_fn=create_model)

param_grid = {'batch_size': batch_size,
              'epochs': epochs,
              'optimizer': optimizer,}
#               'callbacks': [early_stopping]} # Disable callbachs function since we want model run with equal epochs for comparing


grid = GridSearchCV(estimator=model, param_grid=param_grid, cv=cv) 
grid_result = grid.fit(data, labels_one_hot, verbose=0)

print("Best: %f using %s" % (grid_result.best_score_, grid_result.best_params_))
means = grid_result.cv_results_['mean_test_score']
stds = grid_result.cv_results_['std_test_score']
params = grid_result.cv_results_['params']
for mean, stdev, param in zip(means, stds, params):
    print("%f (%f) with: %r" % (mean, stdev, param))
    
end_time = time.time() # To show the training time 
training_time = end_time - start_time
print("Training time:", training_time, "seconds")
grid_time = training_time
```

    30/30 [==============================] - 0s 9ms/step - loss: 0.0176 - accuracy: 0.9946
    30/30 [==============================] - 0s 9ms/step - loss: 0.0026 - accuracy: 0.9995
    15/15 [==============================] - 0s 17ms/step - loss: 0.0314 - accuracy: 0.9893
    15/15 [==============================] - 0s 16ms/step - loss: 0.0029 - accuracy: 0.9995
    Best: 0.999463 using {'batch_size': 128, 'epochs': 100, 'optimizer': 'adam'}
    0.994628 (0.000000) with: {'batch_size': 128, 'epochs': 50, 'optimizer': 'adam'}
    0.999463 (0.000000) with: {'batch_size': 128, 'epochs': 100, 'optimizer': 'adam'}
    0.989256 (0.000000) with: {'batch_size': 256, 'epochs': 50, 'optimizer': 'adam'}
    0.999463 (0.000000) with: {'batch_size': 256, 'epochs': 100, 'optimizer': 'adam'}
    Training time: 289.4944784641266 seconds
    


```python
'''
Overview detailed information about the grid search cross-validation process
'''
import pandas as pd
print(pd.DataFrame(grid_result.cv_results_))
```

       mean_fit_time  std_fit_time  mean_score_time  std_score_time  \
    0      36.606178           0.0         0.805221             0.0   
    1      70.527983           0.0         0.814474             0.0   
    2      42.000096           0.0         0.803917             0.0   
    3      66.191189           0.0         0.820317             0.0   
    
      param_batch_size param_epochs param_optimizer  \
    0              128           50            adam   
    1              128          100            adam   
    2              256           50            adam   
    3              256          100            adam   
    
                                                  params  split0_test_score  \
    0  {'batch_size': 128, 'epochs': 50, 'optimizer':...           0.994628   
    1  {'batch_size': 128, 'epochs': 100, 'optimizer'...           0.999463   
    2  {'batch_size': 256, 'epochs': 50, 'optimizer':...           0.989256   
    3  {'batch_size': 256, 'epochs': 100, 'optimizer'...           0.999463   
    
       mean_test_score  std_test_score  rank_test_score  
    0         0.994628             0.0                3  
    1         0.999463             0.0                1  
    2         0.989256             0.0                4  
    3         0.999463             0.0                1  
    


```python
'''
Transform predicted result into list
'''
output_labels = ['Full  Water level', 'Half water level', 'Overflowing']
result = grid.predict(X_test)

predicted_labels = list(map(lambda x: output_labels[x], result))
# print(predicted_labels[:5])
```


```python
'''
Plot and confusion metrix
'''
import seaborn as sns

# Load output data
y_pred = predicted_labels
y_true = y_test

# Plot the confusion matrix
# create a DataFrame from y_true and y_pred
df = pd.DataFrame({'y_true': y_true, 'y_pred': y_pred})

# create cross-tabulation matrix
ctab = pd.crosstab(df['y_true'], df['y_pred'])

# create heatmap using seaborn
sns.heatmap(ctab, annot=True, cmap='Blues', fmt='d')

# add labels and title
plt.xlabel('Predicted label')
plt.ylabel('True label')
plt.title('GridSerachCV result Confusion Matrix')

# show the plot
plt.show()

# Calculate accuracy score
from sklearn.metrics import accuracy_score
accuracy = accuracy_score(y_true, y_pred)
print("GridSerachCV accuracy score:{}".format(accuracy))
```


    
![png](water-bottle-images-classification-cnn-resnet50_files/water-bottle-images-classification-cnn-resnet50_47_0.png)
    


    GridSerachCV accuracy score:0.9344790547798066
    


```python
'''
Show a prediction of images from the test set
'''

import matplotlib.pyplot as plt

# Load the data
X_test = X_test

# choose 20 random indices
indices = np.random.randint(0, len(X_test), 20)

# Get 20 sample images
sample_images = X_test[indices]

# Plot the images
fig = plt.figure(figsize=(10,10))
for i, img in enumerate(sample_images):
    plt.subplot(4, 5, i+1)
    plt.imshow(img)
    plt.axis('off')
    plt.title( y_true[indices[i]] + "\n" + "Predicted result: " + "\n"+ y_pred[indices[i]])
    
plt.show()
```


    
![png](water-bottle-images-classification-cnn-resnet50_files/water-bottle-images-classification-cnn-resnet50_48_0.png)
    


<a id='summary'></a>
# Summary
* CNN model yields higher accuracy while takes slightly same amount of time. 
* The training loss of CNN model decreased to nearly 0, while the ResNet50 model has more resistance of lowering. Due to the differences in the architectures of the two models. This can be assume that CNN model is overfitting the training data more easier the pre-trained model.
* GridSearchCV is possible to perform but consuming enormous time and risk of exceeding the memory. Thus, a better approach is to start with a pre-trained model and fine-tune the model to adapt it to our dataset.
* Overall, our CNN model was able to classify the water level of a given water bottle image with a high degree of accuracy (<85%).

<a id='note'></a>
# Note
#### Improving a machine learning model can be achieved by using various techniques such as:
* ✅ The techniques that I've applied
* ❌ The technoques that I think it not neccesary.
* 🤔 I don't know how to apply with codek and still figure out.

1. **Feature Engineering**❌: Adding or modifying the features used in the model to better capture the underlying patterns in the data.

2. **Model Selection**🤔: Choosing a different machine learning model that is more suitable for the data and the problem being solved.

3. **Hyperparameter Tuning**✅: Adjusting the parameters of the machine learning model to improve its performance. This can be done manually or using techniques such as grid search or random search.

4. **Ensemble Methods**🤔: Combining multiple models to create a more robust model. This can be done by averaging the predictions of multiple models or by training a separate model to make predictions based on the outputs of other models.

5. **Regularization**✅: Adding a penalty term to the loss function to prevent overfitting and improve generalization.

6. **Data Augmentation**✅: Increasing the size of the dataset by generating new data samples based on the original data. This can help to prevent overfitting and improve generalization.
    - After predicting unseen test set model return an ugly result.
    - To solve problem I will try clone an equally proportion labels dataset.

7. **Early Stopping**✅: Stopping the training process when the model's performance on the validation set starts to deteriorate. This can prevent overfitting and help to avoid the use of models that are too complex.

8. **Transfer Learning**🤔: Reusing pre-trained models to reduce the time and computational resources required to train a new model.
9. **Data Resampling**✅: Randomly adding or removing data from the dataset to balance the classes. 

# [Back to top](#back_to_top)


```python

```

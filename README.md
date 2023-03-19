<a id='back_to_top'></a>
# Bottle Water Images Classification-CNN & ResNet50
"Classifying Water Bottle Images Based on Water Level Using Machine Learning"

**Created By**: Wuttipat S. <br>
**Created Date**: 2023-02-10 <br>
**Status**: <span style="color:green">Completed</span>

#### Update: 
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
input_size= 64
image_size = (input_size, input_size)

# Access the directory and sub-directories and so on
directory = "water-bottle-dataset"
# directory = "/kaggle/input/water-bottle-dataset"

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
# import tensorflow as tf
# print("Num GPUs Available: ", len(tf.config.list_physical_devices('GPU')))
```


```python
# with tf.device('/GPU:0'):


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


import time
start_time = time.time() #To show the training time

# Train the model

# set an early stopping mechanism
# set patience to be tolerant against random validation loss increases
early_stopping = tf.keras.callbacks.EarlyStopping(patience=5)

# history = model.fit(data, labels_one_hot, batch_size=32, epochs=10, validation_split=0.2)
history = model.fit(x=data,
                    y=labels_one_hot,
                    batch_size=256,
                    epochs=100,
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
```

    Model: "sequential"
    _________________________________________________________________
     Layer (type)                Output Shape              Param #   
    =================================================================
     conv2d (Conv2D)             (None, 62, 62, 32)        896       
                                                                     
     conv2d_1 (Conv2D)           (None, 60, 60, 32)        9248      
                                                                     
     max_pooling2d (MaxPooling2D  (None, 30, 30, 32)       0         
     )                                                               
                                                                     
     dropout (Dropout)           (None, 30, 30, 32)        0         
                                                                     
     conv2d_2 (Conv2D)           (None, 28, 28, 64)        18496     
                                                                     
     conv2d_3 (Conv2D)           (None, 26, 26, 64)        36928     
                                                                     
     max_pooling2d_1 (MaxPooling  (None, 13, 13, 64)       0         
     2D)                                                             
                                                                     
     dropout_1 (Dropout)         (None, 13, 13, 64)        0         
                                                                     
     conv2d_4 (Conv2D)           (None, 11, 11, 256)       147712    
                                                                     
     conv2d_5 (Conv2D)           (None, 9, 9, 256)         590080    
                                                                     
     max_pooling2d_2 (MaxPooling  (None, 4, 4, 256)        0         
     2D)                                                             
                                                                     
     dropout_2 (Dropout)         (None, 4, 4, 256)         0         
                                                                     
     flatten (Flatten)           (None, 4096)              0         
                                                                     
     dense (Dense)               (None, 1024)              4195328   
                                                                     
     dropout_3 (Dropout)         (None, 1024)              0         
                                                                     
     dense_1 (Dense)             (None, 3)                 3075      
                                                                     
    =================================================================
    Total params: 5,001,763
    Trainable params: 5,001,763
    Non-trainable params: 0
    _________________________________________________________________
    Epoch 1/100
    12/12 [==============================] - 10s 386ms/step - loss: 1.0854 - accuracy: 0.3697 - val_loss: 1.0768 - val_accuracy: 0.3718
    Epoch 2/100
    12/12 [==============================] - 2s 154ms/step - loss: 1.0157 - accuracy: 0.4627 - val_loss: 0.9717 - val_accuracy: 0.5074
    Epoch 3/100
    12/12 [==============================] - 2s 154ms/step - loss: 0.9234 - accuracy: 0.5289 - val_loss: 0.9057 - val_accuracy: 0.5530
    Epoch 4/100
    12/12 [==============================] - 2s 154ms/step - loss: 0.8520 - accuracy: 0.5682 - val_loss: 0.8133 - val_accuracy: 0.5799
    Epoch 5/100
    12/12 [==============================] - 2s 154ms/step - loss: 0.7929 - accuracy: 0.5970 - val_loss: 0.8585 - val_accuracy: 0.5732
    Epoch 6/100
    12/12 [==============================] - 2s 154ms/step - loss: 0.7439 - accuracy: 0.6427 - val_loss: 0.7479 - val_accuracy: 0.6295
    Epoch 7/100
    12/12 [==============================] - 2s 154ms/step - loss: 0.6914 - accuracy: 0.6961 - val_loss: 0.7470 - val_accuracy: 0.6376
    Epoch 8/100
    12/12 [==============================] - 2s 154ms/step - loss: 0.6593 - accuracy: 0.7035 - val_loss: 0.8395 - val_accuracy: 0.6067
    Epoch 9/100
    12/12 [==============================] - 2s 154ms/step - loss: 0.6932 - accuracy: 0.6978 - val_loss: 0.7302 - val_accuracy: 0.6685
    Epoch 10/100
    12/12 [==============================] - 2s 153ms/step - loss: 0.6246 - accuracy: 0.7354 - val_loss: 0.8526 - val_accuracy: 0.6134
    Epoch 11/100
    12/12 [==============================] - 2s 154ms/step - loss: 0.6045 - accuracy: 0.7344 - val_loss: 0.6579 - val_accuracy: 0.7047
    Epoch 12/100
    12/12 [==============================] - 2s 154ms/step - loss: 0.5161 - accuracy: 0.7921 - val_loss: 0.6849 - val_accuracy: 0.7168
    Epoch 13/100
    12/12 [==============================] - 2s 154ms/step - loss: 0.5109 - accuracy: 0.7931 - val_loss: 0.6319 - val_accuracy: 0.7477
    Epoch 14/100
    12/12 [==============================] - 2s 153ms/step - loss: 0.4569 - accuracy: 0.8143 - val_loss: 0.5747 - val_accuracy: 0.7745
    Epoch 15/100
    12/12 [==============================] - 2s 154ms/step - loss: 0.4255 - accuracy: 0.8345 - val_loss: 0.4991 - val_accuracy: 0.7973
    Epoch 16/100
    12/12 [==============================] - 2s 153ms/step - loss: 0.4354 - accuracy: 0.8146 - val_loss: 0.5495 - val_accuracy: 0.7597
    Epoch 17/100
    12/12 [==============================] - 2s 154ms/step - loss: 0.3852 - accuracy: 0.8526 - val_loss: 0.5405 - val_accuracy: 0.7678
    Epoch 18/100
    12/12 [==============================] - 2s 155ms/step - loss: 0.3858 - accuracy: 0.8479 - val_loss: 0.4992 - val_accuracy: 0.7960
    Epoch 19/100
    12/12 [==============================] - 2s 154ms/step - loss: 0.3419 - accuracy: 0.8694 - val_loss: 0.5731 - val_accuracy: 0.7852
    Epoch 20/100
    12/12 [==============================] - 2s 154ms/step - loss: 0.3569 - accuracy: 0.8647 - val_loss: 0.4998 - val_accuracy: 0.7973
    Epoch 21/100
    12/12 [==============================] - 2s 154ms/step - loss: 0.3175 - accuracy: 0.8751 - val_loss: 0.5312 - val_accuracy: 0.8013
    Epoch 22/100
    12/12 [==============================] - 2s 154ms/step - loss: 0.2966 - accuracy: 0.8858 - val_loss: 0.4707 - val_accuracy: 0.8134
    Epoch 23/100
    12/12 [==============================] - 2s 154ms/step - loss: 0.3120 - accuracy: 0.8811 - val_loss: 0.3677 - val_accuracy: 0.8631
    Epoch 24/100
    12/12 [==============================] - 2s 154ms/step - loss: 0.2435 - accuracy: 0.9097 - val_loss: 0.3664 - val_accuracy: 0.8604
    Epoch 25/100
    12/12 [==============================] - 2s 154ms/step - loss: 0.2041 - accuracy: 0.9275 - val_loss: 0.4797 - val_accuracy: 0.8309
    Epoch 26/100
    12/12 [==============================] - 2s 154ms/step - loss: 0.1849 - accuracy: 0.9308 - val_loss: 0.3029 - val_accuracy: 0.8872
    Epoch 27/100
    12/12 [==============================] - 2s 154ms/step - loss: 0.1891 - accuracy: 0.9268 - val_loss: 0.4197 - val_accuracy: 0.8550
    Epoch 28/100
    12/12 [==============================] - 2s 154ms/step - loss: 0.1535 - accuracy: 0.9476 - val_loss: 0.3267 - val_accuracy: 0.8832
    Epoch 29/100
    12/12 [==============================] - 2s 154ms/step - loss: 0.1235 - accuracy: 0.9557 - val_loss: 0.3098 - val_accuracy: 0.8886
    Epoch 30/100
    12/12 [==============================] - 2s 155ms/step - loss: 0.1263 - accuracy: 0.9520 - val_loss: 0.3222 - val_accuracy: 0.8886
    Epoch 31/100
    12/12 [==============================] - 2s 155ms/step - loss: 0.1164 - accuracy: 0.9584 - val_loss: 0.4111 - val_accuracy: 0.8738
    Epoch 32/100
    12/12 [==============================] - 2s 154ms/step - loss: 0.1203 - accuracy: 0.9557 - val_loss: 0.4365 - val_accuracy: 0.8631
    Epoch 33/100
    12/12 [==============================] - 2s 154ms/step - loss: 0.1409 - accuracy: 0.9469 - val_loss: 0.4134 - val_accuracy: 0.8805
    Epoch 34/100
    12/12 [==============================] - 2s 155ms/step - loss: 0.1360 - accuracy: 0.9500 - val_loss: 0.4742 - val_accuracy: 0.8497
    Epoch 35/100
    12/12 [==============================] - 2s 155ms/step - loss: 0.1204 - accuracy: 0.9624 - val_loss: 0.3545 - val_accuracy: 0.8859
    Epoch 36/100
    12/12 [==============================] - 2s 155ms/step - loss: 0.0963 - accuracy: 0.9644 - val_loss: 0.3700 - val_accuracy: 0.8899
    Epoch 37/100
    12/12 [==============================] - 2s 154ms/step - loss: 0.0839 - accuracy: 0.9688 - val_loss: 0.3685 - val_accuracy: 0.8940
    Epoch 38/100
    12/12 [==============================] - 2s 156ms/step - loss: 0.0841 - accuracy: 0.9708 - val_loss: 0.3871 - val_accuracy: 0.8953
    Epoch 39/100
    12/12 [==============================] - 2s 154ms/step - loss: 0.1053 - accuracy: 0.9607 - val_loss: 0.3993 - val_accuracy: 0.8819
    Epoch 40/100
    12/12 [==============================] - 2s 154ms/step - loss: 0.1050 - accuracy: 0.9607 - val_loss: 0.3446 - val_accuracy: 0.9047
    Epoch 41/100
    12/12 [==============================] - 2s 154ms/step - loss: 0.0832 - accuracy: 0.9708 - val_loss: 0.2699 - val_accuracy: 0.9315
    Epoch 42/100
    12/12 [==============================] - 2s 154ms/step - loss: 0.0540 - accuracy: 0.9795 - val_loss: 0.3137 - val_accuracy: 0.9128
    Epoch 43/100
    12/12 [==============================] - 2s 154ms/step - loss: 0.0514 - accuracy: 0.9842 - val_loss: 0.3877 - val_accuracy: 0.9060
    Epoch 44/100
    12/12 [==============================] - 2s 154ms/step - loss: 0.0501 - accuracy: 0.9809 - val_loss: 0.3718 - val_accuracy: 0.9128
    Epoch 45/100
    12/12 [==============================] - 2s 154ms/step - loss: 0.0764 - accuracy: 0.9741 - val_loss: 0.4866 - val_accuracy: 0.8819
    Epoch 46/100
    12/12 [==============================] - 2s 155ms/step - loss: 0.0916 - accuracy: 0.9701 - val_loss: 0.2959 - val_accuracy: 0.9235
    Epoch 47/100
    12/12 [==============================] - 2s 155ms/step - loss: 0.0660 - accuracy: 0.9785 - val_loss: 0.3034 - val_accuracy: 0.9221
    Epoch 48/100
    12/12 [==============================] - 2s 155ms/step - loss: 0.0504 - accuracy: 0.9839 - val_loss: 0.4241 - val_accuracy: 0.8913
    Epoch 49/100
    12/12 [==============================] - 2s 155ms/step - loss: 0.0413 - accuracy: 0.9849 - val_loss: 0.3278 - val_accuracy: 0.9195
    Epoch 50/100
    12/12 [==============================] - 2s 155ms/step - loss: 0.0456 - accuracy: 0.9842 - val_loss: 0.3222 - val_accuracy: 0.9128
    Epoch 51/100
    12/12 [==============================] - 2s 155ms/step - loss: 0.0506 - accuracy: 0.9815 - val_loss: 0.3277 - val_accuracy: 0.9248
    Epoch 52/100
    12/12 [==============================] - 2s 155ms/step - loss: 0.0436 - accuracy: 0.9835 - val_loss: 0.3649 - val_accuracy: 0.9034
    Epoch 53/100
    12/12 [==============================] - 2s 155ms/step - loss: 0.0607 - accuracy: 0.9775 - val_loss: 0.3235 - val_accuracy: 0.9154
    Epoch 54/100
    12/12 [==============================] - 2s 155ms/step - loss: 0.0438 - accuracy: 0.9862 - val_loss: 0.3182 - val_accuracy: 0.9181
    Epoch 55/100
    12/12 [==============================] - 2s 156ms/step - loss: 0.0401 - accuracy: 0.9872 - val_loss: 0.3151 - val_accuracy: 0.9181
    Epoch 56/100
    12/12 [==============================] - 2s 155ms/step - loss: 0.0312 - accuracy: 0.9889 - val_loss: 0.3654 - val_accuracy: 0.9221
    Epoch 57/100
    12/12 [==============================] - 2s 155ms/step - loss: 0.0215 - accuracy: 0.9916 - val_loss: 0.2842 - val_accuracy: 0.9329
    Epoch 58/100
    12/12 [==============================] - 2s 155ms/step - loss: 0.0208 - accuracy: 0.9936 - val_loss: 0.3318 - val_accuracy: 0.9275
    Epoch 59/100
    12/12 [==============================] - 2s 155ms/step - loss: 0.0222 - accuracy: 0.9919 - val_loss: 0.3533 - val_accuracy: 0.9248
    Epoch 60/100
    12/12 [==============================] - 2s 155ms/step - loss: 0.0174 - accuracy: 0.9936 - val_loss: 0.4218 - val_accuracy: 0.9195
    Epoch 61/100
    12/12 [==============================] - 2s 155ms/step - loss: 0.0390 - accuracy: 0.9879 - val_loss: 0.3938 - val_accuracy: 0.9060
    Epoch 62/100
    12/12 [==============================] - 2s 154ms/step - loss: 0.0429 - accuracy: 0.9886 - val_loss: 0.4399 - val_accuracy: 0.9141
    Epoch 63/100
    12/12 [==============================] - 2s 154ms/step - loss: 0.0334 - accuracy: 0.9899 - val_loss: 0.3707 - val_accuracy: 0.9262
    Epoch 64/100
    12/12 [==============================] - 2s 155ms/step - loss: 0.0271 - accuracy: 0.9906 - val_loss: 0.4102 - val_accuracy: 0.9181
    Epoch 65/100
    12/12 [==============================] - 2s 155ms/step - loss: 0.0241 - accuracy: 0.9923 - val_loss: 0.3961 - val_accuracy: 0.9168
    Epoch 66/100
    12/12 [==============================] - 2s 155ms/step - loss: 0.0207 - accuracy: 0.9943 - val_loss: 0.3672 - val_accuracy: 0.9208
    Epoch 67/100
    12/12 [==============================] - 2s 155ms/step - loss: 0.0413 - accuracy: 0.9866 - val_loss: 0.6178 - val_accuracy: 0.8832
    Epoch 68/100
    12/12 [==============================] - 2s 155ms/step - loss: 0.0470 - accuracy: 0.9835 - val_loss: 0.4435 - val_accuracy: 0.8980
    Epoch 69/100
    12/12 [==============================] - 2s 155ms/step - loss: 0.0806 - accuracy: 0.9741 - val_loss: 0.3334 - val_accuracy: 0.9101
    Epoch 70/100
    12/12 [==============================] - 2s 155ms/step - loss: 0.0937 - accuracy: 0.9718 - val_loss: 0.3592 - val_accuracy: 0.8993
    Epoch 71/100
    12/12 [==============================] - 2s 155ms/step - loss: 0.0421 - accuracy: 0.9862 - val_loss: 0.3487 - val_accuracy: 0.9154
    Epoch 72/100
    12/12 [==============================] - 2s 155ms/step - loss: 0.0220 - accuracy: 0.9926 - val_loss: 0.3466 - val_accuracy: 0.9195
    Epoch 73/100
    12/12 [==============================] - 2s 155ms/step - loss: 0.0173 - accuracy: 0.9943 - val_loss: 0.3487 - val_accuracy: 0.9235
    Epoch 74/100
    12/12 [==============================] - 2s 155ms/step - loss: 0.0192 - accuracy: 0.9906 - val_loss: 0.4393 - val_accuracy: 0.9114
    Epoch 75/100
    12/12 [==============================] - 2s 155ms/step - loss: 0.0252 - accuracy: 0.9899 - val_loss: 0.4039 - val_accuracy: 0.9248
    Epoch 76/100
    12/12 [==============================] - 2s 155ms/step - loss: 0.0300 - accuracy: 0.9893 - val_loss: 0.4018 - val_accuracy: 0.9154
    Epoch 77/100
    12/12 [==============================] - 2s 155ms/step - loss: 0.0188 - accuracy: 0.9943 - val_loss: 0.4720 - val_accuracy: 0.9168
    Epoch 78/100
    12/12 [==============================] - 2s 155ms/step - loss: 0.0271 - accuracy: 0.9913 - val_loss: 0.3789 - val_accuracy: 0.9154
    Epoch 79/100
    12/12 [==============================] - 2s 155ms/step - loss: 0.0270 - accuracy: 0.9909 - val_loss: 0.5809 - val_accuracy: 0.8779
    Epoch 80/100
    12/12 [==============================] - 2s 154ms/step - loss: 0.0385 - accuracy: 0.9876 - val_loss: 0.4402 - val_accuracy: 0.9101
    Epoch 81/100
    12/12 [==============================] - 2s 155ms/step - loss: 0.0360 - accuracy: 0.9859 - val_loss: 0.3424 - val_accuracy: 0.9221
    Epoch 82/100
    12/12 [==============================] - 2s 155ms/step - loss: 0.0261 - accuracy: 0.9909 - val_loss: 0.3282 - val_accuracy: 0.9208
    Epoch 83/100
    12/12 [==============================] - 2s 155ms/step - loss: 0.0217 - accuracy: 0.9946 - val_loss: 0.4499 - val_accuracy: 0.9248
    Epoch 84/100
    12/12 [==============================] - 2s 155ms/step - loss: 0.0293 - accuracy: 0.9893 - val_loss: 0.4104 - val_accuracy: 0.9168
    Epoch 85/100
    12/12 [==============================] - 2s 155ms/step - loss: 0.0234 - accuracy: 0.9899 - val_loss: 0.4264 - val_accuracy: 0.9074
    Epoch 86/100
    12/12 [==============================] - 2s 155ms/step - loss: 0.0359 - accuracy: 0.9862 - val_loss: 0.5206 - val_accuracy: 0.9034
    Epoch 87/100
    12/12 [==============================] - 2s 155ms/step - loss: 0.0469 - accuracy: 0.9852 - val_loss: 0.3823 - val_accuracy: 0.9154
    Epoch 88/100
    12/12 [==============================] - 2s 155ms/step - loss: 0.0333 - accuracy: 0.9903 - val_loss: 0.4379 - val_accuracy: 0.9128
    Epoch 89/100
    12/12 [==============================] - 2s 155ms/step - loss: 0.0209 - accuracy: 0.9943 - val_loss: 0.3844 - val_accuracy: 0.9114
    Epoch 90/100
    12/12 [==============================] - 2s 155ms/step - loss: 0.0139 - accuracy: 0.9953 - val_loss: 0.3223 - val_accuracy: 0.9315
    Epoch 91/100
    12/12 [==============================] - 2s 155ms/step - loss: 0.0109 - accuracy: 0.9966 - val_loss: 0.4084 - val_accuracy: 0.9168
    Epoch 92/100
    12/12 [==============================] - 2s 156ms/step - loss: 0.0050 - accuracy: 0.9976 - val_loss: 0.3810 - val_accuracy: 0.9289
    Epoch 93/100
    12/12 [==============================] - 2s 155ms/step - loss: 0.0049 - accuracy: 0.9997 - val_loss: 0.3815 - val_accuracy: 0.9356
    Epoch 94/100
    12/12 [==============================] - 2s 155ms/step - loss: 0.0043 - accuracy: 0.9987 - val_loss: 0.3919 - val_accuracy: 0.9342
    Epoch 95/100
    12/12 [==============================] - 2s 155ms/step - loss: 0.0024 - accuracy: 0.9993 - val_loss: 0.4191 - val_accuracy: 0.9356
    Epoch 96/100
    12/12 [==============================] - 2s 155ms/step - loss: 0.0023 - accuracy: 0.9993 - val_loss: 0.4122 - val_accuracy: 0.9302
    Epoch 97/100
    12/12 [==============================] - 2s 155ms/step - loss: 0.0063 - accuracy: 0.9983 - val_loss: 0.4574 - val_accuracy: 0.9275
    Epoch 98/100
    12/12 [==============================] - 2s 155ms/step - loss: 0.0157 - accuracy: 0.9980 - val_loss: 0.3963 - val_accuracy: 0.9369
    Epoch 99/100
    12/12 [==============================] - 2s 157ms/step - loss: 0.0100 - accuracy: 0.9963 - val_loss: 0.4040 - val_accuracy: 0.9302
    Epoch 100/100
    12/12 [==============================] - 2s 155ms/step - loss: 0.0050 - accuracy: 0.9997 - val_loss: 0.4285 - val_accuracy: 0.9195
    Test accuracy:  0.9369127750396729
    Training time: 192.84626126289368 seconds
    

<a id='resnet50'></a>
### Modified ResNet50


```python
from tensorflow.keras.applications.resnet50 import ResNet50
from tensorflow.keras.layers import Dense, Flatten
from tensorflow.keras.models import Model

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

    Epoch 1/100
    12/12 [==============================] - 10s 442ms/step - loss: 1.1283 - accuracy: 0.3687 - val_loss: 1.0435 - val_accuracy: 0.4456
    Epoch 2/100
    12/12 [==============================] - 2s 153ms/step - loss: 1.0261 - accuracy: 0.4859 - val_loss: 0.9994 - val_accuracy: 0.5101
    Epoch 3/100
    12/12 [==============================] - 2s 152ms/step - loss: 0.9938 - accuracy: 0.5238 - val_loss: 0.9679 - val_accuracy: 0.5262
    Epoch 4/100
    12/12 [==============================] - 2s 152ms/step - loss: 0.9673 - accuracy: 0.5426 - val_loss: 0.9606 - val_accuracy: 0.5651
    Epoch 5/100
    12/12 [==============================] - 2s 152ms/step - loss: 0.9427 - accuracy: 0.5651 - val_loss: 0.9426 - val_accuracy: 0.5517
    Epoch 6/100
    12/12 [==============================] - 2s 153ms/step - loss: 0.9276 - accuracy: 0.5702 - val_loss: 0.9156 - val_accuracy: 0.5544
    Epoch 7/100
    12/12 [==============================] - 2s 153ms/step - loss: 0.9058 - accuracy: 0.5870 - val_loss: 0.8981 - val_accuracy: 0.5745
    Epoch 8/100
    12/12 [==============================] - 2s 153ms/step - loss: 0.8916 - accuracy: 0.5977 - val_loss: 0.8903 - val_accuracy: 0.6067
    Epoch 9/100
    12/12 [==============================] - 2s 152ms/step - loss: 0.8800 - accuracy: 0.6111 - val_loss: 0.8926 - val_accuracy: 0.5624
    Epoch 10/100
    12/12 [==============================] - 2s 153ms/step - loss: 0.8767 - accuracy: 0.5913 - val_loss: 0.8859 - val_accuracy: 0.5919
    Epoch 11/100
    12/12 [==============================] - 3s 276ms/step - loss: 0.8604 - accuracy: 0.6179 - val_loss: 0.8598 - val_accuracy: 0.6188
    Epoch 12/100
    12/12 [==============================] - 3s 275ms/step - loss: 0.8440 - accuracy: 0.6316 - val_loss: 0.8631 - val_accuracy: 0.6148
    Epoch 13/100
    12/12 [==============================] - 2s 152ms/step - loss: 0.8507 - accuracy: 0.6242 - val_loss: 0.8599 - val_accuracy: 0.6054
    Epoch 14/100
    12/12 [==============================] - 2s 151ms/step - loss: 0.8292 - accuracy: 0.6397 - val_loss: 0.8499 - val_accuracy: 0.6174
    Epoch 15/100
    12/12 [==============================] - 2s 153ms/step - loss: 0.8223 - accuracy: 0.6414 - val_loss: 0.8522 - val_accuracy: 0.6148
    Epoch 16/100
    12/12 [==============================] - 2s 152ms/step - loss: 0.8141 - accuracy: 0.6444 - val_loss: 0.8539 - val_accuracy: 0.6242
    Epoch 17/100
    12/12 [==============================] - 2s 151ms/step - loss: 0.8186 - accuracy: 0.6323 - val_loss: 0.8625 - val_accuracy: 0.5826
    Epoch 18/100
    12/12 [==============================] - 2s 153ms/step - loss: 0.8229 - accuracy: 0.6310 - val_loss: 0.8526 - val_accuracy: 0.6174
    Epoch 19/100
    12/12 [==============================] - 2s 151ms/step - loss: 0.7947 - accuracy: 0.6521 - val_loss: 0.8197 - val_accuracy: 0.6268
    Epoch 20/100
    12/12 [==============================] - 2s 152ms/step - loss: 0.7887 - accuracy: 0.6545 - val_loss: 0.8048 - val_accuracy: 0.6497
    Epoch 21/100
    12/12 [==============================] - 2s 152ms/step - loss: 0.7747 - accuracy: 0.6649 - val_loss: 0.8082 - val_accuracy: 0.6591
    Epoch 22/100
    12/12 [==============================] - 2s 152ms/step - loss: 0.7752 - accuracy: 0.6672 - val_loss: 0.8409 - val_accuracy: 0.5919
    Epoch 23/100
    12/12 [==============================] - 2s 152ms/step - loss: 0.7936 - accuracy: 0.6447 - val_loss: 0.8322 - val_accuracy: 0.6174
    Epoch 24/100
    12/12 [==============================] - 2s 152ms/step - loss: 0.7781 - accuracy: 0.6575 - val_loss: 0.8119 - val_accuracy: 0.6255
    Epoch 25/100
    12/12 [==============================] - 2s 152ms/step - loss: 0.7642 - accuracy: 0.6716 - val_loss: 0.8102 - val_accuracy: 0.6564
    Epoch 26/100
    12/12 [==============================] - 2s 152ms/step - loss: 0.7543 - accuracy: 0.6770 - val_loss: 0.7896 - val_accuracy: 0.6537
    Epoch 27/100
    12/12 [==============================] - 2s 152ms/step - loss: 0.7497 - accuracy: 0.6733 - val_loss: 0.8032 - val_accuracy: 0.6362
    Epoch 28/100
    12/12 [==============================] - 2s 151ms/step - loss: 0.7462 - accuracy: 0.6860 - val_loss: 0.7868 - val_accuracy: 0.6443
    Epoch 29/100
    12/12 [==============================] - 2s 153ms/step - loss: 0.7457 - accuracy: 0.6706 - val_loss: 0.7874 - val_accuracy: 0.6658
    Epoch 30/100
    12/12 [==============================] - 2s 151ms/step - loss: 0.7449 - accuracy: 0.6807 - val_loss: 0.8166 - val_accuracy: 0.6309
    Epoch 31/100
    12/12 [==============================] - 2s 151ms/step - loss: 0.7400 - accuracy: 0.6850 - val_loss: 0.8239 - val_accuracy: 0.6362
    Epoch 32/100
    12/12 [==============================] - 2s 152ms/step - loss: 0.7375 - accuracy: 0.6914 - val_loss: 0.7729 - val_accuracy: 0.6577
    Epoch 33/100
    12/12 [==============================] - 2s 152ms/step - loss: 0.7290 - accuracy: 0.6807 - val_loss: 0.7732 - val_accuracy: 0.6497
    Epoch 34/100
    12/12 [==============================] - 2s 152ms/step - loss: 0.7227 - accuracy: 0.6917 - val_loss: 0.7835 - val_accuracy: 0.6564
    Epoch 35/100
    12/12 [==============================] - 2s 152ms/step - loss: 0.7258 - accuracy: 0.6914 - val_loss: 0.7860 - val_accuracy: 0.6738
    Epoch 36/100
    12/12 [==============================] - 2s 151ms/step - loss: 0.7167 - accuracy: 0.6961 - val_loss: 0.7624 - val_accuracy: 0.6698
    Epoch 37/100
    12/12 [==============================] - 2s 152ms/step - loss: 0.7118 - accuracy: 0.7005 - val_loss: 0.7671 - val_accuracy: 0.6550
    Epoch 38/100
    12/12 [==============================] - 2s 153ms/step - loss: 0.7063 - accuracy: 0.6998 - val_loss: 0.7665 - val_accuracy: 0.6765
    Epoch 39/100
    12/12 [==============================] - 2s 154ms/step - loss: 0.7021 - accuracy: 0.7042 - val_loss: 0.7627 - val_accuracy: 0.6617
    Epoch 40/100
    12/12 [==============================] - 2s 152ms/step - loss: 0.7253 - accuracy: 0.6793 - val_loss: 0.7695 - val_accuracy: 0.6738
    Epoch 41/100
    12/12 [==============================] - 2s 152ms/step - loss: 0.7094 - accuracy: 0.6961 - val_loss: 0.7942 - val_accuracy: 0.6483
    Epoch 42/100
    12/12 [==============================] - 2s 152ms/step - loss: 0.7127 - accuracy: 0.6864 - val_loss: 0.8021 - val_accuracy: 0.6376
    Epoch 43/100
    12/12 [==============================] - 2s 152ms/step - loss: 0.7161 - accuracy: 0.6891 - val_loss: 0.7745 - val_accuracy: 0.6779
    Epoch 44/100
    12/12 [==============================] - 2s 152ms/step - loss: 0.7116 - accuracy: 0.6924 - val_loss: 0.7792 - val_accuracy: 0.6456
    Epoch 45/100
    12/12 [==============================] - 2s 153ms/step - loss: 0.7314 - accuracy: 0.6850 - val_loss: 0.7661 - val_accuracy: 0.6483
    Epoch 46/100
    12/12 [==============================] - 2s 152ms/step - loss: 0.7181 - accuracy: 0.6897 - val_loss: 0.8038 - val_accuracy: 0.6658
    Epoch 47/100
    12/12 [==============================] - 2s 153ms/step - loss: 0.7043 - accuracy: 0.6917 - val_loss: 0.7489 - val_accuracy: 0.6779
    Epoch 48/100
    12/12 [==============================] - 2s 152ms/step - loss: 0.6876 - accuracy: 0.7055 - val_loss: 0.7510 - val_accuracy: 0.6725
    Epoch 49/100
    12/12 [==============================] - 2s 152ms/step - loss: 0.6800 - accuracy: 0.7156 - val_loss: 0.7548 - val_accuracy: 0.6698
    Epoch 50/100
    12/12 [==============================] - 2s 152ms/step - loss: 0.6928 - accuracy: 0.7042 - val_loss: 0.7793 - val_accuracy: 0.6752
    Epoch 51/100
    12/12 [==============================] - 2s 153ms/step - loss: 0.6972 - accuracy: 0.7015 - val_loss: 0.7522 - val_accuracy: 0.6658
    Epoch 52/100
    12/12 [==============================] - 2s 153ms/step - loss: 0.6816 - accuracy: 0.7095 - val_loss: 0.7608 - val_accuracy: 0.6685
    Epoch 53/100
    12/12 [==============================] - 2s 152ms/step - loss: 0.6865 - accuracy: 0.7021 - val_loss: 0.7491 - val_accuracy: 0.6738
    Epoch 54/100
    12/12 [==============================] - 2s 152ms/step - loss: 0.6762 - accuracy: 0.7179 - val_loss: 0.7586 - val_accuracy: 0.6577
    Epoch 55/100
    12/12 [==============================] - 2s 153ms/step - loss: 0.6831 - accuracy: 0.7095 - val_loss: 0.7353 - val_accuracy: 0.6725
    Epoch 56/100
    12/12 [==============================] - 2s 152ms/step - loss: 0.6834 - accuracy: 0.7109 - val_loss: 0.8334 - val_accuracy: 0.6174
    Epoch 57/100
    12/12 [==============================] - 2s 152ms/step - loss: 0.6739 - accuracy: 0.7058 - val_loss: 0.7372 - val_accuracy: 0.6846
    Epoch 58/100
    12/12 [==============================] - 2s 154ms/step - loss: 0.6675 - accuracy: 0.7072 - val_loss: 0.7858 - val_accuracy: 0.6698
    Epoch 59/100
    12/12 [==============================] - 2s 155ms/step - loss: 0.6700 - accuracy: 0.7193 - val_loss: 0.7440 - val_accuracy: 0.6671
    Epoch 60/100
    12/12 [==============================] - 2s 154ms/step - loss: 0.6590 - accuracy: 0.7186 - val_loss: 0.7935 - val_accuracy: 0.6550
    Epoch 61/100
    12/12 [==============================] - 2s 155ms/step - loss: 0.6756 - accuracy: 0.7032 - val_loss: 0.7417 - val_accuracy: 0.6805
    Epoch 62/100
    12/12 [==============================] - 2s 155ms/step - loss: 0.6641 - accuracy: 0.7173 - val_loss: 0.7343 - val_accuracy: 0.6725
    Epoch 63/100
    12/12 [==============================] - 2s 153ms/step - loss: 0.6612 - accuracy: 0.7152 - val_loss: 0.7527 - val_accuracy: 0.6658
    Epoch 64/100
    12/12 [==============================] - 2s 212ms/step - loss: 0.6648 - accuracy: 0.7112 - val_loss: 0.7389 - val_accuracy: 0.6711
    Epoch 65/100
    12/12 [==============================] - 4s 309ms/step - loss: 0.6485 - accuracy: 0.7287 - val_loss: 0.7303 - val_accuracy: 0.6846
    Epoch 66/100
    12/12 [==============================] - 2s 155ms/step - loss: 0.6438 - accuracy: 0.7277 - val_loss: 0.7634 - val_accuracy: 0.6805
    Epoch 67/100
    12/12 [==============================] - 2s 154ms/step - loss: 0.6539 - accuracy: 0.7210 - val_loss: 0.7342 - val_accuracy: 0.6819
    Epoch 68/100
    12/12 [==============================] - 2s 153ms/step - loss: 0.6381 - accuracy: 0.7300 - val_loss: 0.7289 - val_accuracy: 0.6859
    Epoch 69/100
    12/12 [==============================] - 2s 153ms/step - loss: 0.6408 - accuracy: 0.7314 - val_loss: 0.7469 - val_accuracy: 0.6765
    Epoch 70/100
    12/12 [==============================] - 2s 153ms/step - loss: 0.6471 - accuracy: 0.7169 - val_loss: 0.7575 - val_accuracy: 0.6832
    Epoch 71/100
    12/12 [==============================] - 2s 152ms/step - loss: 0.6391 - accuracy: 0.7314 - val_loss: 0.7258 - val_accuracy: 0.6886
    Epoch 72/100
    12/12 [==============================] - 2s 153ms/step - loss: 0.6421 - accuracy: 0.7250 - val_loss: 0.7683 - val_accuracy: 0.6644
    Epoch 73/100
    12/12 [==============================] - 2s 154ms/step - loss: 0.6458 - accuracy: 0.7183 - val_loss: 0.8089 - val_accuracy: 0.6416
    Epoch 74/100
    12/12 [==============================] - 2s 152ms/step - loss: 0.6663 - accuracy: 0.7082 - val_loss: 0.7902 - val_accuracy: 0.6309
    Epoch 75/100
    12/12 [==============================] - 2s 152ms/step - loss: 0.6519 - accuracy: 0.7159 - val_loss: 0.7344 - val_accuracy: 0.6846
    Epoch 76/100
    12/12 [==============================] - 2s 154ms/step - loss: 0.6451 - accuracy: 0.7304 - val_loss: 0.7524 - val_accuracy: 0.6711
    Epoch 77/100
    12/12 [==============================] - 2s 152ms/step - loss: 0.6408 - accuracy: 0.7220 - val_loss: 0.7237 - val_accuracy: 0.6779
    Epoch 78/100
    12/12 [==============================] - 2s 152ms/step - loss: 0.6355 - accuracy: 0.7267 - val_loss: 0.7404 - val_accuracy: 0.6725
    Epoch 79/100
    12/12 [==============================] - 2s 153ms/step - loss: 0.6381 - accuracy: 0.7361 - val_loss: 0.7180 - val_accuracy: 0.6711
    Epoch 80/100
    12/12 [==============================] - 2s 154ms/step - loss: 0.6231 - accuracy: 0.7384 - val_loss: 0.7173 - val_accuracy: 0.6886
    Epoch 81/100
    12/12 [==============================] - 2s 153ms/step - loss: 0.6170 - accuracy: 0.7448 - val_loss: 0.7217 - val_accuracy: 0.6913
    Epoch 82/100
    12/12 [==============================] - 2s 151ms/step - loss: 0.6200 - accuracy: 0.7381 - val_loss: 0.7314 - val_accuracy: 0.7034
    Epoch 83/100
    12/12 [==============================] - 2s 151ms/step - loss: 0.6379 - accuracy: 0.7273 - val_loss: 0.7134 - val_accuracy: 0.6886
    Epoch 84/100
    12/12 [==============================] - 2s 152ms/step - loss: 0.6195 - accuracy: 0.7404 - val_loss: 0.7271 - val_accuracy: 0.6779
    Epoch 85/100
    12/12 [==============================] - 2s 152ms/step - loss: 0.6154 - accuracy: 0.7478 - val_loss: 0.7164 - val_accuracy: 0.6980
    Epoch 86/100
    12/12 [==============================] - 2s 152ms/step - loss: 0.6103 - accuracy: 0.7475 - val_loss: 0.7128 - val_accuracy: 0.6940
    Epoch 87/100
    12/12 [==============================] - 2s 152ms/step - loss: 0.6196 - accuracy: 0.7391 - val_loss: 0.7269 - val_accuracy: 0.6738
    Epoch 88/100
    12/12 [==============================] - 2s 152ms/step - loss: 0.6183 - accuracy: 0.7314 - val_loss: 0.7686 - val_accuracy: 0.6537
    Epoch 89/100
    12/12 [==============================] - 2s 151ms/step - loss: 0.6244 - accuracy: 0.7297 - val_loss: 0.7362 - val_accuracy: 0.6671
    Epoch 90/100
    12/12 [==============================] - 2s 151ms/step - loss: 0.6187 - accuracy: 0.7367 - val_loss: 0.7382 - val_accuracy: 0.7020
    Epoch 91/100
    12/12 [==============================] - 2s 152ms/step - loss: 0.6069 - accuracy: 0.7424 - val_loss: 0.7694 - val_accuracy: 0.6819
    Epoch 92/100
    12/12 [==============================] - 2s 151ms/step - loss: 0.6138 - accuracy: 0.7381 - val_loss: 0.7254 - val_accuracy: 0.6926
    Epoch 93/100
    12/12 [==============================] - 2s 152ms/step - loss: 0.6104 - accuracy: 0.7421 - val_loss: 0.7457 - val_accuracy: 0.6846
    Epoch 94/100
    12/12 [==============================] - 2s 151ms/step - loss: 0.6231 - accuracy: 0.7445 - val_loss: 0.7563 - val_accuracy: 0.6872
    Epoch 95/100
    12/12 [==============================] - 2s 151ms/step - loss: 0.6085 - accuracy: 0.7424 - val_loss: 0.7283 - val_accuracy: 0.6886
    Epoch 96/100
    12/12 [==============================] - 2s 153ms/step - loss: 0.6019 - accuracy: 0.7448 - val_loss: 0.7215 - val_accuracy: 0.7047
    Epoch 97/100
    12/12 [==============================] - 2s 152ms/step - loss: 0.5973 - accuracy: 0.7562 - val_loss: 0.7376 - val_accuracy: 0.6779
    Epoch 98/100
    12/12 [==============================] - 2s 153ms/step - loss: 0.6313 - accuracy: 0.7297 - val_loss: 0.7343 - val_accuracy: 0.6738
    Epoch 99/100
    12/12 [==============================] - 2s 151ms/step - loss: 0.6267 - accuracy: 0.7347 - val_loss: 0.8120 - val_accuracy: 0.6282
    Epoch 100/100
    12/12 [==============================] - 2s 151ms/step - loss: 0.6321 - accuracy: 0.7243 - val_loss: 0.7238 - val_accuracy: 0.6926
    Test accuracy:  0.7046979665756226
    Training time: 195.98048996925354 seconds
    

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

    30/30 [==============================] - 1s 14ms/step
    


    
![png](water-bottle-images-classification-cnn-resnet50_files/water-bottle-images-classification-cnn-resnet50_40_1.png)
    


    Self Train CNNs accuracy score: 0.9291084854994629
    
    
    
    30/30 [==============================] - 3s 45ms/step
    


    
![png](water-bottle-images-classification-cnn-resnet50_files/water-bottle-images-classification-cnn-resnet50_40_3.png)
    


    With Pre-trained Model (Resnet50) accuracy score: 0.6455424274973147
    


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

    30/30 [==============================] - 1s 29ms/step - loss: 0.0045 - accuracy: 0.9987
    30/30 [==============================] - 1s 29ms/step - loss: 0.0012 - accuracy: 1.0000
    15/15 [==============================] - 1s 58ms/step - loss: 0.0347 - accuracy: 0.9933
    Best: 1.000000 using {'batch_size': 128, 'epochs': 100, 'optimizer': 'adam'}
    0.998657 (0.000000) with: {'batch_size': 128, 'epochs': 50, 'optimizer': 'adam'}
    1.000000 (0.000000) with: {'batch_size': 128, 'epochs': 100, 'optimizer': 'adam'}
    0.993285 (0.000000) with: {'batch_size': 256, 'epochs': 50, 'optimizer': 'adam'}
    nan (nan) with: {'batch_size': 256, 'epochs': 100, 'optimizer': 'adam'}
    Training time: 808.008086681366 seconds
    


```python
'''
Overview detailed information about the grid search cross-validation process
'''
import pandas as pd
print(pd.DataFrame(grid_result.cv_results_))
```

       mean_fit_time  std_fit_time  mean_score_time  std_score_time  \
    0     127.332934           0.0         1.171087             0.0   
    1     258.078088           0.0         1.187089             0.0   
    2     154.280100           0.0         1.202089             0.0   
    3      13.380104           0.0         0.000000             0.0   
    
      param_batch_size param_epochs param_optimizer  \
    0              128           50            adam   
    1              128          100            adam   
    2              256           50            adam   
    3              256          100            adam   
    
                                                  params  split0_test_score  \
    0  {'batch_size': 128, 'epochs': 50, 'optimizer':...           0.998657   
    1  {'batch_size': 128, 'epochs': 100, 'optimizer'...           1.000000   
    2  {'batch_size': 256, 'epochs': 50, 'optimizer':...           0.993285   
    3  {'batch_size': 256, 'epochs': 100, 'optimizer'...                NaN   
    
       mean_test_score  std_test_score  rank_test_score  
    0         0.998657             0.0                2  
    1         1.000000             0.0                1  
    2         0.993285             0.0                3  
    3              NaN             NaN                4  
    


```python
'''
Transform predicted result into list
'''
output_labels = ['Full  Water level', 'Half water level', 'Overflowing']
result = grid.predict(X_test)

predicted_labels = list(map(lambda x: output_labels[x], result))
# print(predicted_labels[:5])
```

    30/30 [==============================] - 1s 13ms/step
    


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
    


    GridSerachCV accuracy score:0.9419978517722879
    


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

    Clipping input data to the valid range for imshow with RGB data ([0..1] for floats or [0..255] for integers).
    Clipping input data to the valid range for imshow with RGB data ([0..1] for floats or [0..255] for integers).
    Clipping input data to the valid range for imshow with RGB data ([0..1] for floats or [0..255] for integers).
    Clipping input data to the valid range for imshow with RGB data ([0..1] for floats or [0..255] for integers).
    Clipping input data to the valid range for imshow with RGB data ([0..1] for floats or [0..255] for integers).
    Clipping input data to the valid range for imshow with RGB data ([0..1] for floats or [0..255] for integers).
    Clipping input data to the valid range for imshow with RGB data ([0..1] for floats or [0..255] for integers).
    Clipping input data to the valid range for imshow with RGB data ([0..1] for floats or [0..255] for integers).
    Clipping input data to the valid range for imshow with RGB data ([0..1] for floats or [0..255] for integers).
    Clipping input data to the valid range for imshow with RGB data ([0..1] for floats or [0..255] for integers).
    Clipping input data to the valid range for imshow with RGB data ([0..1] for floats or [0..255] for integers).
    Clipping input data to the valid range for imshow with RGB data ([0..1] for floats or [0..255] for integers).
    Clipping input data to the valid range for imshow with RGB data ([0..1] for floats or [0..255] for integers).
    Clipping input data to the valid range for imshow with RGB data ([0..1] for floats or [0..255] for integers).
    Clipping input data to the valid range for imshow with RGB data ([0..1] for floats or [0..255] for integers).
    Clipping input data to the valid range for imshow with RGB data ([0..1] for floats or [0..255] for integers).
    Clipping input data to the valid range for imshow with RGB data ([0..1] for floats or [0..255] for integers).
    Clipping input data to the valid range for imshow with RGB data ([0..1] for floats or [0..255] for integers).
    Clipping input data to the valid range for imshow with RGB data ([0..1] for floats or [0..255] for integers).
    


    
![png](water-bottle-images-classification-cnn-resnet50_files/water-bottle-images-classification-cnn-resnet50_48_1.png)
    


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

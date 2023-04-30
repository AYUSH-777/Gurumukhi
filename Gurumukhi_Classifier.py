import os
import numpy as np
import matplotlib.pyplot as plt
from skimage.io import imread, imshow
from skimage.transform import resize
from sklearn.model_selection import train_test_split
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Flatten, Conv2D, MaxPooling2D

# Define the image size
IMG_HEIGHT = 32
IMG_WIDTH = 32

# Read images and labels
def read_images():
    images = []
    labels = []
    folder_path = "drive/MyDrive/GurNum/train/"
    subfolders = os.listdir(folder_path)
    for i in range(len(subfolders)):
        subfolder = subfolders[i]
        subfolder_path = os.path.join(folder_path, subfolder)
        for image_path in os.listdir(subfolder_path):
            image = imread(os.path.join(subfolder_path, image_path), as_gray=True)
            resized_image = resize(image, (IMG_HEIGHT, IMG_WIDTH), anti_aliasing=True)
            images.append(resized_image)
            labels.append(i)
    return np.array(images), np.array(labels)

# Normalize the images
def normalize_images(images):
    return images / 255.0

# Load and preprocess the data
images, labels = read_images()
images = normalize_images(images)
x_train, x_test, y_train, y_test = train_test_split(images, labels, test_size=0.2, random_state=42)

# Define the model
model = Sequential([
    Flatten(input_shape=(IMG_HEIGHT, IMG_WIDTH)),
    Dense(128, activation='relu'),
    Dense(10, activation='softmax')
])

# Compile the model
model.compile(optimizer='adam',
              loss='sparse_categorical_crossentropy',
              metrics=['accuracy'])

# Train the model
history = model.fit(x_train, y_train, epochs=10, validation_data=(x_test, y_test))

# Evaluate the model
test_loss, test_acc = model.evaluate(x_test, y_test)
print('Test accuracy:', test_acc)

# Plot the accuracy and loss curves
acc = history.history['accuracy']
val_acc = history.history['val_accuracy']
loss = history.history['loss']
val_loss = history.history['val_loss']

epochs_range = range(10)

plt.figure(figsize=(8, 8))
plt.subplot(2, 1, 1)
plt.plot(epochs_range, acc, label='Training Accuracy')
plt.plot(epochs_range, val_acc, label='Validation Accuracy')
plt.legend(loc='lower right')
plt.title('Training and Validation Accuracy')

plt.subplot(2, 1, 2)
plt.plot(epochs_range, loss, label='Training Loss')
plt.plot(epochs_range, val_loss, label='Validation Loss')
plt.legend(loc='upper right')
plt.title('Training and Validation Loss')
plt.show()

import os
import cv2
import numpy as np
from sklearn.model_selection import train_test_split
import tensorflow as tf
from tensorflow import keras
from keras import layers

data_dir = "C:\\Users\\harsh\\Desktop\\5th sem\\project\\attempt 2\\datacollected"
labels = os.listdir(data_dir)
image_size = (640, 480)  


def data_generator(data_dir, labels, image_size, batch_size):
    while True:
        X, y = [], []
        for label_id, label in enumerate(labels):
            label_path = os.path.join(data_dir, label)
            for image_file in os.listdir(label_path):
                image_path = os.path.join(label_path, image_file)
                if os.path.isfile(image_path):
                    image = cv2.imread(image_path)
                    if image is not None:
                        image = cv2.resize(image, image_size)
                        X.append(image)
                        y.append(label_id)
                        if len(X) == batch_size:
                            yield (np.array(X) / 255.0, np.array(y))
                            X, y = [], []

batch_size = 32
epochs = 10

num_classes = len(labels)

model = keras.Sequential([
    layers.Conv2D(32, (3, 3), activation='relu', input_shape=(*image_size, 3)),
    layers.MaxPooling2D((2, 2)),
    layers.Conv2D(64, (3, 3), activation='relu'),
    layers.MaxPooling2D((2, 2)),
    layers.Flatten(),
    layers.Dense(128, activation='relu'),
    layers.Dense(num_classes, activation='softmax')
])

model.compile(optimizer='adam',
              loss='sparse_categorical_crossentropy',
              metrics=['accuracy'])


train_generator = data_generator(data_dir, labels, image_size, batch_size)
val_generator = data_generator(data_dir, labels, image_size, batch_size)

train_steps_per_epoch = len(labels) * len(os.listdir(os.path.join(data_dir, labels[0]))) // batch_size
val_steps_per_epoch = len(labels) * len(os.listdir(os.path.join(data_dir, labels[0]))) // batch_size

model.fit(train_generator, validation_data=val_generator, 
          steps_per_epoch=train_steps_per_epoch,
          validation_steps=val_steps_per_epoch,
          epochs=epochs)

# Save the trained model
model.save("my_landmark_signlang_model.h5")
print("Model saved successfully.")

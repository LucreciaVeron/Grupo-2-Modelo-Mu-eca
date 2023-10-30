# Importar las bibliotecas necesarias
import os
import cv2
import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
from tensorflow import keras
from keras.preprocessing.image import ImageDataGenerator
from keras.models import Sequential
from keras.layers import Conv2D, MaxPooling2D, Flatten, Dense
from keras.optimizers import Adam
from Load_Images import preprocess_image
from sklearn.model_selection import train_test_split
from keras.layers import Flatten, Dense, Dropout

os.environ["CUDA_VISIBLE_DEVICES"] = "-1"

# Tamaño de las imágenes
image_size = (128, 128)

def load_images_from_folder(folder, image_size, label):
    images = []
    labels = []

    for case_folder in os.listdir(folder):
        image_path = os.path.join(folder, case_folder)

        image = cv2.imread(image_path)
        # Aplica el preprocesamiento a cada imagen
        image = preprocess_image(image, image_size)
        image = np.expand_dims(image, axis=-1)  # Agregar dimensión del canal
        images.append(image)

        labels.append(label) 
    return np.array(images), np.array(labels)


images_0, labels_0 = load_images_from_folder("dataset wrist x-ray\\0", image_size, 0)
images_1, labels_1 = load_images_from_folder("dataset wrist x-ray\\1", image_size, 1)

# Combinar imágenes y etiquetas
images = np.concatenate((images_0, images_1))
labels = np.concatenate((labels_0, labels_1))

# Dividir en conjuntos de entrenamiento y validación
images_train, images_valid, labels_train, labels_valid = train_test_split(images, labels, test_size=0.2, random_state=42)

model = Sequential([
    Conv2D(32, (3, 3), activation='relu', input_shape=(image_size[0], image_size[1], 1)),
    MaxPooling2D(2, 2),
    Conv2D(64, (3, 3), activation='relu'),
    MaxPooling2D(2, 2),
    Conv2D(128, (3, 3), activation='relu'),
    MaxPooling2D(2, 2),
    Flatten(),
    Dense(512, activation="relu"),
    Dense(256, activation="relu"),
    Dense(128, activation="relu"), 
    Dense(64, activation="relu"),
    Dropout(0.2), # 10% de dropout
    Dense(1, activation='sigmoid')
])


model.compile(
    optimizer="adam",
    loss="binary_crossentropy",
    metrics=[
        tf.keras.metrics.BinaryAccuracy(),
        tf.keras.metrics.AUC(),
    ],
)

checkpoint = tf.keras.callbacks.ModelCheckpoint(
    filepath="src/models/Best_CNN.h5",
    monitor="binary_accuracy",
    mode="max",
    save_best_only=True,
    verbose=1,
)

model.fit(
    images_train,
    labels_train,
    epochs=30,
    batch_size=32,
    validation_data=(images_valid, labels_valid),
    callbacks=[checkpoint],
)

model.summary()

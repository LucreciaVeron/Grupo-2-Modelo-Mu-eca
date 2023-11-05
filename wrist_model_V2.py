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
from keras import regularizers


# Tamaño de las imágenes
image_size = (256, 256)

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


images_0, labels_0 = load_images_from_folder("DATASET 3ER MODELO\\Dataset equilibrado REFINADO\\0", image_size, 0)
images_1, labels_1 = load_images_from_folder("DATASET 3ER MODELO\\Dataset equilibrado REFINADO\\1", image_size, 1)

# Combinar imágenes y etiquetas
images = np.concatenate((images_0, images_1))
labels = np.concatenate((labels_0, labels_1))

# Dividir en conjuntos de entrenamiento y validación
images_train, images_valid, labels_train, labels_valid = train_test_split(images, labels, test_size=0.3, random_state=42)

#Modelo
model = Sequential([
    Conv2D(128, (3, 3), activation='relu', input_shape=(128, 128, 1)),
    MaxPooling2D((2, 2)),
    Conv2D(256, (3, 3), activation='relu'),
    MaxPooling2D((2, 2)),
    Conv2D(512, (3, 3), activation='relu'),
    MaxPooling2D((2, 2)),

    Flatten(),
    Dense(512, activation='relu'),
    Dense(256, activation='relu'),
    Dropout(0.2), 
    Dense(64, activation='relu'),
    Dense(1, activation='sigmoid')
])

model.compile(
    optimizer=tf.keras.optimizers.Adam(learning_rate=1e-4),  # Aprendizaje más lento    
    loss="binary_crossentropy",
    metrics=[
        tf.keras.metrics.BinaryAccuracy(),
        tf.keras.metrics.AUC(),
    ],
)

checkpoint = tf.keras.callbacks.ModelCheckpoint(
    filepath="src/models/Best_CNN.h5",
    monitor="val_binary_accuracy",
    mode="max",
    save_best_only=True,
    verbose=1,
)

model.fit(
    images_train,
    labels_train,
    epochs=25,
    validation_data=(images_valid, labels_valid),
    callbacks=[checkpoint],
)

model.summary()

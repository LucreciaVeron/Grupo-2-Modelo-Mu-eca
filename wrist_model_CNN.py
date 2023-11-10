import tensorflow as tf
import numpy as np
from keras.models import Sequential
from keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout
from Load_Images import load_images_from_folder_CNN
from sklearn.model_selection import train_test_split

##################### PARÁMETROS ###############################

# Directorios con etiqueta 0 (Normal) y etiqueta 1 (Fractura de muñeca)
data_dir_0 = "DATASET 3ER MODELO\\Dataset equilibrado REFINADO\\0"
data_dir_1 = "DATASET 3ER MODELO\\Dataset equilibrado REFINADO\\1"
# Tamaño de las imágenes
image_size = (128, 128)

##################### PREPARACIÓN DE DATOS #####################

# Cargar imágenes y etiquetas para las clases 0 y 1
images_0, labels_0 = load_images_from_folder_CNN(data_dir_0, image_size, 0)
images_1, labels_1 = load_images_from_folder_CNN(data_dir_1, image_size, 1)

# Combinar imágenes y etiquetas
images = np.concatenate((images_0, images_1))
labels = np.concatenate((labels_0, labels_1))

# Dividir en conjuntos de entrenamiento y validación
images_train, images_valid, labels_train, labels_valid = train_test_split(images, labels, test_size=0.3, random_state=42)

##################### MODELO CNN PERSONALIZADO #################

# Crear el modelo
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

# Compilación del modelo
model.compile(
    optimizer=tf.keras.optimizers.Adam(learning_rate=1e-4),  # Aprendizaje más lento    
    loss="binary_crossentropy",
    metrics=[
        tf.keras.metrics.BinaryAccuracy(),
        tf.keras.metrics.AUC(),
    ],
)

# Guardar el mejor modelo según su val binary accuracy
checkpoint = tf.keras.callbacks.ModelCheckpoint(
    filepath="src/models/Best_CNN.h5",
    monitor="val_binary_accuracy",
    mode="max",
    save_best_only=True,
    verbose=1,
)

# Ajustes en el entrenamiento
model.fit(
    images_train,
    labels_train,
    epochs=30,
    validation_data=(images_valid, labels_valid),
    callbacks=[checkpoint],
)

# Muestra un resumen detallado del modelo
model.summary()
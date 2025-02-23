import tensorflow as tf
import numpy as np
from keras.models import Sequential
from Load_Images import load_images_from_folder_Inception
from sklearn.model_selection import train_test_split
from keras.layers import Flatten, Dense, Dropout
from keras.applications import InceptionV3

##################### PARÁMETROS ###############################

# Directorios con etiqueta 0 (Normal) y etiqueta 1 (Fractura de muñeca)
data_dir_0 = "DATASET 3ER MODELO\\Dataset equilibrado REFINADO\\0"
data_dir_1 = "DATASET 3ER MODELO\\Dataset equilibrado REFINADO\\1"
# Tamaño de las imágenes
image_size = (128, 128)

##################### PREPARACIÓN DE DATOS #####################

images_0, labels_0 = load_images_from_folder_Inception(data_dir_0, image_size, 0)
images_1, labels_1 = load_images_from_folder_Inception(data_dir_1, image_size, 1)

# Combinar imágenes y etiquetas
images = np.concatenate((images_0, images_1))
labels = np.concatenate((labels_0, labels_1))

# Dividir en conjuntos de entrenamiento y validación
images_train, images_valid, labels_train, labels_valid = train_test_split(images, labels, test_size=0.3, random_state=42)

##################### MODELO InceptionV3 #######################

# Generar modelo preentrenado
inception = InceptionV3(
    input_shape=(
        256, # Tamaño de la imagen
        256, # Tamaño de la imagen
        3, # 3 canales, en modelos preentrenados es obligatorio RGB
    ),
    include_top=False,  # No incluir capas completamente conectadas
    weights="imagenet",  # Cargar pesos preentrenados
)

# Evitar el entrenamiento en estas capas a futuro
inception.trainable = False

# Crear el modelo
model = Sequential([
    inception,  # Capas de InceptionV3
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
    epochs=25,
    validation_data=(images_valid, labels_valid),
    callbacks=[checkpoint],
)

# Muestra un resumen detallado del modelo
model.summary()

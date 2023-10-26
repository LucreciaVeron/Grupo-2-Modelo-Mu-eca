import os
from pathlib import Path
import tensorflow as tf
from tensorflow.keras import regularizers
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Activation, Dense, Flatten, Dropout
from tensorflow.keras.models import Sequential
import HeartImg_Prep

# Deshabilitar MKL para evitar errores de memoria
os.environ['TF_DISABLE_MKL'] = '1'

##################### PARÁMETROS #####################

size = 224
count_per_cat_train = 2000
cont_per_cat_test = (count_per_cat_train * 10) // 100

##################### PREPARACIÓN DE IMÁGENES #####################

# Cargar datos de entrenamiento
train_directory = Path("./src/train")
dataframe_train = HeartImg_Prep.load_data(train_directory, count_per_cat_train)

# Cargar datos de test
val_directory = Path("./src/test")
print(train_directory)
print(val_directory)
dataframe_val = HeartImg_Prep.load_data(val_directory, cont_per_cat_test)

# Obtener muestras aleatorias para entrenamiento y test
dataframe_train = HeartImg_Prep.obtener_muestras_aleatorias(dataframe_train, count_per_cat_train)
dataframe_val = HeartImg_Prep.obtener_muestras_aleatorias(dataframe_val, cont_per_cat_test)

# Preparación de imágenes para train y test
train_images = HeartImg_Prep.prepare_image_generator_ge(dataframe_train, "training") # Entrenamiento
val_images = HeartImg_Prep.prepare_image_generator_ge(dataframe_val, "validation") # Validación

##################### MODELO CNN PERSONALIZADO #####################

# Crear el modelo
model = Sequential([
    Conv2D(32, (3, 3), input_shape=(size, size, 1)),
    Activation("relu"),
    MaxPooling2D((2, 2)),
    Conv2D(64, (3, 3)),
    Activation("relu"),
    MaxPooling2D((2, 2)),
    Flatten(),  # Capa de aplanamiento
    Dense(256, activation="relu", kernel_regularizer=regularizers.l2(0.01)),  # Completa conectada con regularización L2
    Dropout(0.5),  # Añadimos dropout para evitar overfitting
    Dense(128, activation="relu", kernel_regularizer=regularizers.l2(0.01)),  # Completa conectada con regularización L2
    Dropout(0.5),  # Añadimos dropout para evitar overfitting
    Dense(64, activation="relu"),  # Capa completamente conectada
    Dense(2, activation="softmax")  # Capa de salida con 6 neuronas para las 6 categorías
])

# Compilación del modelo
model.compile(
    loss="categorical_crossentropy",
    optimizer=tf.keras.optimizers.Adam(learning_rate=1e-4),  # Aprendizaje más lento
    metrics=[
        tf.keras.metrics.CategoricalAccuracy(),
        tf.keras.metrics.Precision(),
        tf.keras.metrics.AUC(
            num_thresholds=200,
            curve="ROC",
            summation_method="interpolation",
            multi_label=False,
        ),
    ],
)

# Guardar el mejor modelo según su categorical accuracy (precisión para cada categoría)
checkpoint = tf.keras.callbacks.ModelCheckpoint(
    filepath="./src/models/Best_CNN.h5",
    save_weights_only=False,
    monitor="categorical_accuracy",
    mode="max",
    save_best_only=True,
    verbose=1,
)

# Ajustes en el entrenamiento
result = model.fit(
    train_images,
    validation_data=val_images,
    epochs=30,  # Numero de iteraciones por epoca
    callbacks=[checkpoint],
)

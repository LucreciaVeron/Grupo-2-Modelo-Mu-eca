import pandas as pd
import tensorflow as tf
import numpy as np
import cv2

num_batch = 32

######################## FUNCIONES ########################

# Función para cargar imágenes y etiquetas
def load_data(directory, count_per_category):
    filepaths = []
    labels = []

    for category in ["F", "NF"]: # Categorías de fracturado / no fracturado
        category_dir = directory / category
        category_filepaths = list(category_dir.glob("*.jpg"))
        category_filepaths = category_filepaths[:count_per_category]
        # Convertir las rutas de archivo a cadenas
        category_filepaths = list(map(str, category_filepaths))
        filepaths.extend(category_filepaths)
        labels.extend([category] * len(category_filepaths))

    return pd.DataFrame({"Filepath": filepaths, "Label": labels})


# Función para preprocesar la imagen en RGB
def preprocess_image(image):
    target_size = (224, 224)
    image = tf.image.resize(image, target_size)
    image = image / 255.0
    return image


# Función para preparar generadores de imágenes en RGB
def prepare_image_generator(dataframe, subset):
    generator = tf.keras.preprocessing.image.ImageDataGenerator(
        preprocessing_function=preprocess_image,
    )
    return generator.flow_from_dataframe(
        dataframe=dataframe,
        x_col="Filepath",
        y_col="Label",
        target_size=(224, 224),
        color_mode="rgb",
        class_mode="categorical",
        batch_size=num_batch,
        shuffle=True,
        seed=42,
        subset=subset,
    )

# Función para preprocesar la imagen en escala de grises
def preprocess_image_ge(image):
    target_size = (224, 224)
    if image.shape[2] == 3: # Si la imagen es RGB
        image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    image = cv2.resize(image, target_size)
    image = image / 255.0
    return np.expand_dims(image, axis=-1)  # Añadir una dimensión para representar los canales

# Función para preparar generadores de imágenes en escala de grises
def prepare_image_generator_ge(dataframe, subset):
    generator = tf.keras.preprocessing.image.ImageDataGenerator(
        preprocessing_function=preprocess_image_ge,
    )

    return generator.flow_from_dataframe(
        dataframe=dataframe,
        x_col="Filepath",
        y_col="Label",
        target_size=(224, 224),
        color_mode="grayscale",
        class_mode="categorical",
        batch_size=num_batch,
        shuffle=True if subset == "training" else False,
        seed=42,
        subset=subset,
    )

# Función para obtener muestras aleatorias por categoría
def obtener_muestras_aleatorias(dataframe, num_muestras):
    samples = []
    for category in ["F", "NF"]:
        category_slice = dataframe.query("Label == @category")
        samples.append(category_slice.sample(num_muestras, random_state=1))
    return (
        pd.concat(samples, axis=0)
        .sample(frac=1.0, random_state=1)
        .reset_index(drop=True)
    )

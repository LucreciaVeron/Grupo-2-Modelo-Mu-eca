import numpy as np
import tensorflow as tf
from tensorflow.keras.models import load_model
import HeartImg_Prep
import cv2

# Cargar el modelo entrenado
model = load_model('./src/comprimido/compressed_model.h5')

def predict_image_ge(image_path):
    # Cargar la imagen y preprocesarla
    image = cv2.imread(image_path)
    preprocessed_image = HeartImg_Prep.preprocess_image_ge(image)
    preprocessed_image = np.expand_dims(preprocessed_image, axis=0)  # Agregar dimensión del lote

    # Realizar la predicción
    predictions = model.predict(preprocessed_image)

    # Obtener la etiqueta predicha
    labels = ["F", "NF"]
    predicted_label = labels[np.argmax(predictions)] #np.argmax(predictions) toma el índice del valor más alto

    return predicted_label, predictions.flatten().tolist() #flatten() convierte el array en una lista

def predict_image(image_path):
    # Cargar la imagen y preprocesarla
    image = cv2.imread(image_path)
    preprocessed_image = HeartImg_Prep.preprocess_image(image)
    preprocessed_image = np.expand_dims(preprocessed_image, axis=0)  # Agregar dimensión del lote

    # Realizar la predicción
    predictions = model.predict(preprocessed_image)

    # Obtener la etiqueta predicha
    labels = ["F", "NF"]
    predicted_label = labels[np.argmax(predictions)]

    return predicted_label, predictions.flatten().tolist()
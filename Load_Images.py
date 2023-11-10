import os
import cv2
import numpy as np


def preprocess_image_CNN(image, image_size):
    """
    Realiza el preprocesamiento de una imagen para modelos de CNN.

    Parameters:
        image (numpy.ndarray): Imagen a preprocesar.
        image_size (tuple): Tamaño al cual redimensionar la imagen.

    Returns:
        numpy.ndarray: Imagen preprocesada.
    """
    # Convierte a escala de grises
    img_gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    # Normaliza y redimensiona la imagen
    img_resized = cv2.resize(img_gray, image_size)
    img_resized = img_resized / 255.0
    return img_resized
    
def load_images_from_folder_CNN(folder, image_size, label):
    """
    Carga imágenes desde un directorio para modelos de CNN.

    Parameters:
        folder (str): Ruta del directorio que contiene las imágenes.
        image_size (tuple): Tamaño al cual redimensionar las imágenes.
        label (int): Etiqueta para las imágenes cargadas.

    Returns:
        tuple (images, labels): Arreglo de imágenes y etiquetas.
    """
    images = []
    labels = []

    for case_folder in os.listdir(folder):
        image_path = os.path.join(folder, case_folder)

        image = cv2.imread(image_path)
        # Aplica el preprocesamiento a cada imagen
        image = preprocess_image_CNN(image, image_size)
        image = np.expand_dims(image, axis=-1)  # Agregar dimensión del canal
        images.append(image)

        labels.append(label) 
    return np.array(images), np.array(labels)

def preprocess_image_Inception(image, image_size):
    """
    Realiza el preprocesamiento de una imagen para modelos Inception.

    Parameters:
        image (numpy.ndarray): Imagen a preprocesar.
        image_size (tuple): Tamaño al cual redimensionar la imagen.

    Returns:
        numpy.ndarray: Imagen preprocesada.
    """
    # Normaliza y redimensiona la imagen
    img_resized = cv2.resize(image, image_size)
    img_resized = img_resized / 255.0
    return img_resized

def load_images_from_folder_Inception(folder, image_size, label):
    """
    Carga imágenes desde un directorio para modelos Inception.

    Parameters:
        folder (str): Ruta del directorio que contiene las imágenes.
        image_size (tuple): Tamaño al cual redimensionar las imágenes.
        label (int): Etiqueta para las imágenes cargadas.

    Returns:
        tuple (images, labels): Arreglo de imágenes y etiquetas.
    """
    images = []
    labels = []

    for case_folder in os.listdir(folder):
        image_path = os.path.join(folder, case_folder)

        image = cv2.imread(image_path)
        # Aplica el preprocesamiento a cada imagen
        image = preprocess_image_Inception(image, image_size)
        image = np.expand_dims(image, axis=-1)  # Agregar dimensión del canal
        images.append(image)

        labels.append(label) 
    return np.array(images), np.array(labels)



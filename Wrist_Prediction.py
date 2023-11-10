import os
import cv2
import numpy as np
from keras.models import load_model

# Cargar modelo
model = load_model("src\models\Best_CNN_ MODELO 15.h5")

# Predicción para modelo CNN
def predict_image_CNN(predict_dir):
    # Obtener la lista de archivos de la carpeta
    image_files = os.listdir(predict_dir)

    i=0
    z=0
    # Realizar predicciones para cada imagen en la carpeta
    for file in image_files:
        img_path = os.path.join(predict_dir, file)

        # Cargar la imagen y preprocesarla para realizar la predicción
        img = cv2.imread(img_path)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)  # Convertir a escala de grises
        img = cv2.resize(img, (128, 128))  # Ajustar tamaño a 128x128
        img = np.expand_dims(img, axis=0)
        img = img / 255.  # Normalizar la imagen

        # Realizar la predicción
        prediction = model.predict(img)
        if prediction < 0.5:
            print(f"La imagen {file} es 'normal'. ({prediction})")
            i=i+1
        else:
            print(f"La imagen {file} tiene 'fractura de muñeca'.({prediction})")
            z=z+1

    print(f"Total de rx normal: ({i})")
    print(f"Total de rx fractura: ({z})")

# Predicción para modelo Inception
def predict_image_Inception(predict_dir):
    # Obtener la lista de archivos de la carpeta
    image_files = os.listdir(predict_dir)

    i=0
    z=0
    # Realizar predicciones para cada imagen en la carpeta
    for file in image_files:
        img_path = os.path.join(predict_dir, file)

        # Cargar la imagen y preprocesarla para realizar la predicción
        img = cv2.imread(img_path)
        img = cv2.resize(img, (256, 256))  # Ajustar tamaño a 256X256
        img = np.expand_dims(img, axis=0)
        img = img / 255.  # Normalizar la imagen

        # Realizar la predicción
        prediction = model.predict(img)
        if prediction < 0.5:
            print(f"La imagen {file} es 'normal'. ({prediction})")
            i=i+1
        else:
            print(f"La imagen {file} tiene 'fractura de muñeca'.({prediction})")
            z=z+1

    print(f"Total de rx normal: ({i})")
    print(f"Total de rx fractura: ({z})")
import os
import time
import cv2
import numpy as np
import tensorflow as tf
from keras.models import load_model
from keras.preprocessing import image

# Cargar modelo
model = load_model("./src/models/Best_CNN.h5")
predict_dir = "./DATASET 3ER MODELO/Test/1"

# Obtener la lista de archivos de la carpeta
image_files = os.listdir(predict_dir)

i=0
z=0
# Realizar predicciones para cada imagen en la carpeta
for file in image_files:
    img_path = os.path.join(predict_dir, file)

    # Cargar la imagen y preprocesarla para realizar la predicción
    img = cv2.imread(img_path)
    img = cv2.resize(img, (256, 256))  # Ajustar tamaño a 128x128
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
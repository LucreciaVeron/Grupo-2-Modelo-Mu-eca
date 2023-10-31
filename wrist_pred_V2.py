import os
import cv2
import numpy as np
from keras.models import load_model
from Load_Images import preprocess_image

# Directorio que contiene las imágenes de validación
valid_folder = "./src/dataset wrist x-ray/0"

# Cargar el modelo preentrenado
model = load_model("./src/models/Best_CNN.h5")

image_size = (256, 256)

# Función para cargar y preprocesar imágenes de la carpeta "valid"
def load_images_for_prediction(folder, image_size):
    images = []

    for filename in os.listdir(folder):
        image_path = os.path.join(folder, filename)
        image = cv2.imread(image_path)
        # Aplicar el preprocesamiento a cada imagen
        image = preprocess_image(image, image_size)
        image = np.expand_dims(image, axis=-1)  # Agregar dimensión del canal
        images.append(image)

    return np.array(images)

# Realizar la carga y preprocesamiento de las imágenes de validación
images_valid_predict = load_images_for_prediction(valid_folder, image_size)

# Realizar predicciones
predictions = model.predict(images_valid_predict)

# Las predicciones contendrán los valores de probabilidad para cada clase
# Puedes convertir las probabilidades en clases (0 o 1) usando un umbral, por ejemplo 0.5
threshold = 0.5
predicted_classes = (predictions > threshold).astype(int)

# Imprimir las predicciones
print("Predicciones:")
print(predicted_classes)

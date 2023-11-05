import os
import cv2
import pandas as pd
import numpy as np
from keras.preprocessing import image
import csv

def preprocess_image(image, image_size):
    # Convierte a escala de grises
    img_gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    # Normaliza y redimensiona la imagen
    img_resized = cv2.resize(img_gray, image_size)
    img_resized = img_resized / 255.0
    return img_resized


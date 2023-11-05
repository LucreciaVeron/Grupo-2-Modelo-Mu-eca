import os
import cv2
import pandas as pd
import numpy as np
from keras.preprocessing import image
import csv

def preprocess_image(image, image_size):
    # Normaliza y redimensiona la imagen
    img_resized = cv2.resize(image, image_size)
    img_resized = img_resized / 255.0
    return img_resized


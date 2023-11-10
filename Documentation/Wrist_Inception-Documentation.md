# Documentación del código - Modelo InceptionV3 para clasificación de imágenes de radiografías de muñeca
Este código implementa un modelo de red neuronal utilizando la arquitectura InceptionV3 para la clasificación binaria de radiografías de muñeca. La clasificación se realiza entre dos categorías: 
- "Normal" (0)
- "Fractura de muñeca" (1)

## Librerías Importadas
- `tensorflow`: Librería principal para la construcción y entrenamiento de modelos de aprendizaje profundo.
- `numpy`: Biblioteca para operaciones matemáticas eficientes en Python.
- `Sequential` de `keras.models`: Proporciona una forma lineal de construir modelos de Keras, donde las capas se añaden secuencialmente una después de otra.
- `load_images_from_folder_Inception` de `Load_Images`: Módulo personalizado para cargar imágenes y etiquetas desde directorios específicos.
- `train_test_split` de`sklearn.model_selection`: Se utiliza para dividir el conjunto de datos en conjuntos de entrenamiento y validación.
- `Conv2D, MaxPooling2D, Flatten, Dense, Dropout` de `keras.layers`: Definen la arquitectura del modelo CNN.
- `InceptionV3 ` de `keras.applications`: Arquitectura InceptionV3 preentrenada para la extracción de características.

## Parámetros

- `data_dir_0`: Directorio que contiene las imágenes de la clase "Normal" 
- `data_dir_1`: Directorio que contiene las imágenes de la clase "Fractura de muñeca".
- `image_size`: Tamaño de las imágenes. Se establece en (128, 128).

## Preparación de Datos

Se cargarán las imágenes y etiquetas para las clases 0 y 1 utilizando la función `load_images_from_folder_Inception`. Estas imágenes y etiquetas se combinan y dividen en conjuntos de entrenamiento y validación mediante la función `train_test_split`.

## Modelo InceptionV3

Se genera un modelo InceptionV3 preentrenado utilizando las siguientes configuraciones:

- `input_shape`: Tamaño de la imagen de entrada, establecida en (256, 256, 3) para admitir 3 canales RGB.
- `include_top=False`: No se incluyen las capas completamente conectadas para personalizar la salida del modelo.
- `weights="imagenet"`: Pesos preentrenados en el conjunto de datos de ImageNet para aprovechar el conocimiento previo.

Luego, se crea un nuevo modelo secuencial que incluye el modelo InceptionV3, seguido de capas adicionales:

- 1 Capa de aplanamiento.
- 3 Capas completamente conectadas.
- 1 Capa de dropout del 0.2.
- 1 Capa de salida con activación `sigmoid`para clasificación binaria.

## Compilación del Modelo

El modelo se compila utilizando la función de pérdida `binary_crossentropy`, para problemas de clasificación binaria. La misma mide la discrepancia entre las predicciones del modelo y las etiquetas reales.

Se elige como optimizador `Adam` con una tasa de aprendizaje de 1e-4. Un optimizador en el contexto del aprendizaje automático es un algoritmo que determina cómo los pesos del modelo deben ser ajustados para mejorar el aprendizaje. La tasa 1e-4, es la tasa de actualización de los pesos del modelo. Esta es una tasa menor a la habitual para mitigar aún más el sobreajuste. Además, se agregan las siguientes métricas de evaluación:

-  La métrica de `BinaryAccuracy` para evaluar la precisión del modelo en términos de clasificación binaria.
- `Area under the ROC Curve (AUC)` para evaluar la capacidad de clasificación del modelo.

## Guardado del Mejor Modelo

El modelo se guarda en un archivo h5 utilizando la función `checkpoint`. Este método establece los parámetros de guardado, incluyendo la ubicación y tratamiento de capas. Además, selecciona el mejor modelo comparando la metrica `val_binary_accuracy`, quedandose siempre con la de mayor rendimiento.

## Entrenamiento del modelo

El modelo se entrena con los conjuntos de entrenamiento y validación durante 30 épocas.

## Resumen del modelo:

Se muestra un resumen detallado del modelo que incluye información sobre cada capa, el número de parámetros y la estructura general de la red.

---

**NOTA**: Este documento proporciona una descripción general de la estructura y funcionamiento del código. Para obtener detalles más específicos sobre cada parte del código, se recomienda revisar los comentarios incluidos en el mismo.

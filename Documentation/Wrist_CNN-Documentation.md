# Documentación del código- Modelo CNN para clasificación de imágenes de radiografías de muñeca

Este código implementa una red neuronal convolucional (CNN) utilizando Keras y TensorFlow para la clasificación binaria de imágenes de radiografías de muñeca.

Categorías:
- 0: Normal
- 1: Fractura de muñeca

## Librerías Importadas

- `tensorflow`: Proporciona la base para la construcción y entrenamiento del modelo de CNN.
- `numpy`: Biblioteca para operaciones matemáticas eficientes en Python.
- `Sequential` de `keras.models`: Proporciona una forma lineal de construir modelos de Keras, donde las capas se añaden secuencialmente una después de otra.
- `Conv2D, MaxPooling2D, Flatten, Dense, Dropout` de `keras.layers`: Definen la arquitectura del modelo CNN.
- `train_test_split` de`sklearn.model_selection`: Se utiliza para dividir el conjunto de datos en conjuntos de entrenamiento y validación.
- `load_images_from_folder_CNN` de`Load_Images`: Módulo personalizado que se utiliza para cargar imágenes y etiquetas desde directorios específicos.

## Parámetros:

- `data_dir_0`: Directorio que contiene las imágenes de la clase "Normal" 
- `data_dir_1`: Directorio que contiene las imágenes de la clase "Fractura de muñeca".
- `image_size`: Tamaño de las imágenes. Se establece en (128, 128).

## Preparación de Datos

Se cargan las imágenes y etiquetas para las clases 0 y 1 utilizando la función `load_images_from_folder_CNN`.
Éstas se combinan y se dividen en conjuntos de entrenamiento y validación utilizando `train_test_split`.

## Modelo CNN Personalizado

El modelo CNN 3D consta de varias `capas convolucionales (Conv2D)`, `capas de pooling(MaxPooling2D)`, `capas de aplanamiento (Flatten)`, `capas completamente conectadas(Dense)` y `capas de Dropout`.

- 3 Capas convolucionales y de activación.
- 3 Capas de pooling para reducir la dimensionalidad.
- 1 Capa de aplanamiento.
- 3 Capas completamente conectadas.
- 1 Capa de dropout del 0.2.
- 1 Capa de salida con activación `sigmoid` para clasificación binaria.

En esta etapa se toma 1 estrategia para poder compensar el sobreajuste:

- Capa de doupout: desactiva aleatoriamente un porcentaje de neuronas durante el entrenamiento, reduciendo así la interdependencia y la especialización excesiva de las neuronas.

La primera Conv2D, establece las características de las imágenes que recibe como entrada a través de `input_shape=(128, 128, 1)`. Se determina que las imágenes que recibe el modelo son de 128px de alto, 128 de ancho y de 1 canal, es decir, en escala de grises.

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

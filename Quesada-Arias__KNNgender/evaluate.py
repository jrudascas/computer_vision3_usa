import cv2
import numpy as np
import os
import joblib

# ------------------------------------------------------------ #

# Clasificación de imágenes aleatorias
test_folder = "evaluate/"
male_count = 0
female_count = 0
size = 20
valid_count = 0  # Inicializar el contador de imágenes válidas

# Cargar el modelo entrenado
knn = joblib.load('knn_model.joblib')

while valid_count < size:
    try:
        filename = np.random.choice(os.listdir(test_folder))
        if not filename.endswith('.jpg'):
            continue  # Saltar los archivos que no sean .jpg
        img = cv2.imread(os.path.join(test_folder, filename))
        img = cv2.resize(img, (100, 100))
        label = knn.predict(img.reshape(1, -1))[0]
        if label == 1:
            print(f"Archivo: {filename}, género: Male")
            male_count += 1
        else:
            print(f"Archivo: {filename}, género: Female")
            female_count += 1
        valid_count += 1  # Incrementar el contador de imágenes válidas
    except Exception as e:
        print(f"Error al clasificar la imagen {filename}: {str(e)}")
        continue

print(f"Aciertos: {valid_count} de {size}. Accuracy: {size / valid_count}")


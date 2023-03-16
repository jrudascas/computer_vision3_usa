import cv2
import numpy as np
import os
import joblib

# Definir la ruta de las carpetas de entrenamiento
male_path = "male_faces/"
female_path = "female_faces/"

# Definir las listas para las imágenes y las etiquetas
images = []
labels = []

# Cargar las imágenes de la carpeta "male_faces"
for filename in os.listdir(male_path):
    try:
        img = cv2.imread(os.path.join(male_path, filename))
        img = cv2.resize(img, (100, 100))
        images.append(img)
        labels.append(1) # etiquetar como hombre (1)
    except Exception as e:
        print(f"Error al cargar la imagen {filename}: {str(e)}")
        continue

# Cargar las imágenes de la carpeta "female_faces"
for filename in os.listdir(female_path):
    try:
        img = cv2.imread(os.path.join(female_path, filename))
        img = cv2.resize(img, (100, 100))
        images.append(img)
        labels.append(0) # etiquetar como mujer (0)
    except Exception as e:
        print(f"Error al cargar la imagen {filename}: {str(e)}")
        continue

# Convertir las listas en arrays de numpy
images = np.array(images)
labels = np.array(labels)

# Separar los datos de entrenamiento y prueba
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(images, labels, test_size=0.2, random_state=42)

# Entrenar el clasificador KNN
from sklearn.neighbors import KNeighborsClassifier
knn = KNeighborsClassifier(n_neighbors=5, p=1, weights='distance')
knn.fit(X_train.reshape(X_train.shape[0], -1), y_train)

# Guardar el modelo entrenado
joblib.dump(knn, 'knn_model.joblib')

# Evaluar el modelo
accuracy = knn.score(X_test.reshape(X_test.shape[0], -1), y_test)
print(f"Accuracy: {accuracy}")
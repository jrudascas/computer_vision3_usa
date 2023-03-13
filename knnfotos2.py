import os
import numpy as np
import cv2
from PIL import Image
from sklearn.metrics import confusion_matrix
from sklearn.metrics import roc_curve
import matplotlib.pyplot as plt
from sklearn.metrics import roc_auc_score
import pandas as pd
import seaborn as sns

def draw_knn(image_path, result):
    "Test"
    # Dibujar círculos en los puntos extraídos por KNN
    img = cv2.imread(image_path)
    for i in range(len(result)):
        if result[i] == 0:
            color = (0, 0, 255)  # Rojo para las mujeres
        else:
            color = (255, 0, 0)  # Azul para los hombres
        pt = (int(i % 100 * 25 + 12.5), int(i / 100 * 25 + 12.5))
        cv2.circle(img, pt, 5, color, -1)

    # Guardar la imagen con los puntos dibujados
    output_path = f"knn_result.png"
    cv2.imwrite(output_path, img)

def load_images():
    print("Comienza la lectura y transformación de imágenes")
    male_dir = "./TotalResize/ResizeHombre/"
    female_dir = "./TotalResize/ResizeMujer/"
    #print(os.listdir(male_dir))
    #exit()
    images = []
    labels = []

    for male, female in zip(os.listdir(male_dir), os.listdir(female_dir)):
        if male.endswith(".jpeg") and female.endswith(".jpeg"):
            # Leer la imagen en escala de grises
            img_female = cv2.imread(os.path.join(female_dir, female), cv2.IMREAD_GRAYSCALE)
            img_male = cv2.imread(os.path.join(male_dir, male), cv2.IMREAD_GRAYSCALE)
            # Agregar la imagen y la etiqueta a las listas

            images.append(img_female)
            labels.append(0)
            images.append(img_male)
            labels.append(1)
    return np.array(images), np.array(labels)

def split_data(img, label):
    print("Comienza el proceso de partición de datos")
    train_images = img[:4320]
    train_labels = label[:4320]
    test_images = img[4320:]
    test_labels = label[4320:]

    return train_images, train_labels, test_images, test_labels

def resultados(result, test_labels):
    print("Commienza el proceso de impresión de resultados")
    matrix = confusion_matrix(result, test_labels)
    tn, fp, fn, tp = confusion_matrix(result, test_labels).ravel()
    print(tn, fp, fn, tp)

    target_names = ['Mujeres','Hombres']

    # crear marco de datos de pandas Crear un conjunto de datos
    dataframe = pd.DataFrame(matrix, index=target_names, columns=target_names)

    # crear mapa de calor dibujar mapa de calor
    sns.heatmap(dataframe, annot=True, cbar=None, cmap="Blues")
    plt.title("Confusion Matrix"), plt.tight_layout()
    plt.ylabel("True Class"), plt.xlabel("Predicted Class")
    plt.show()

    fpr, tpr, thresholds = roc_curve(result, test_labels)
    plt.figure()
    plt.plot(fpr, tpr, color='darkorange',
            label='ROC curve')
    plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('Grafica Curva ROC')
    plt.legend(loc="lower right")
    plt.show()

    print(roc_auc_score(result, test_labels))

def knn(train_images, train_labels, test_images, test_labels):
    print("Comienza proceso KNN image")
    knn = cv2.ml.KNearest_create()
    # Entrenar el clasificador
    knn.train(train_images.reshape(-1, 2500).astype(np.float32), cv2.ml.ROW_SAMPLE, train_labels)

    # Evaluar el clasificador en los datos de prueba
    ret, result, neighbours, dist = knn.findNearest(test_images.reshape(-1, 2500).astype(np.float32), k=3)

    draw_knn("./TotalResize/ResizeHombre/1 (1)2699_resized.jpeg", result)

    # Calcular la precisión del clasificador
    matches = result.flatten() == test_labels
    correct = np.count_nonzero(matches)
    accuracy = (correct * 100.0) / len(test_labels)
    # Función que calcula la matríz de confusión y la curva roc
    resultados(result, test_labels)
    return  accuracy

if __name__ == "__main__":

   print("Comienza el proceso de cargue de imágenes")
   images, labels = load_images()
   train_images, train_labels, test_images, test_labels = split_data(images, labels)
   accuracy = knn(train_images, train_labels, test_images, test_labels)

   print("Accuracy:", accuracy)
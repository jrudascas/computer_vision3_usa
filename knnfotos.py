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
from sklearn.metrics import classification_report
from sklearn.neighbors import KNeighborsClassifier

def load_images():
    male_dir = "./TotalResize/ResizeHombre/"
    female_dir = "./TotalResize/ResizeMujer/"
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
    train_images = img[:4320]
    train_labels = label[:4320]
    test_images = img[4320:]
    test_labels = label[4320:]

    return train_images, train_labels, test_images, test_labels

def knn(train_images, train_labels, test_images, test_labels):
    print("Comienza proceso KNN")
    n_neighbors = 3

    knn = KNeighborsClassifier(n_neighbors)
    knn.fit(train_images.reshape(-1, 2500).astype(np.float32), train_labels)
    print('Accuracy of K-NN classifier on training set: {:.2f}'
        .format(knn.score(train_images.reshape(-1, 2500).astype(np.float32), train_labels)))
    print('Accuracy of K-NN classifier on test set: {:.2f}'
        .format(knn.score(test_images.reshape(-1, 2500).astype(np.float32), test_labels)))
    
    pred = knn.predict(test_images.reshape(-1, 2500).astype(np.float32))
    print(confusion_matrix(test_labels, pred))
    print(classification_report(test_labels, pred))

if __name__ == "__main__":

   print("Comienza el proceso de cargue de imágenes")
   images, labels = load_images()
   train_images, train_labels, test_images, test_labels = split_data(images, labels)
   knn(train_images, train_labels, test_images, test_labels)

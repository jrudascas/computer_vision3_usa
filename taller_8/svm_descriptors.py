import cv2 as cv
import numpy as np
import matplotlib.pyplot as plt
import load
import extract_face
import cv2
from sklearn.model_selection import train_test_split
import pickle


def trainData():
    """ Recorre cada imagen del conjunto de imagenes almacenado en train_data y 
        se obtione los 10 descriptores para cada imagen."""
    
    train_data = np.array(load.load_data_from_pickle('train_data.pickle'))
    trainingData = []

    for img in train_data: # 5396
        descriptores = extract_face.extract_face_descriptors(img)
        trainingData.append(descriptores)

    trainingData = np.array(trainingData, dtype=np.float32)

    # Guardar las variables en un archivo
    with open('trainingData.pickle', 'wb') as f:
        pickle.dump(trainingData, f)


# importar datos
trainingData = np.array(load.load_data_from_pickle('trainingData.pickle'))
labels = np.array(load.load_data_from_pickle('train_label.pickle'))

# print(trainingData[:5])
########################################################################################

# Split data into training and testing sets
trainData, testData, trainLabels, testLabels = train_test_split(trainingData, labels, test_size=0.2)


# Train the SVM
svm = cv.ml.SVM_create()
svm.setType(cv.ml.SVM_C_SVC)
svm.setKernel(cv.ml.SVM_LINEAR)
svm.setTermCriteria((cv.TERM_CRITERIA_MAX_ITER, 100, 1e-6))
svm.train(trainData, cv.ml.ROW_SAMPLE, trainLabels)


# Predecir etiquetas para datos de prueba
n_test = testData.shape[0]
predictions = np.zeros(n_test, dtype=np.int32)
for i in range(n_test):
    sample_mat = np.matrix(testData[i], dtype=np.float32)
    response = svm.predict(sample_mat)[1]
    predictions[i] = response

# Calcular precisión de la predicción
accuracy = np.mean(predictions == testLabels)
print("Precisión de la predicción: {:.2f}%".format(accuracy*100))

from sklearn.metrics import confusion_matrix, classification_report

# Predecir etiquetas para datos de prueba
n_test = testData.shape[0]
predictions = np.zeros(n_test, dtype=np.int32)
for i in range(n_test):
    sample_mat = np.matrix(testData[i], dtype=np.float32)
    response = svm.predict(sample_mat)[1]
    predictions[i] = response

# Calcular matriz de confusión
cm = confusion_matrix(testLabels, predictions)
print("Matriz de confusión:\n", cm)
print('===='*14)

# Calcular reporte de clasificación
cr = classification_report(testLabels, predictions)
print("Reporte de clasificación:\n", cr)
print('===='*14)
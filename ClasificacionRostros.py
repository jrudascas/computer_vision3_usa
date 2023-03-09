import os
import numpy as np
import cv2
from glob import glob
from dask import delayed
from skimage.transform import integral_image
from skimage.feature import haar_like_feature
from skimage.feature import haar_like_feature_coord
from tqdm import tqdm

from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score
import matplotlib.pyplot as plt
from skimage.feature import draw_haar_like_feature
from time import time
from sklearn import svm
from sklearn.svm import SVC
from sklearn.decomposition import PCA as RandomizedPCA
from sklearn.pipeline import make_pipeline
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.utils.fixes import loguniform
from sklearn.model_selection import RandomizedSearchCV
from sklearn.metrics import classification_report
from sklearn.metrics import ConfusionMatrixDisplay

descriptors = []
index = 0

""" Crear un directorio """
def create_dir(path):
    if not os.path.exists(path):
        os.makedirs(path)

# Método que carga todas las imagenes con extensión jpeg
def load_data(path):
    data = sorted(glob(os.path.join(path, "*.jpeg")))
    return data

# Método que recibe la imagen y extrae las características
def extract_feature_image(img, feature_type, feature_coord=None):
    ii = integral_image(img)

    return haar_like_feature(ii, 0, 0, ii.shape[1], ii.shape[0],
                             feature_type=feature_type,
                             feature_coord=feature_coord)

def SVM(label, training):
    labels = np.array(label)
    trainingData = np.matrix(training, dtype=np.float32)
    h = labels.shape
    w = trainingData.shape
    print("tamaños", h, w)
    X_train, X_test, y_train, y_test = train_test_split(
    training, label, test_size=0.25, random_state=42
    )
    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_test = scaler.transform(X_test)
    print("X_train", X_train.shape)
    print("X_test", X_test.shape)

    n_components = w[1]

    print(
        "Extracting the top %d eigenfaces from %d faces" % (n_components, X_train.shape[0])
    )
    t0 = time()
    pca = PCA(n_components=n_components, svd_solver="randomized", whiten=True).fit(X_train)
    print("done in %0.3fs" % (time() - t0))

    # eigenfaces = pca.components_.reshape((n_components, 34, 10))

    print("Projecting the input data on the eigenfaces orthonormal basis")
    t0 = time()
    X_train_pca = pca.transform(X_train)
    X_test_pca = pca.transform(X_test)
    print("X_train_pca", X_train_pca.shape)
    print("X_test_pca", X_test_pca.shape)
    print("done in %0.3fs" % (time() - t0))

    print("Fitting the classifier to the training set")
    t0 = time()
    param_grid = {
        "C": loguniform(1e3, 1e5),
        "gamma": loguniform(1e-4, 1e-1),
    }
    clf = RandomizedSearchCV(
        SVC(kernel="rbf", class_weight="balanced"), param_grid, n_iter=10
    )
    clf = clf.fit(X_train_pca, y_train)
    print("done in %0.3fs" % (time() - t0))
    print("Best estimator found by grid search:")
    print(clf.best_estimator_)

    print("Predicting people's faces on the test set")
    t0 = time()
    y_pred = clf.predict(X_test_pca)
    print("done in %0.3fs" % (time() - t0))

    target_names = ['Mujeres','Hombres']

    print("y_test",y_test, y_test.shape)
    print("y_pred",y_pred, y_pred.shape)

    print(classification_report(y_test, y_pred, target_names=target_names))
    ConfusionMatrixDisplay.from_estimator(
        clf, X_test_pca, y_test, display_labels=target_names, xticks_rotation="vertical"
    )
    plt.tight_layout()
    plt.show()

    # Train the SVM
    # clf = svm.SVC()
    # clf.fit(trainingData, labels)
    # svm = cv2.ml.SVM_create()
    # svm.setType(cv2.ml.SVM_C_SVC)
    # svm.setKernel(cv2.ml.SVM_LINEAR)
    # svm.setTermCriteria((cv2.TERM_CRITERIA_MAX_ITER, 100, 1e-6))
    # svm.train(trainingData, cv2.ml.ROW_SAMPLE, labels)
    

if __name__ == "__main__":

    """ Cargar las imagenes """
    data_path = "./data/Resize2/"
    images = load_data(data_path)

    # Para mejorar el desempeño extrae las dos características
    feature_types = ['type-2-x', 'type-2-y']
    imagetotal = []
    for idx, (foto,y) in tqdm(enumerate(zip(images, images)), total=len(images)):
        foto = foto.replace("\\", "/")
        name = foto.split("/")[-1].split(".")[0]

        # """ Leer la imagen """
        imgColor = cv2.imread(foto, cv2.IMREAD_COLOR)
        gray = cv2.cvtColor(imgColor, cv2.COLOR_BGR2GRAY)

        imagetotal.append(gray)

        arr = np.array(imagetotal) 

    print("Comienza la extracción de las características")
    t0 = time()
    # Ciclo para guardar la lista de características
    for img in arr: 
        print("Ciclo ", index)
        index += 1
        X = delayed(extract_feature_image(img, feature_types))
        X = np.array(X.compute(scheduler='single-threaded'))
        Y =  np.argsort(X)[::-1]
        YMostimportant = Y[0:10]
        descriptors.append(YMostimportant.tolist())
    print("done in %0.3fs" % (time() - t0))
    print("Comienza el proceso de clasificación SVM")
    t0 = time()
    # Genera los labels para pasarlo al SVM
    labels = np.array([0] * 2698 + [1] * 2720)
    SVM(labels, descriptors)
    print("done in %0.3fs" % (time() - t0))
        
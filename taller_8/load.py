import os
import cv2
import pickle

def load_images(folder_path, label):
    """
    Carga las imágenes de una carpeta y las redimensiona a 64x64.

    Parameters:
    folder_path (str): Ruta de la carpeta que contiene las imágenes.
    label (int): Etiqueta de clase para las imágenes (1 para hombre, 0 para mujer).

    Returns:
    images (list): Lista de imágenes redimensionadas.
    labels (list): Lista de etiquetas de clase correspondientes a las imágenes.
    """
    images = []
    labels = []

    for filename in os.listdir(folder_path):
        
        image_path = os.path.join(folder_path, filename)
        image = cv2.imread(image_path)
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        resized_image = cv2.resize(gray, (256, 256))
        images.append(resized_image)
        labels.append(label)
    
    return images, labels

def load_data_from_pickle(file_path):
    with open(file_path, 'rb') as f:
        data = pickle.load(f)
    return data

def run():

    base_folder = os.path.expanduser('~')
    female_folder = os.path.join(base_folder, 'Downloads/Female Faces')
    male_folder = os.path.join(base_folder, 'Downloads/Male Faces')


    hombres_data, hombres_label = load_images(male_folder, 1)
    mujeres_data, mujeres_label = load_images(female_folder, 0)

    train_data = hombres_data + mujeres_data
    train_label = hombres_label + mujeres_label

    # Guardar las variables en un archivo
    with open('train_data.pickle', 'wb') as f:
        pickle.dump(train_data, f)

    with open('train_label.pickle', 'wb') as f:
        pickle.dump(train_label, f)

if __name__ == "__main__":
    run()

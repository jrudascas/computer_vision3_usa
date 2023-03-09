import os
"""
Tries to remove unnecessary black borders around the images, and
"trim" the images to they take up the entirety of the image.
"""

import cv2
import numpy as np
from PIL import Image
import warnings
from multiprocessing import Pool
from tqdm import tqdm

cont = 0

def resize_mantain_aspect(image, desired_size):
    old_size = image.size # old_size[0] is in (width, height) format
    ratio = float(desired_size) / max(old_size)
    new_size = tuple([int(x * ratio) for x in old_size])
    im = image.resize(new_size, Image.ANTIALIAS)
    new_im = Image.new("RGB", (desired_size, desired_size))
    # Nueva imágen rgb del tamaño deseado
    new_im.paste(im, ((desired_size - new_size[0]) // 2, (desired_size - new_size[1]) // 2))

    # wpercent = (desired_size/float(image.size[0]))
    # hsize = int((float(image.size[1])*float(wpercent)))
    # new_im = image.resize((desired_size,hsize), resample=Image.BICUBIC)
    
    return new_im


def save_single(args):
    global cont
    img_file, input_path_folder, output_path_folder, output_size = args
    image_original = Image.open(os.path.join(input_path_folder, img_file))
    image = resize_mantain_aspect(image_original, desired_size=output_size[0])
    splitName = img_file.split('.')
    newName = splitName[0] + '_' + str(cont) + '.jpeg'
    # image = image.resize(output_size)
    image.save(os.path.join(output_path_folder + newName))
    cont += 1

def fast_image_resize(input_path_folder, output_path_folder, output_size=None):
    """
    Uses multiprocessing to make it fast
    """
    if not output_size:
        warnings.warn("Need to specify output_size! For example: output_size=100")
        exit()
    if not os.path.exists(output_path_folder):
        os.makedirs(output_path_folder)
    
    jobs = [
        (file, input_path_folder, output_path_folder, output_size)
        for file in os.listdir(input_path_folder)
    ]

    with Pool() as p:
        list(tqdm(p.imap_unordered(save_single, jobs), total=len(jobs)))


if __name__ == "__main__":
    fast_image_resize("./data/TotalFaces/", "./data/Resize2/", output_size=(25,25))

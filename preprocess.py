import cv2
import os
import numpy as np

def load_images_from_folder(folder, label, img_size=(64, 64)):
    data = []
    for filename in os.listdir(folder):
        img_path = os.path.join(folder, filename)
        img = cv2.imread(img_path)
        if img is not None:
            img = cv2.resize(img, img_size)
            data.append((img, label))
    return data

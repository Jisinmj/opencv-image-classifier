import cv2
import numpy as np
import joblib

# Load trained model
model = joblib.load("image_classifier.pkl")

def predict_image(image_path):
    img = cv2.imread(image_path)
    img = cv2.resize(img, (64, 64)).flatten().reshape(1, -1)
    prediction = model.predict(img)
    return "Dog" if prediction[0] == 1 else "Cat"

print(predict_image("test_image.jpg"))

import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score
from preprocess import load_images_from_folder

# Load dataset
cats = load_images_from_folder("data/cats", 0)
dogs = load_images_from_folder("data/dogs", 1)

dataset = cats + dogs
X = np.array([x[0].flatten() for x in dataset])  # flatten image
y = np.array([x[1] for x in dataset])

# Split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train classifier
model = SVC(kernel="linear")
model.fit(X_train, y_train)

# Predict
y_pred = model.predict(X_test)
acc = accuracy_score(y_test, y_pred)
print(f"Accuracy: {acc*100:.2f}%")

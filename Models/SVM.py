import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
import cv2
import os
from tqdm import tqdm

# Set the path to your dataset
data_dir = "C:\\Users\\user\\OneDrive\\Desktop\\Thesis\\MRI DATA\\Brain_MRI_data\\images"

# Define the classes
classes = ["NORMAL_TUMOR", "ALZ_NORMAL", "INME_NORMAL", "MENENJIT_NORMAL", "NORMAL_NORM"]

# Define image dimensions
img_size = 210

# Function to load and preprocess images
def load_data(data_dir):
    images = []
    labels = []
    for idx, cls in enumerate(classes):
        cls_dir = os.path.join(data_dir, cls)
        for img in tqdm(os.listdir(cls_dir)):
            img_path = os.path.join(cls_dir, img)
            image = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)  # Read image as grayscale
            image = cv2.resize(image, (img_size, img_size))
            flattened_image = image.flatten()  # Flatten the image
            images.append(flattened_image)
            labels.append(idx)
    images = np.array(images)
    labels = np.array(labels)
    return images, labels

# Load and preprocess the data
images, labels = load_data(data_dir)

# Normalize pixel values to be between 0 and 1
images = images.astype('float32') / 210

# Split the data into training and testing sets
x_train, x_test, y_train, y_test = train_test_split(images, labels, test_size=0.2, random_state=42)

# Train the SVM model
svm_model = SVC(kernel='linear')  # You can choose different kernels like 'linear', 'rbf', etc.
svm_model.fit(x_train, y_train)

# Predictions
y_pred_train = svm_model.predict(x_train)
y_pred_test = svm_model.predict(x_test)

# Calculate accuracy
train_accuracy = accuracy_score(y_train, y_pred_train)
test_accuracy = accuracy_score(y_test, y_pred_test)

print(f"Training Accuracy: {train_accuracy}")
print(f"Testing Accuracy: {test_accuracy}")

precision = precision_score(y_test, y_pred_test, average='weighted')
recall = recall_score(y_test, y_pred_test, average='weighted')
f1 = f1_score(y_test, y_pred_test, average='weighted')

print("Precision:", precision)
print("Recall:", recall)
print("F1 Score:", f1)
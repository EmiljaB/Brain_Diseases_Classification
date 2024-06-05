import numpy as np
import cv2
import os
import skimage
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from skimage.feature import graycomatrix, graycoprops

# Set the path to your dataset
data_dir = "C:\\Users\\user\\OneDrive\\Desktop\\Thesis\\MRI DATA\\Brain_MRI_data\\images"

# Define the classes
classes = ["NORMAL_TUMOR", "ALZ_NORMAL", "INME_NORMAL", "MENENJIT_NORMAL", "NORMAL_NORM"]

# Define GLCM parameters
distances = [1]  # Distance between pixels
angles = [0, np.pi/4, np.pi/2, 3*np.pi/4]  # Angles for co-occurrence matrix

# Define other parameters
num_features = 10  # Number of GLCM features to extract

# Function to extract GLCM features from an image
def extract_features(image):
    glcm = graycomatrix(image, distances=distances, angles=angles, levels=256, symmetric=True, normed=True)
    features = np.hstack([graycoprops(glcm, prop).ravel() for prop in ['contrast', 'dissimilarity', 'homogeneity', 'energy', 'correlation']])
    return features

# Function to load and preprocess images
def load_data(data_dir):
    images = []
    labels = []
    for idx, cls in enumerate(classes):
        cls_dir = os.path.join(data_dir, cls)
        for img in os.listdir(cls_dir):
            img_path = os.path.join(cls_dir, img)
            image = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)  # Read image as grayscale
            image = cv2.resize(image, (128, 128))  # Resize image if necessary
            images.append(image)
            labels.append(idx)
    images = np.array(images)
    labels = np.array(labels)
    return images, labels

# Load and preprocess the data
images, labels = load_data(data_dir)

# Extract features using GLCM
X = np.array([extract_features(image) for image in images])

# Split the data into training and testing sets
x_train, x_test, y_train, y_test = train_test_split(X, labels, test_size=0.3, random_state=42)

# Initialize kNN classifier
knn_classifier = KNeighborsClassifier(n_neighbors=5)  # You can adjust the number of neighbors

# Train the kNN classifier
knn_classifier.fit(x_train, y_train)

# Predict labels for test set
y_pred = knn_classifier.predict(x_test)
precision = precision_score(y_test, y_pred, average='weighted')
recall = recall_score(y_test, y_pred, average='weighted')
f1 = f1_score(y_test, y_pred, average='weighted')
# Calculate accuracy
accuracy = accuracy_score(y_test, y_pred)
print("Accuracy:", accuracy)
print("Precision:", precision)
print("Recall:", recall)
print("F1 Score:", f1)
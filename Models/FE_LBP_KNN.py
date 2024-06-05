import numpy as np
import cv2
import os
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from skimage.feature import local_binary_pattern

# Set the path to your dataset
data_dir = "C:\\Users\\user\\OneDrive\\Desktop\\Thesis\\MRI DATA\\Brain_MRI_data\\images"

# Define the classes
classes = ["NORMAL_TUMOR", "ALZ_NORMAL", "INME_NORMAL", "MENENJIT_NORMAL", "NORMAL_NORM"]

# Define LBP parameters
radius = 5
n_points = 8 * radius

# Define other parameters
num_features = 10  # Number of LBP features to extract

# Function to extract LBP features from an image
def extract_features(image):
    lbp = local_binary_pattern(image, n_points, radius, method='uniform')
    hist, _ = np.histogram(lbp.ravel(), bins=np.arange(0, n_points + 3), range=(0, n_points + 2))
    hist = hist.astype("float")
    hist /= (hist.sum() + 1e-7)  # Normalize histogram
    return hist

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

# Extract features using LBP
X = np.array([extract_features(image) for image in images])

# Split the data into training and testing sets
x_train, x_test, y_train, y_test = train_test_split(X, labels, test_size=0.3, random_state=42)

# Initialize kNN classifier
knn_classifier = KNeighborsClassifier(n_neighbors=5)  # You can adjust the number of neighbors

# Train the kNN classifier
knn_classifier.fit(x_train, y_train)

# Predict labels for test set
y_pred = knn_classifier.predict(x_test)

# Calculate accuracy
accuracy = accuracy_score(y_test, y_pred)
precision = precision_score(y_test, y_pred, average='weighted')
recall = recall_score(y_test, y_pred, average='weighted')
f1 = f1_score(y_test, y_pred, average='weighted')
print("Accuracy:", accuracy)
print("Precision:", precision)
print("Recall:", recall)
print("F1 Score:", f1)
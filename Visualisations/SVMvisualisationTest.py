import cv2
import numpy as np
import os
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from sklearn.decomposition import PCA
from tqdm import tqdm
import matplotlib.pyplot as plt

# Set the path to your dataset
data_dir = "C:\\Users\\user\\OneDrive\\Desktop\\Thesis\\Brain_MRI_data\\images"

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

# load the data
images, labels = load_data(data_dir)

# normalize pixel values to be between 0 and 1
images = images.astype('float32') / 210

# split the data
x_train, x_test, y_train, y_test = train_test_split(images, labels, test_size=0.3, random_state=42)

# Train the SVM model
svm_model = SVC(kernel='linear')  # You can choose different kernels like 'linear', 'rbf', etc.
svm_model.fit(x_train, y_train)

# predictions
y_pred_train = svm_model.predict(x_train)
y_pred_test = svm_model.predict(x_test)

# calculate accuracy
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

# apply PCA to reduce the feature space to 2 dimensions for visualization
pca = PCA(n_components=2)
x_train_pca = pca.fit_transform(x_train)

# train the SVM model using reduced feature space
svm_model.fit(x_train_pca, y_train)

# decision boundaries
plt.figure(figsize=(10, 8))
h = .02  # step size in the mesh
x_min, x_max = x_train_pca[:, 0].min() - 1, x_train_pca[:, 0].max() + 1
y_min, y_max = x_train_pca[:, 1].min() - 1, x_train_pca[:, 1].max() + 1
xx, yy = np.meshgrid(np.arange(x_min, x_max, h),
                     np.arange(y_min, y_max, h))

Z = svm_model.predict(np.c_[xx.ravel(), yy.ravel()])

Z = Z.reshape(xx.shape)
plt.contourf(xx, yy, Z, cmap=plt.cm.coolwarm, alpha=0.8)

# Plot
plt.scatter(x_train_pca[:, 0], x_train_pca[:, 1], c=y_train, cmap=plt.cm.coolwarm, edgecolors='k')
plt.title('SVM Decision Boundaries (PCA-reduced space)')
plt.xlabel('Principal Component 1')
plt.ylabel('Principal Component 2')
plt.show()

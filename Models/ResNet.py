import numpy as np
import tensorflow as tf
from tensorflow.keras.applications import ResNet50
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Flatten
from sklearn.model_selection import train_test_split
import cv2
import os
from tqdm import tqdm
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

# Set the path to your dataset
data_dir = "C:\\Users\\user\\OneDrive\\Desktop\\Thesis\\MRI DATA\\Brain_MRI_data\\images"

# Define the classes
classes = ["NORMAL_TUMOR", "ALZ_NORMAL", "INME_NORMAL", "MENENJIT_NORMAL", "NORMAL_NORM"]

# Define image dimensions
img_size = 224  # ResNet50 expects input images to be 224x224

# Function to load and preprocess images
def load_data(data_dir):
    images = []
    labels = []
    for idx, cls in enumerate(classes):
        cls_dir = os.path.join(data_dir, cls)
        for img in tqdm(os.listdir(cls_dir)):
            img_path = os.path.join(cls_dir, img)
            image = cv2.imread(img_path)
            image = cv2.resize(image, (img_size, img_size))
            images.append(image)
            labels.append(idx)
    images = np.array(images)
    labels = np.array(labels)
    return images, labels

# Load and preprocess the data
images, labels = load_data(data_dir)

# Normalize pixel values to be between 0 and 1
images = images.astype('float32') / 255

# Split the data into training and testing sets
x_train, x_test, y_train, y_test = train_test_split(images, labels, test_size=0.2, random_state=42)

# Load pre-trained ResNet50 model
resnet_model = ResNet50(weights='imagenet', include_top=False, input_shape=(img_size, img_size, 3))

# Freeze pre-trained layers
for layer in resnet_model.layers:
    layer.trainable = False

# Create a new model
model = Sequential([
    resnet_model,
    Flatten(),
    Dense(256, activation='relu'),
    Dense(len(classes), activation='softmax')
])

# Compile the model
model.compile(optimizer='adam',
              loss='sparse_categorical_crossentropy',
              metrics=['accuracy'])

# Train the model
model.fit(x_train, y_train, epochs=300, batch_size=32, validation_data=(x_test, y_test))

# Evaluate the model
y_pred = model.predict(x_test)
y_pred_classes = np.argmax(y_pred, axis=1)

precision = precision_score(y_test, y_pred_classes, average='weighted')
recall = recall_score(y_test, y_pred_classes, average='weighted')
f1 = f1_score(y_test, y_pred_classes, average='weighted')

test_loss, test_acc = model.evaluate(x_test, y_test)
print(f'Test Accuracy: {test_acc}')

print("Precision:", precision)
print("Recall:", recall)
print("F1 Score:", f1)

# Save the model
model.save("brain_mri_resnet_model.h5")

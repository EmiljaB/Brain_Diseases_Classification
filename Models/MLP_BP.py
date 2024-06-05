import numpy as np
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Flatten
from tensorflow.keras.optimizers import Adam
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
img_size = 128  # Adjust as per your image dimensions

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
            images.append(image)
            labels.append(idx)
    images = np.array(images)
    labels = np.array(labels)
    return images, labels

# Load and preprocess the data
images, labels = load_data(data_dir)

# Normalize pixel values to be between 0 and 1
images = images.astype('float32') / 255

# Flatten the images
images = images.reshape(images.shape[0], -1)

# Split the data into training and testing sets
x_train, x_test, y_train, y_test = train_test_split(images, labels, test_size=0.2, random_state=42)

# Build the neural network model
model = Sequential([
    Dense(256, activation='relu', input_shape=(img_size * img_size,)),
    Dense(128, activation='relu'),
    Dense(len(classes), activation='softmax')
])

# Compile the model
model.compile(optimizer=Adam(), loss='sparse_categorical_crossentropy', metrics=['accuracy'])

# Train the model
model.fit(x_train, y_train, epochs=300, batch_size=32, validation_data=(x_test, y_test))

y_pred = model.predict(x_test)
y_pred_classes = np.argmax(y_pred, axis=1)

# Evaluate the model
precision = precision_score(y_test, y_pred_classes, average='weighted')
recall = recall_score(y_test, y_pred_classes, average='weighted')
f1 = f1_score(y_test, y_pred_classes, average='weighted')

test_loss, test_acc = model.evaluate(x_test, y_test)
print(f'Test Accuracy: {test_acc}')


print("Precision:", precision)
print("Recall:", recall)
print("F1 Score:", f1)
# Save the model
model.save("brain_mri_mlp_model.keras")
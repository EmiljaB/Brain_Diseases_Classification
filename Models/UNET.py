import numpy as np
import tensorflow as tf
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, Conv2D, MaxPooling2D, Dropout, Conv2DTranspose, concatenate
from sklearn.model_selection import train_test_split
import cv2
import os
from tqdm import tqdm
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
import matplotlib.pyplot as plt

# Set the path to your dataset
data_dir = "C:\\Users\\user\\OneDrive\\Desktop\\Thesis\\MRI DATA\\Brain_MRI_data\\images"

# Define the classes
classes = ["NORMAL_TUMOR", "ALZ_NORMAL", "INME_NORMAL", "MENENJIT_NORMAL", "NORMAL_NORM"]

# Define image dimensions
img_size = 128  # Adjust as per your image dimensions

# Function to load and preprocess images and masks
def load_data(data_dir):
    images = []
    masks = []
    for idx, cls in enumerate(classes):
        cls_dir = os.path.join(data_dir, cls)
        for img in tqdm(os.listdir(cls_dir)):
            img_path = os.path.join(cls_dir, img)
            image = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)  # Read image as grayscale
            image = cv2.resize(image, (img_size, img_size))
            images.append(image)
            # Create mask for segmentation
            mask = np.zeros_like(image)
            mask[image > 0] = 255  # Set mask value to 255 where image intensity is non-zero
            masks.append(mask)
    images = np.array(images)
    masks = np.array(masks)
    return images, masks

# Load and preprocess the data
images, masks = load_data(data_dir)

# Normalize pixel values to be between 0 and 1
images = images.astype('float32') / 255
masks = masks.astype('float32') / 255

# Add channel dimension for masks
masks = masks[..., np.newaxis]

# Split the data into training and testing sets
x_train, x_test, y_train, y_test = train_test_split(images, masks, test_size=0.2, random_state=42)

# Define U-Net architecture
def unet(input_size=(img_size, img_size, 1)):
    inputs = Input(input_size)
    conv1 = Conv2D(64, 3, activation='relu', padding='same')(inputs)
    conv1 = Conv2D(64, 3, activation='relu', padding='same')(conv1)
    pool1 = MaxPooling2D(pool_size=(2, 2))(conv1)

    conv2 = Conv2D(128, 3, activation='relu', padding='same')(pool1)
    conv2 = Conv2D(128, 3, activation='relu', padding='same')(conv2)
    pool2 = MaxPooling2D(pool_size=(2, 2))(conv2)

    conv3 = Conv2D(256, 3, activation='relu', padding='same')(pool2)
    conv3 = Conv2D(256, 3, activation='relu', padding='same')(conv3)
    pool3 = MaxPooling2D(pool_size=(2, 2))(conv3)

    conv4 = Conv2D(512, 3, activation='relu', padding='same')(pool3)
    conv4 = Conv2D(512, 3, activation='relu', padding='same')(conv4)
    drop4 = Dropout(0.5)(conv4)
    pool4 = MaxPooling2D(pool_size=(2, 2))(drop4)

    conv5 = Conv2D(1024, 3, activation='relu', padding='same')(pool4)
    conv5 = Conv2D(1024, 3, activation='relu', padding='same')(conv5)
    drop5 = Dropout(0.5)(conv5)

    up6 = Conv2DTranspose(512, (2, 2), strides=(2, 2), padding='same')(drop5)
    up6 = concatenate([up6, drop4], axis=3)
    conv6 = Conv2D(512, 3, activation='relu', padding='same')(up6)
    conv6 = Conv2D(512, 3, activation='relu', padding='same')(conv6)

    up7 = Conv2DTranspose(256, (2, 2), strides=(2, 2), padding='same')(conv6)
    up7 = concatenate([up7, conv3], axis=3)
    conv7 = Conv2D(256, 3, activation='relu', padding='same')(up7)
    conv7 = Conv2D(256, 3, activation='relu', padding='same')(conv7)

    up8 = Conv2DTranspose(128, (2, 2), strides=(2, 2), padding='same')(conv7)
    up8 = concatenate([up8, conv2], axis=3)
    conv8 = Conv2D(128, 3, activation='relu', padding='same')(up8)
    conv8 = Conv2D(128, 3, activation='relu', padding='same')(conv8)

    up9 = Conv2DTranspose(64, (2, 2), strides=(2, 2), padding='same')(conv8)
    up9 = concatenate([up9, conv1], axis=3)
    conv9 = Conv2D(64, 3, activation='relu', padding='same')(up9)
    conv9 = Conv2D(64, 3, activation='relu', padding='same')(conv9)

    outputs = Conv2D(1, 1, activation='sigmoid')(conv9)

    model = Model(inputs=[inputs], outputs=[outputs])
    return model

# Define U-Net model
model = unet(input_size=(img_size, img_size, 1))

# Compile the model
model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

# Train the model
model.fit(x_train, y_train, epochs=300, batch_size=32, validation_data=(x_test, y_test))

# Make predictions
y_pred = model.predict(x_test)
y_pred = (y_pred > 0.5).astype(np.uint8)  # Threshold predictions to get binary masks

# Flatten the arrays to compute metrics
y_test_flat = y_test.flatten()
y_pred_flat = y_pred.flatten()

# Compute metrics
accuracy = accuracy_score(y_test_flat, y_pred_flat)
precision = precision_score(y_test_flat, y_pred_flat)
recall = recall_score(y_test_flat, y_pred_flat)
f1 = f1_score(y_test_flat, y_pred_flat)

print(f'Accuracy: {accuracy:.4f}')
print(f'Precision: {precision:.4f}')
print(f'Recall: {recall:.4f}')
print(f'F1 Score: {f1:.4f}')

# Save the model
model.save("brain_mri_unet_model.h5")

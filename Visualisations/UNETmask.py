import numpy as np
import cv2
from tensorflow.keras.models import load_model
import matplotlib.pyplot as plt

# load the trained U-Net model
model = load_model("brain_mri_unet_model.h5")

# image dimensions
img_size = 210
def predict_mask(image_path):
    # Load and preprocess the input image
    image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
    image = cv2.resize(image, (img_size, img_size))
    image = image.astype('float32') / 255
    image = image[np.newaxis, ..., np.newaxis]  # Add batch and channel dimensions

    mask = model.predict(image)

    return mask.squeeze()  # Remove batch dimension

# input image
input_image_path = "norm_10.jpg"

# Predict mask for image
predicted_mask = predict_mask(input_image_path)

# Display the input and mask
input_image = cv2.imread(input_image_path)
plt.figure(figsize=(10, 5))
plt.subplot(1, 2, 1)
plt.imshow(input_image, cmap='gray')
plt.title('Input Image')
plt.subplot(1, 2, 2)
plt.imshow(predicted_mask, cmap='gray')
plt.title('Predicted Mask')
plt.show()

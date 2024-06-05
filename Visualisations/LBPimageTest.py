import cv2
import numpy as np
import matplotlib.pyplot as plt
from skimage.feature import local_binary_pattern

# Define the path to the image file
image_path = '/images/inme_15.jpg'

# Load the image
image = cv2.imread(image_path)

# Check if the image was loaded successfully
if image is None:
    raise FileNotFoundError(f"Image file '{image_path}' not found or unable to read")

# Convert to grayscale
gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

# Calculate LBP
radius = 3
n_points = 8 * radius
lbp = local_binary_pattern(gray_image, n_points, radius, method='uniform')

# Display the grayscale image and LBP image
plt.figure(figsize=(12, 6))

plt.subplot(1, 2, 1)
plt.title('Grayscale Image')
plt.imshow(gray_image, cmap='gray')
plt.axis('off')

plt.subplot(1, 2, 2)
plt.title('Local Binary Pattern (LBP)')
plt.imshow(lbp, cmap='gray')
plt.axis('off')

plt.show()

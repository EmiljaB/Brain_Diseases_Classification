import cv2
import numpy as np
import matplotlib.pyplot as plt
from skimage.feature import graycomatrix, graycoprops

# Load the image
image_path = "/images/inme_15.jpg"

image = cv2.imread(image_path)
# Check if the image was loaded successfully
if image is None:
    raise FileNotFoundError(f"Image file '{image_path}' not found or unable to read")

# Convert to grayscale
gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

# Display the grayscale image to ensure it's correct
#plt.figure(figsize=(6, 6))
#plt.title('Grayscale Image')
#plt.imshow(gray_image, cmap='gray')
#plt.show()

# Define GLCM parameters
distances = [1]  # Pixel pair distance
angles = [0, np.pi/4, np.pi/2, 3*np.pi/4]  # Angles for the pixel pairs

# Calculate GLCM
glcm = graycomatrix(gray_image, distances, angles, symmetric=True, normed=True)

# Print the GLCM to check values
print("GLCM Matrix:")
print(glcm[:, :, 0, 0])

# Print min and max values of the GLCM for debugging
glcm_min = np.min(glcm[:, :, 0, 0])
glcm_max = np.max(glcm[:, :, 0, 0])
print(f"GLCM Min Value: {glcm_min}")
print(f"GLCM Max Value: {glcm_max}")

# Normalize and scale the GLCM for better visualization
scaled_glcm = glcm[:, :, 0, 0].astype(float)
if glcm_max != 0:
    scaled_glcm /= glcm_max
    scaled_glcm *= 255  # Scale to range [0, 255] for better visibility

# Enhance contrast using a logarithmic scale for visualization
log_glcm = np.log1p(scaled_glcm)  # log1p avoids log(0) issues

# Illustrate the normalized GLCM
plt.figure(figsize=(10, 8))
plt.title('Scaled and Log-Scaled Gray Level Co-occurrence Matrix')
plt.imshow(log_glcm, cmap='viridis', interpolation='nearest')
plt.colorbar()
plt.show()

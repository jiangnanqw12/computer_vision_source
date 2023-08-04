# Importing necessary libraries
from matplotlib import pyplot as plt
import cv2

# Load the image
image_path = 'Interview/4.jpg'
image = cv2.imread(image_path)
image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB) # Converting from BGR to RGB

# Display the original image
plt.imshow(image)
plt.axis('off')
plt.title('Original Image')
plt.show()

# Convert the image to grayscale
gray_image = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)

# Apply Gaussian blur
gray_image = cv2.GaussianBlur(gray_image, (5, 5), 0)
# Apply a threshold to separate objects from the background
_, thresholded_image = cv2.threshold(gray_image, 127, 255, cv2.THRESH_BINARY)

# Find the contours
contours, _ = cv2.findContours(thresholded_image, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

# Draw the contours on a blank image
contour_image = image.copy()
cv2.drawContours(contour_image, contours, -1, (0, 255, 0), 3)

# Display the image with contours
plt.imshow(contour_image)
plt.axis('off')
plt.title('Contours of Shapes')
plt.show()

# Importing NumPy library
import numpy as np

# Create a blank image to draw filled shapes
filled_shapes_image = image.copy()

# Iterate through the contours and fill each shape with its original color
for contour in contours:
    # Create a mask for the current shape
    mask = np.zeros_like(gray_image)
    cv2.drawContours(mask, [contour], -1, 255, -1)

    # Find the color of the original shape by taking the mean color within the mask
    shape_color = cv2.mean(image, mask=mask)[:3]
    shape_color = tuple(map(int, shape_color))

    # Fill the shape with the original color
    cv2.drawContours(filled_shapes_image, [contour], -1, shape_color, -1)

# Display the image with filled shapes
plt.imshow(filled_shapes_image)
plt.axis('off')
plt.title('Filled Shapes')
plt.show()

# Saving the filled shapes image
filled_shapes_image_path = "output/filled_shapes4.jpg"
cv2.imwrite(filled_shapes_image_path, filled_shapes_image)
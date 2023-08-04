# Importing necessary libraries
import cv2
import numpy as np
import matplotlib.pyplot as plt

# Reading the image
image_path = 'Interview/4.jpg'
image = cv2.imread(image_path)

# Converting the image to grayscale for contour detection
gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

# Applying a threshold to the grayscale image
ret, thresh = cv2.threshold(gray_image, 127, 255, cv2.THRESH_BINARY)

# Finding contours using OpenCV
contours, hierarchy = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

# Drawing the contours on a blank image
contour_image = np.zeros_like(image)
for contour in contours:
    cv2.drawContours(contour_image, [contour], -1, (255, 255, 255), 1)

plt.imshow(cv2.cvtColor(contour_image, cv2.COLOR_BGR2RGB))
plt.axis('off')
plt.title('Extracted Contours')
plt.show()

# Function to fill the extracted shapes with original colors
def fill_shapes(image, contours):
    filled_image = np.zeros_like(image)
    for contour in contours:
        mask = np.zeros_like(gray_image)
        cv2.drawContours(mask, [contour], -1, 255, -1)
        color = cv2.mean(image, mask=mask)
        cv2.drawContours(filled_image, [contour], -1, tuple(map(int, color[:3])), -1)
    return filled_image

# Filling the extracted shapes with original colors
filled_image = fill_shapes(image, contours)

plt.imshow(cv2.cvtColor(filled_image, cv2.COLOR_BGR2RGB))
plt.axis('off')
plt.title('Filled Shapes')
plt.show()

# Saving the filled shapes image
filled_image_path = 'filled_shapes.jpg'
cv2.imwrite(filled_image_path, filled_image)

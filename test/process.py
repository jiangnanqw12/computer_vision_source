# Importing necessary libraries
import cv2
import numpy as np
import matplotlib.pyplot as plt

# Reading the image
image_path = '/mnt/data/1.jpg'
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

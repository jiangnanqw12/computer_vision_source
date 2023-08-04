# Importing required libraries
import cv2
import matplotlib.pyplot as plt

# Loading the provided image
image_path = "Interview/2.jpg"
image = cv2.imread(image_path)

# Converting the image to RGB (from BGR) for display
image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

# Displaying the original image
plt.imshow(image)
plt.axis('off')
plt.title('Original Image')
plt.show()

# Converting the image to grayscale
gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

# Applying Gaussian blur to reduce noise
blurred_image = cv2.GaussianBlur(gray_image, (5, 5), 0)

# Thresholding the image to create a binary image
_, thresholded_image = cv2.threshold(blurred_image, 127, 255, cv2.THRESH_BINARY_INV)

# Finding the contours
contours, _ = cv2.findContours(thresholded_image, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

# Creating a blank canvas to draw the contours
contour_image = image.copy()
contour_image[:] = 0 # Setting all pixels to black

# Drawing the contours
for contour in contours:
    cv2.drawContours(contour_image, [contour], -1, (255, 255, 255), 2)

# Converting the contour image to RGB for display
contour_image_rgb = cv2.cvtColor(contour_image, cv2.COLOR_BGR2RGB)

# Displaying the contour image
plt.imshow(contour_image_rgb)
plt.axis('off')
plt.title('Extracted Contours')
plt.show()

# Creating a blank canvas to draw the filled shapes
filled_shapes_image = image.copy()
filled_shapes_image[:] = 0 # Setting all pixels to black

# Filling the contours with the original color from the image
for contour in contours:
    # Finding the bounding box of the contour
    x, y, w, h = cv2.boundingRect(contour)

    # Extracting the corresponding area from the original image
    original_area = image[y:y+h, x:x+w]

    # Creating a mask of the same size as the bounding box
    mask = thresholded_image[y:y+h, x:x+w]

    # Applying the mask to the original area and placing it on the filled shapes image
    filled_area = cv2.bitwise_and(original_area, original_area, mask=mask)
    filled_shapes_image[y:y+h, x:x+w] = filled_area

# Converting the filled shapes image to RGB for display
filled_shapes_image_rgb = cv2.cvtColor(filled_shapes_image, cv2.COLOR_BGR2RGB)

# Displaying the filled shapes image
plt.imshow(filled_shapes_image_rgb)
plt.axis('off')
plt.title('Filled Shapes')
plt.show()

# Saving the filled shapes image
filled_shapes_image_path = "output/filled_shapes2.jpg"
cv2.imwrite(filled_shapes_image_path, filled_shapes_image)


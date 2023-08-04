from matplotlib import pyplot as plt
import cv2

# Load the original image
image_path = 'Interview/3.jpg'
image_original = cv2.imread(image_path)
image_original = cv2.cvtColor(image_original, cv2.COLOR_BGR2RGB) # Convert from BGR to RGB

# Display the original image
plt.imshow(image_original)
plt.axis('off')
plt.title('Original Image')
plt.show()
# Convert the image to grayscale
image_gray = cv2.cvtColor(image_original, cv2.COLOR_RGB2GRAY)

# Apply Gaussian blur
image_blur = cv2.GaussianBlur(image_gray, (5, 5), 0)

# Canny edge detection
edges = cv2.Canny(image_blur, 50, 150)

# Find contours
contours, _ = cv2.findContours(edges.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

# Draw the contours on the original image
image_with_contours = image_original.copy()
cv2.drawContours(image_with_contours, contours, -1, (0, 255, 0), 3)

plt.imshow(image_with_contours)
plt.axis('off')
plt.title('Contours Extracted')
plt.show()
import numpy as np

# Create an empty canvas to draw the filled shapes
image_filled_shapes = np.zeros_like(image_original)

# Iterate through the contours and fill each shape with its original colors
for contour in contours:
    # Create a mask for the current contour
    mask = np.zeros_like(image_gray)
    cv2.drawContours(mask, [contour], -1, 255, thickness=cv2.FILLED)

    # Extract the original colors using the mask
    original_colors = cv2.bitwise_and(image_original, image_original, mask=mask)

    # Find the bounding box of the contour to place the filled shape at the correct location
    x, y, w, h = cv2.boundingRect(contour)
    image_filled_shapes[y:y+h, x:x+w] = original_colors[y:y+h, x:x+w]

plt.imshow(image_filled_shapes)
plt.axis('off')
plt.title('Filled Shapes')
plt.show()

# Saving the filled shapes image
filled_shapes_image_path = "output/filled_shapes3.jpg"
cv2.imwrite(filled_shapes_image_path, image_filled_shapes)

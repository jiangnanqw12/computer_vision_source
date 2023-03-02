import cv2

# Load the image
img = cv2.imread('box.png')

# Create BRISK detector and descriptor
brisk = cv2.BRISK_create()

# Detect keypoints
keypoints = brisk.detect(img, None)

# Compute descriptors
keypoints, descriptors = brisk.compute(img, keypoints)

# Draw keypoints on the image
img_with_keypoints = cv2.drawKeypoints(img, keypoints, None)

# Display the image with keypoints
cv2.imshow('BRISK keypoints', img_with_keypoints)
cv2.waitKey(0)
cv2.destroyAllWindows()

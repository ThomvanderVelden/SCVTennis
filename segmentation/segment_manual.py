import cv2
import numpy as np
import matplotlib.pyplot as plt

# Load the image
image_path = r"C:\Users\kuipe\OneDrive\Bureaublad\TU Delft\Master\Deep vision seminar\Project\output_frames\frame_14.jpg"
image = cv2.imread(image_path)

# Convert the image to HSV
hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)

# Define the range for green colors
lower_green = np.array([35, 50, 50])
upper_green = np.array([85, 255, 255])

# Create a mask for the green colors
mask_green = cv2.inRange(hsv, lower_green, upper_green)

# Make the green areas white
image_white_bg = image.copy()
image_white_bg[mask_green != 0] = [255, 255, 255]

# Display the original and modified images
plt.figure(figsize=(12, 6))

plt.subplot(1, 2, 1)
plt.title("Original Image")
plt.imshow(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
plt.axis("off")

plt.subplot(1, 2, 2)
plt.title("Green Removed")
plt.imshow(cv2.cvtColor(image_white_bg, cv2.COLOR_BGR2RGB))
plt.axis("off")

plt.show()

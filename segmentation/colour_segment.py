from PIL import Image
import numpy as np

# Load the image
image_path = "path_to_your_image.png"
image = Image.open(image_path)

# Convert the image to numpy array
image_array = np.array(image)

# Define the color ranges for segmentation
lower_green = np.array([0, 100, 0])
upper_green = np.array([100, 255, 100])

# Define the specific color and a small range around it
specific_color = np.array([220, 203, 168])
tolerance = 10
lower_specific = specific_color - tolerance
upper_specific = specific_color + tolerance

# Create a mask for greenish colors
mask_greenish = (image_array[:, :, 0] >= lower_green[0]) & (image_array[:, :, 0] <= upper_green[0]) & \
                (image_array[:, :, 1] >= lower_green[1]) & (image_array[:, :, 1] <= upper_green[1]) & \
                (image_array[:, :, 2] >= lower_green[2]) & (image_array[:, :, 2] <= upper_green[2])

# Create a mask for the specific color to keep within the tolerance range
mask_specific = (image_array[:, :, 0] >= lower_specific[0]) & (image_array[:, :, 0] <= upper_specific[0]) & \
                (image_array[:, :, 1] >= lower_specific[1]) & (image_array[:, :, 1] <= upper_specific[1]) & \
                (image_array[:, :, 2] >= lower_specific[2]) & (image_array[:, :, 2] <= upper_specific[2])

# Apply the masks to turn greenish colors to white
image_array[mask_greenish & ~mask_specific] = [255, 255, 255]

# Convert the array back to an image
segmented_image = Image.fromarray(image_array)

# Save the result
segmented_image_path = "segmented_image.png"
segmented_image.save(segmented_image_path)

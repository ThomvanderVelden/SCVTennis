import cv2
import numpy as np
import os


# Read the video
methodstring = 'classic'
read_filename = 'rotated_backview_decent.MOV'
video = cv2.VideoCapture(f'input_videos/recorded/{read_filename}')
# Load image
# Set the video capture's position
fps = video.get(cv2.CAP_PROP_FPS)
index_time = 41
index = int(index_time * fps) # frame index
video.set(cv2.CAP_PROP_POS_FRAMES, index)

# Read the frame at the current position
ret, frame = video.read()
save_filename = f'{methodstring}frame_{index}_{read_filename}.png'
if save_filename not in os.listdir('input_videos/frames/'):
    cv2.imwrite(f'input_videos/frames/{save_filename}', frame)
    image = cv2.imread(f'input_videos/frames/{save_filename}')
else:
    image = cv2.imread(f'input_videos/frames/{save_filename}')

# Define the region of interest (ROI) # Based on 0.5 assumption
height, width = image.shape[:2]
square_of_interest = np.array([[(0, height), (0, height*0.0), (width, height*0.0), (width, height)]], dtype=np.int32)

# Create a mask image with the same size as the frame
mask = np.zeros_like(image)

# Fill in the ROI on the mask image with white
cv2.fillPoly(mask, square_of_interest, (255, 255, 255))

# Bitwise AND the mask image with the original image
image = cv2.bitwise_and(image, mask)

# Show the image
cv2.imwrite(f'output_images/{save_filename}_voorblog.png', edges)
cv2.imshow('Image', edges)
cv2.waitKey(0)
cv2.destroyAllWindows()


###
# # Convert to grayscale
gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
#
# # Apply Gaussian blur
blurred = cv2.GaussianBlur(gray, (5, 5), 0)

# Apply Canny Edge Detection
edges = cv2.Canny(blurred   , 50, 150)


# Show the image
cv2.imwrite(f'output_images/{save_filename}_voorblog.png', edges)
cv2.imshow('Image', edges)
cv2.waitKey(0)
cv2.destroyAllWindows()

# Apply Hough Line Transform
lines = cv2.HoughLinesP(edges, 0.8, np.pi/180, 100, minLineLength=5, maxLineGap=100)

# Draw lines on the image
for line in lines:
    x1, y1, x2, y2 = line[0]
    cv2.line(image, (x1, y1), (x2, y2), (0, 255, 0), 2)


def det(a, b):
        return a[0] * b[1] - a[1] * b[0]

def line_intersection(line1, line2):
    # Split the 1D arrays into pairs of coordinates
    line1 = [line1[:2], line1[2:]]
    line2 = [line2[:2], line2[2:]]

    xdiff = (line1[0][0] - line1[1][0], line2[0][0] - line2[1][0])
    ydiff = (line1[0][1] - line1[1][1], line2[0][1] - line2[1][1])

    div = det(xdiff, ydiff)
    if div == 0:
       return None
    d = (det(*line1), det(*line2))
    x = det(d, xdiff) / div
    y = det(d, ydiff) / div
    return x, y

# List to store the keypoints
keypoints = []

# Loop through all pairs of lines
for i in range(len(lines)):
    for j in range(i+1, len(lines)):
        # Check if lines[i][0] and lines[j][0] are arrays
        if isinstance(lines[i][0], np.ndarray) and isinstance(lines[j][0], np.ndarray):
            # Calculate the intersection point of the two lines
            intersection = line_intersection(lines[i][0], lines[j][0])
            if intersection is not None:
                # Add the intersection point to the list of keypoints
                keypoints.append(intersection)

# Draw the keypoints on the image
for keypoint in keypoints:
    cv2.circle(image, tuple(int(i) for i in keypoint), 5, (0, 0, 255), -1)

# Show the image
cv2.imwrite(f'output_images/{save_filename}', image)
cv2.imshow('Image', image)
cv2.waitKey(0)
cv2.destroyAllWindows()
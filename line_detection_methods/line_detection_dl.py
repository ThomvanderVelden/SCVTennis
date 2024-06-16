import cv2
import os
import numpy as np
from court_line_detector import CourtLineDetector6kp


# Read the video
methodstring = 'dl'
read_filename = 'straight_backview_decent.MOV'
video = cv2.VideoCapture(f'input_videos/recorded/{read_filename}')

# Load image
# Set the video capture's position
fps = video.get(cv2.CAP_PROP_FPS)
index_time = 10
index = int(index_time * fps)  # frame index
video.set(cv2.CAP_PROP_POS_FRAMES, index)

# Read the frame at the current position
ret, frame = video.read()
save_filename = f'{methodstring}frame_{index}_{read_filename}.png'
if save_filename not in os.listdir('input_videos/frames/'):
    cv2.imwrite(f'input_videos/frames/{save_filename}', frame)
    image = cv2.imread(f'input_videos/frames/{save_filename}')
else:
    image = cv2.imread(f'input_videos/frames/{save_filename}')

# Define the region of interest (ROI)
height, width = image.shape[:2]
square_of_interest = np.array([[(0, height), (0, height * 0.0), (width, height * 0.0), (width, height)]],
                              dtype=np.int32)

# Create a mask image with the same size as the frame
mask = np.zeros_like(image)

# Fill in the ROI on the mask image with white
cv2.fillPoly(mask, square_of_interest, (255, 255, 255))

# Bitwise AND the mask image with the original image
image = cv2.bitwise_and(image, mask)

###
# Convert to grayscale
# image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

# Court line detector model
court_model_path = "models/keypoints_model_weights_overfit.pth"
court_line_detector = CourtLineDetector6kp(court_model_path)
court_keypoints = court_line_detector.predict(image)

# Filter the keypoints
#court_keypoints = court_line_detector.filter_keypoints(court_keypoints)

# Draw keypoints
output_video_frame = court_line_detector.draw_keypoints_on_frame(image, court_keypoints)
cv2.imwrite(f'output_images/{save_filename}', output_video_frame)
# Show the image
cv2.imshow('Court Line Detection', output_video_frame)
cv2.waitKey(0)
cv2.destroyAllWindows()

import cv2
import numpy as np
import os
from PIL import Image
import torch
from torchvision import transforms, models
import torch.nn as nn
from court_line_detector import CourtLineDetector6kp

def segment_image(image, keypoints, width, height):
    segments = []
    keypoints = keypoints.reshape(-1, 2)
    print(keypoints)
    print(type(keypoints))
    for (x, y) in keypoints:
        x, y = int(x), int(y)
        segment = image[max(0, y-height//2):min(image.shape[0], y+height//2),
                        max(0, x-width//2):min(image.shape[1], x+width//2)]
        segments.append(segment)
    return segments

def apply_classic_method(segment, top_left):
    gray = cv2.cvtColor(segment, cv2.COLOR_BGR2GRAY)
    #
    # # Apply Gaussian blur
    blurred = cv2.GaussianBlur(gray, (5, 5), 0)

    # Apply Canny Edge Detection
    edges = cv2.Canny(blurred, 50, 150)

    # Apply Hough Line Transform
    lines = cv2.HoughLinesP(edges, 0.8, np.pi / 180, 100, minLineLength=5, maxLineGap=100)

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
        for j in range(i + 1, len(lines)):
            # Check if lines[i][0] and lines[j][0] are arrays
            if isinstance(lines[i][0], np.ndarray) and isinstance(lines[j][0], np.ndarray):
                # Calculate the intersection point of the two lines
                intersection = line_intersection(lines[i][0], lines[j][0])
                if intersection is not None:
                    # Add the intersection point to the list of keypoints
                    keypoints.append(intersection)

    # # Draw the keypoints on the image
    # for keypoint in keypoints:
    #     cv2.circle(image, tuple(int(i) for i in keypoint), 5, (0, 0, 255), -1)
    #
    # # Show the image
    # cv2.imwrite(f'output_images/{save_filename}', image)
    # cv2.imshow('Image', image)
    # cv2.waitKey(0)
    # cv2.destroyAllWindows()
    keypoints = [(x + top_left[0], y + top_left[1]) for (x, y) in keypoints]

    return keypoints

def draw_keypoints_on_frame(frame, keypoints, avg_keypoints):
    for (x, y) in keypoints:
        cv2.circle(frame, (int(x), int(y)), 5, (0, 255, 0), -1)
    for (x, y) in avg_keypoints:
        cv2.circle(frame, (int(x), int(y)), 5, (0, 0, 255), -1)
    return frame

# Read the video
methodstring = 'combined'
read_filename = 'straight_backview_decent.MOV'
video = cv2.VideoCapture(f'input_videos/recorded/{read_filename}')
# Set the video capture's position
fps = video.get(cv2.CAP_PROP_FPS)
index_time = 10
index = int(index_time * fps) # frame index
video.set(cv2.CAP_PROP_POS_FRAMES, index)

# Load image
# Read the frame at the current position
ret, frame = video.read()
save_filename = f'{methodstring}frame_{index}_{read_filename}.png'
if save_filename not in os.listdir('input_videos/frames/'):
    cv2.imwrite(f'input_videos/frames/{save_filename}', frame)
    image = cv2.imread(f'input_videos/frames/{save_filename}')
else:
    image = cv2.imread(f'input_videos/frames/{save_filename}')

# Court line detector model
court_model_path = "models/keypoints_model_weights_overfit.pth"
court_line_detector = CourtLineDetector6kp(court_model_path)
court_keypoints = court_line_detector.predict(image)


#sanity check
output_video_frame = court_line_detector.draw_keypoints_on_frame(image, court_keypoints)
cv2.imshow('Court Line Detection', output_video_frame)
cv2.waitKey(0)
cv2.destroyAllWindows()


# Segment the image around the predicted keypoints
segments = segment_image(image, court_keypoints, width=200, height=200)

# Apply the classic method on the segmented image
all_keypoints = []
all_keypoints_avg = []
keypoints_unzip = court_keypoints.reshape(-1, 2)
for (x, y), segment in zip(keypoints_unzip, segments):
    segment_keypoints = apply_classic_method(segment, (x, y))
    if len(segment_keypoints) == 0:
        keypoints_avg = [(x, y)]
    else:
        keypoints_avg = [(sum(x for x, y in segment_keypoints) / len(segment_keypoints),
                          sum(y for x, y in segment_keypoints) / len(segment_keypoints))]
        all_keypoints.extend(segment_keypoints)
        all_keypoints_avg.extend(keypoints_avg)

# Draw keypoints
output_video_frame = draw_keypoints_on_frame(image, all_keypoints, all_keypoints_avg)
cv2.imwrite(f'output_images/{save_filename}', output_video_frame)

# Show the image
cv2.imshow('Court Line Detection', output_video_frame)
cv2.waitKey(0)
cv2.destroyAllWindows()
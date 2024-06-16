import cv2
import numpy as np
from utils import read_video, save_video
from line_detection_classic import line_intersection

# Read the video frames
video_frames = read_video('input_videos/recorded/rotated_backview_decent.MOV')

# Get the frames per second
fps = 60.0 # You may need to adjust this value based on your video

# Calculate the start and end frame indices
start_time = 19.0# sec
end_time = 20.0 # sec
start_index = int(start_time * fps) # start frame index
end_index = int(end_time * fps) # end frame index

# Initialize a list to hold the processed frames
processed_frames = []

# Loop through the frames from the start frame to the end frame
for i in range(start_index, end_index):
    print(f'Processing frame {i}')
    # Get the frame at the current position
    frame = video_frames[i]

    # Process the frame (convert to grayscale, apply Gaussian blur, etc.)
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    blurred = cv2.GaussianBlur(gray, (5, 5), 0)
    edges = cv2.Canny(blurred, 50, 150)
    lines = cv2.HoughLinesP(edges, 1, np.pi/180, 100, minLineLength=10, maxLineGap=250)

    # Draw lines on the frame
    for line in lines:
        x1, y1, x2, y2 = line[0]
        cv2.line(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)

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

    # Draw the keypoints on the frame
    for keypoint in keypoints:
        cv2.circle(frame, tuple(int(i) for i in keypoint), 5, (0, 0, 255), -1)

    # Add the processed frame to the list
    processed_frames.append(frame)

# Save the processed frames as a video
save_video(processed_frames, 'output_videos/processed_video.avi')
import cv2

# Open the video file
video = cv2.VideoCapture("input_videos/new_input.mp4")
print(video)

# Check if video opened successfully
if not video.isOpened():
    print("Error: Could not open video.")
    exit()

# Read and save the first 5 frames
frame_count = 0
while frame_count < 1:
    ret, frame = video.read()
    if not ret:
        print("Error: Could not read frame.")
        break
    frame_filename = f"frame_{frame_count + 1}.jpg"
    cv2.imwrite(f"../output_frames/{frame_filename}", frame)
    print(f"Saved {frame_filename}")
    frame_count += 1

# Release the video capture object
video.release()
print("Video processing completed.")

import cv2
import json
import os
import ctypes

user32 = ctypes.windll.user32
screensize = user32.GetSystemMetrics(0), user32.GetSystemMetrics(1)

# Constants
NUM_KEYPOINTS = 6 # Adjust this based on your requirement

# Load image folder
image_folder_path = 'output_frames/yt_red'
for image in os.listdir(image_folder_path):
    image_path = os.path.join(image_folder_path, image)
    image_name = os.path.splitext(image)[0]
    print(image_name)

    # Read the image
    img = cv2.imread(image_path)
    if img is None:
        print(f"Error: Cannot read image {image_path}")
        continue

    # Resize the image to fit the screen
    screen_width, screen_height = screensize
    img_height, img_width = img.shape[:2]

    # Calculate the scaling factors
    scale_width = screen_width / img_width
    scale_height = screen_height / img_height
    scale = min(scale_width, scale_height)

    # Resize the image
    img = cv2.resize(img, None, fx=scale, fy=scale, interpolation=cv2.INTER_AREA)

    # Initialize list to store keypoints
    keypoints = []

    # Define mouse callback function for drawing keypoints
    def draw_keypoints(event, x, y, flags, param):
        if event == cv2.EVENT_LBUTTONDOWN:
            if len(keypoints) < NUM_KEYPOINTS:
                keypoints.append([x, y])
                cv2.circle(img, (x, y), 10, (0, 255, 255), -1)
                cv2.imshow("Image", img)
            else:
                print(f"Maximum of {NUM_KEYPOINTS} keypoints reached")

    # Display image and set mouse callback
    cv2.imshow("Image", img)
    cv2.setMouseCallback("Image", draw_keypoints)

    # Wait for user to press any key
    cv2.waitKey(0)
    cv2.destroyAllWindows()

    # Ensure there are exactly NUM_KEYPOINTS keypoints
    while len(keypoints) < NUM_KEYPOINTS:
        keypoints.append([-1, -1])  # Placeholder for missing keypoints

    # Prepare annotation data
    annotation = {
        "id": image_name,
        "metric": 0.0,  # Default metric value, update as needed
        "kps": keypoints
    }

    # Load existing annotations
    json_path = os.path.join(image_folder_path, '../annotations_1A.json')
    if os.path.exists(json_path):
        with open(json_path, 'r') as f:
            data = json.load(f)
    else:
        data = []

    # Add new annotation to the list
    data.append(annotation)

    # Save updated annotations
    with open(json_path, 'w') as f:
        json.dump(data, f, indent=4)

    print("Annotation saved successfully for image:", image_name)
print("All annotations saved successfully")
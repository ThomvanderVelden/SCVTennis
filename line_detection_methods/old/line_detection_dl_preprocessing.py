import torch
import torch.nn as nn
import cv2
import os
import numpy as np
from PIL import Image
from torchvision import models
from torchvision import transforms


# Define the preprocess function
def preprocess_image(image):
    # Convert to grayscale
    gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
    # Apply Gaussian blur
    blurred = cv2.GaussianBlur(gray, (5, 5), 0)
    # Apply Canny edge detection
    edges = cv2.Canny(blurred, 50, 150)
    # Create a white mask
    mask = np.ones_like(edges) * 255
    # Apply the mask to the edges
    preprocessed_image = cv2.bitwise_and(mask, mask, mask=edges)
    # Convert back to 3 channels
    preprocessed_image = cv2.cvtColor(preprocessed_image, cv2.COLOR_GRAY2RGB)
    return preprocessed_image


# Prediction function
def predict(model, image, preprocess=True):
    # Preprocess the image if necessary
    if preprocess:
        image = preprocess_image(image)

    # Convert the image to a PIL Image
    image = Image.fromarray(image)

    # Apply the same transformations as during training
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor()
    ])

    image = transform(image).unsqueeze(0)  # Add batch dimension

    # Move the image to the appropriate device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    image = image.to(device)

    # Set the model to evaluation mode
    model.eval()

    with torch.no_grad():
        # Make predictions
        outputs = model(image)

    # Convert outputs to keypoints
    keypoints = outputs.view(-1, 2).cpu().numpy()

    return keypoints


# Load your trained model
class CourtLineDetector6kp:
    def __init__(self, model_path):
        self.model = models.resnet50()
        self.model.fc = nn.Linear(self.model.fc.in_features, 6 * 2)  # For (x, y) coordinates of 6 keypoints
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model.load_state_dict(torch.load(model_path, map_location=self.device))
        self.model.to(self.device)

    def predict(self, image):
        keypoints = predict(self.model, image, preprocess=True)
        return keypoints

    def draw_keypoints_on_frame(self, frame, keypoints):
        for (x, y) in keypoints:
            cv2.circle(frame, (int(x), int(y)), 5, (0, 255, 0), -1)
        return frame


# Read the video
methodstring = 'dl'
read_filename = 'input_video.mp4'
video = cv2.VideoCapture(f'input_videos/recorded/{read_filename}')

# Load image
# Set the video capture's position
fps = video.get(cv2.CAP_PROP_FPS)
index_time = 5
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

# Court line detector model
court_model_path = "models/keypoints_model_weights_preprocessing.pth"
court_line_detector = CourtLineDetector6kp(court_model_path)
court_keypoints = court_line_detector.predict(image)

# Draw keypoints
output_video_frame = court_line_detector.draw_keypoints_on_frame(image, court_keypoints)
cv2.imwrite(f'output_images/{save_filename}', output_video_frame)

# Show the image
cv2.imshow('Court Line Detection', output_video_frame)
cv2.waitKey(0)
cv2.destroyAllWindows()

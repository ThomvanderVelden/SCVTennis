import torch
import torchvision.transforms as transforms
import torchvision.models as models
import cv2


class CourtLineDetector6kp:
    def __init__(self, model_path):
        self.model = models.resnet50(pretrained=False)
        self.model.fc = torch.nn.Linear(self.model.fc.in_features, 6*2)
        self.model.load_state_dict(torch.load(model_path, map_location='cpu'))

        self.transforms = transforms.Compose([
            transforms.ToPILImage(),
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])

    def predict(self, image):
        #only predict on first frame because camera is not moving
        img_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        image_tensor = self.transforms(img_rgb).unsqueeze(0)

        with torch.no_grad():
            outputs = self.model(image_tensor)

        keypoints = outputs.squeeze().cpu().numpy()
        original_h, original_w = image.shape[:2]

        keypoints[1::2] *= original_h / 224.0
        keypoints[::2] *= original_w / 224.0

        return keypoints

    def draw_keypoints(self, image, keypoints):
        for i in range(0, len(keypoints), 2):
            x = int(keypoints[i])
            y = int(keypoints[i+1])
            cv2.putText(image, str(i//2), (x,y-10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2)
            cv2.circle(image, (x,y),8,(0, 255, 255), -1)

        return image

    def draw_keypoints_on_video(self, video_frames, keypoints):
        output_video_frames = []
        for frame in video_frames:
            frame = self.draw_keypoints(frame, keypoints)
            output_video_frames.append(frame)
        return output_video_frames

    def draw_keypoints_on_frame(self, frame, keypoints):
        frame = self.draw_keypoints(frame, keypoints)
        return frame


    def filter_keypoints(self, keypoints):
        # Reshape the keypoints into a 2D array
        reshaped_keypoints = keypoints.reshape(-1, 2)

        # Sort the keypoints based on the y-values
        sorted_keypoints = reshaped_keypoints[reshaped_keypoints[:,1].argsort()]

        # Select the top 7 keypoints
        filtered_keypoints = sorted_keypoints[-7:]

        # Flatten the keypoints back into a 1D array
        filtered_keypoints = filtered_keypoints.flatten()

        return filtered_keypoints
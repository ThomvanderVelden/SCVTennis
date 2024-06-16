from ultralytics import YOLO 
import cv2
import pickle
import pandas as pd

class BallTracker:
    def __init__(self,model_path):
        self.model = YOLO(model_path)

    def interpolate_ball_positions(self, ball_positions):
        ball_positions = [x.get(1,[]) for x in ball_positions]
        # convert the list into pandas dataframe
        df_ball_positions = pd.DataFrame(ball_positions,columns=['x1','y1','x2','y2'])

        # interpolate the missing values
        df_ball_positions = df_ball_positions.interpolate()
        df_ball_positions = df_ball_positions.bfill()

        ball_positions = [{1:x} for x in df_ball_positions.to_numpy().tolist()]

        return ball_positions

    def detect_frames(self,frames, read_from_stub=False, stub_path=None):
        ball_detections = []

        if read_from_stub and stub_path is not None:
            with open(stub_path, 'rb') as f:
                ball_detections = pickle.load(f)
            return ball_detections

        for frame in frames:
            player_dict = self.detect_frame(frame)
            ball_detections.append(player_dict)
        
        if stub_path is not None:
            with open(stub_path, 'wb') as f:
                pickle.dump(ball_detections, f)
        
        return ball_detections

    def detect_frame(self,frame):
        results = self.model.predict(frame,conf=0.25)[0]

        ball_dict = {}
        for box in results.boxes:
            result = box.xyxy.tolist()[0]
            ball_dict[1] = result
        
        return ball_dict

    def draw_bboxes(self, video_frames, player_detections):
        output_video_frames = []
        prev_y2 = None  # Initialize prev_y2 to None to handle the first frame correctly
        descend = True

        for frame, ball_dict in zip(video_frames, player_detections):
            # Draw Bounding Boxes
            bounce = False
            no_detection = True
            current_y2 = None  # Initialize current_y2 for each frame
            # print(prev_y2)
            # print(frame)

            for track_id, bbox in ball_dict.items():
                x1, y1, x2, y2 = bbox
                if y2 == 0:
                    continue  # Skip if the detection is zero

                current_y2 = y2  # Update current_y2 with the current bbox's y2

                if prev_y2 is not None:
                    if y2 > prev_y2 and descend and not bounce:
                        print(f"y2:  {y2} prev_y2:  {prev_y2}")
                        print("bounce_detected")
                        bounce = True
                        descend = False
                    
                if bounce:
                    cv2.putText(frame, "Bounce", (int(bbox[0]), int(bbox[1] - 10)), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 255), 2)
                else:
                    cv2.putText(frame, f"Ball ID: {track_id}", (int(bbox[0]), int(bbox[1] - 10)), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 255), 2)

                cv2.rectangle(frame, (int(x1), int(y1)), (int(x2), int(y2)), (0, 255, 255), 2)
                no_detection = False
                # bounce = False
            

            if not no_detection and current_y2 is not None:
                if prev_y2 is not None and current_y2 < prev_y2:
                    descend = True
                prev_y2 = current_y2  # Update prev_y2 after processing all bboxes in the current frame

            output_video_frames.append(frame)

        return output_video_frames



    
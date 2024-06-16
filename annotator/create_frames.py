import cv2
import os


def save_frames_from_video(video_path, start_time, end_time, interval=20):
    # Create output directory based on video name
    video_name = os.path.splitext(os.path.basename(video_path))[0]
    output_dir = f'output_frames/{video_name}'
    os.makedirs(output_dir, exist_ok=True)

    # Read the video
    video = cv2.VideoCapture(video_path)
    if not video.isOpened():
        print(f"Error: Cannot open video file {video_path}")
        return

    fps = video.get(cv2.CAP_PROP_FPS)
    if fps == 0:
        print("Error: Cannot determine FPS of the video")
        return

    start_frame = int(start_time * fps)
    end_frame = int(end_time * fps)
    interval_frames = int(interval * fps)

    current_frame = start_frame
    while current_frame <= end_frame:
        video.set(cv2.CAP_PROP_POS_FRAMES, current_frame)
        ret, frame = video.read()
        if not ret:
            print(f"Error: Cannot read frame at position {current_frame}")
            break

        save_filename = f'{video_name}_frame_{current_frame}.png'
        save_path = os.path.join(output_dir, save_filename)
        cv2.imwrite(save_path, frame)
        print(f"Saved frame {current_frame} to {save_path}")

        current_frame += interval_frames

    video.release()
    print("Finished saving frames")


# Usage example
video_path = 'input_videos/yt/yt_red.mp4'
start_time = 120  # start time in seconds
end_time = 180 # end time in seconds

save_frames_from_video(video_path, start_time, end_time, interval=10)

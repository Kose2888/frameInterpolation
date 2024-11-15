import cv2
import os

def extract_frames(video_path, output_folder):
    # Create the output folder if it doesn't exist
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)

    # Open the video file
    video_capture = cv2.VideoCapture(video_path)

    # Get the original FPS of the video
    original_fps = video_capture.get(cv2.CAP_PROP_FPS)
    frame_interval = int(original_fps / 10)  # Number of frames to skip to achieve 10 FPS

    # Initialize frame count
    frame_count = 0
    saved_frame_count = 0

    # Loop through the video frames
    while True:
        success, frame = video_capture.read()
        if not success:
            break  # Exit if no more frames

        # Only save every 'frame_interval'-th frame
        if frame_count % frame_interval == 0:
            frame_filename = os.path.join(output_folder, f"frame_{saved_frame_count:05d}.jpg")
            cv2.imwrite(frame_filename, frame)
            saved_frame_count += 1

        frame_count += 1

    # Release the video capture object
    video_capture.release()
    print(f"Extracted {saved_frame_count} frames at 10 FPS and saved to '{output_folder}'")

# Example usage

video_path = "/mnt/c/Users/Ethan/frameInterpVids/Family Guy Season 19 funny scenes compilation. - GXT Plays (360p, h264, youtube).mp4"
output_folder = "frames/familyGuyComp"
extract_frames(video_path, output_folder)

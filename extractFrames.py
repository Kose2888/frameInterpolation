import cv2
import os

def extract_frames(video_path, output_folder):
    # Create the output folder if it doesn't exist
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)

    # Open the video file
    video_capture = cv2.VideoCapture(video_path)

    # Initialize frame count
    frame_count = 0

    # Loop through the video frames
    while True:
        success, frame = video_capture.read()
        if not success:
            break  # Exit if no more frames

        # Save the frame as an image file
        frame_filename = os.path.join(output_folder, f"frame_{frame_count:04d}.jpg")
        cv2.imwrite(frame_filename, frame)
        frame_count += 1

    # Release the video capture object
    video_capture.release()
    print(f"Extracted {frame_count} frames and saved to '{output_folder}'")

# Example usage
video_path = "C:/Users/Ethan/frameInterpVids/demonSlayer.mp4"
output_folder = "frames/test1"
extract_frames(video_path, output_folder)

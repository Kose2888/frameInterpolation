import cv2
import os

frames_folder = "/mnt/c/Users/Ethan/github/kose2888/frameInterpolation/output/familyGuyComp/"
output_video_path = "producedVideos/FIM_familyGuy_1hr/output_video.mp4"
fps = 10 # Matching the extracted frames 

def frames_to_video(frames_folder, output_video_path, fps):
    # Get a sorted list of all frame filenames
    frame_files = sorted(
        [f for f in os.listdir(frames_folder) if f.endswith(('.png', '.jpg', '.jpeg'))]
    )

    if not frame_files:
        print("No frames found in the specified folder.")
        return

    # Get the size of the first frame to set the video size
    first_frame_path = os.path.join(frames_folder, frame_files[0])
    first_frame = cv2.imread(first_frame_path)
    height, width, layers = first_frame.shape
    frame_size = (width, height)

    # Initialize the video writer
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')  # Codec for .mp4 files
    video_writer = cv2.VideoWriter(output_video_path, fourcc, fps, frame_size)

    # Write each frame to the video
    for frame_file in frame_files:
        frame_path = os.path.join(frames_folder, frame_file)
        frame = cv2.imread(frame_path)
        video_writer.write(frame)

    # Release the video writer
    video_writer.release()
    print(f"Video created and saved as '{output_video_path}'")

frames_to_video(frames_folder, output_video_path, fps)
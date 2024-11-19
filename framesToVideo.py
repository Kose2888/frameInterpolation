import cv2
import os

frames_folder = "/mnt/c/Users/Ethan/github/kose2888/frameInterpolation/output/FIM_familyGuy_5000/"
output_video_path = "producedVideos/FIM_familyGuy_5000/output_video.mp4"
fps = 10 # Matching the extracted frames 

def sortFrames(frames_folder):
    files = []
    for dirname, dirnames, filenames in os.walk(frames_folder):
        # print path to all subdirectories first.
        for subdirname in dirnames:
            files.append(os.path.join(dirname, subdirname))

        # print path to all filenames.
        for filename in filenames:
            files.append(os.path.join(dirname, filename))

    return sorted(files)

def frames_to_video(frames_folder, output_video_path, fps):
    # Get a sorted list of all frame filenames
    frame_files = sortFrames(frames_folder)

    if not frame_files:
        print("No frames found in the specified folder.")
        return

    # Get the size of the first frame to set the video size
    first_frame_path = os.path.join(frames_folder, frame_files[0])
    first_frame = cv2.imread(frame_files[0])
    height, width, layers = first_frame.shape
    frame_size = (width, height)

    # Initialize the video writer
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')  # Codec for .mp4 files
    video_writer = cv2.VideoWriter(output_video_path, fourcc, fps, frame_size)

    # Write each frame to the video
    for frame in frame_files:
        f = cv2.imread(frame)
        video_writer.write(f)

    # Release the video writer
    video_writer.release()
    print(f"Video created and saved as '{output_video_path}'")

frames_to_video(frames_folder, output_video_path, fps)
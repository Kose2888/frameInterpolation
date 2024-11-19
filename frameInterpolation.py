import cv2
import numpy as np
import os
from tensorflow.keras.models import load_model

# Parameters
img_size = (96, 160)  # Same size used during training
model_path = "/mnt/c/Users/Ethan/github/kose2888/frameInterpolation/models/FIM_familyGuy_1hr.keras"  # Path to your saved model
data_dir = "/mnt/c/Users/Ethan/github/kose2888/frameInterpolation/frames/familyGuyComp/"  # A dataset that the model was not trained on
output_dir = "/mnt/c/Users/Ethan/github/kose2888/frameInterpolation/output/FIM_familyGuy_1hr/"

# Load the trained model
model = load_model(model_path)

def preprocess_frame(image_path, img_size):
    """Load and preprocess a single frame."""
    frame = cv2.imread(image_path)
    frame = cv2.resize(frame, img_size) / 255.0  # Normalize to [0,1]
    return frame

def sortFrames(data_dir):
    files = []
    for dirname, dirnames, filenames in os.walk(data_dir):
        # print path to all subdirectories first.
        for subdirname in dirnames:
            files.append(os.path.join(dirname, subdirname))

        # print path to all filenames.
        for filename in filenames:
            files.append(os.path.join(dirname, filename))

    return sorted(files)

dataset = sortFrames(data_dir)

for frame in range(len(dataset) - 2):
    #print(dataset[frame])

    # Load a pair of frames (frame1 and frame3)
    frame1 = preprocess_frame(dataset[frame], img_size)
    frame3 = preprocess_frame(dataset[frame + 2], img_size)

    # Stack and expand dimensions for the model
    input_data = np.expand_dims(np.concatenate((frame1, frame3), axis=-1), axis=0)

    # Call the model to predict the intermediate frame
    predicted_frame = model.predict(input_data)[0]

    # Postprocess the predicted frame (convert back to [0,255] range)
    predicted_frame = (predicted_frame * 255).astype(np.uint8)

    predicted_frame_name = f"predicted_frame_{(frame + 1):05d}.jpg"
    # Save the predicted frame to verify the result
    cv2.imwrite(output_dir + predicted_frame_name, predicted_frame)
    print("Intermediate frame saved: ", predicted_frame_name)

print("Frame Interpolation of dataset complete")
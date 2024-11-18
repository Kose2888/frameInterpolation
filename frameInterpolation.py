import cv2
import numpy as np
from tensorflow.keras.models import load_model

# Parameters
img_size = (96, 160)  # Same size used during training
model_path = "/mnt/c/Users/Ethan/github/kose2888/frameInterpolation/models/FIM_familyGuy_1hr.keras"  # Path to your saved model

# Load the trained model
model = load_model(model_path)

def preprocess_frame(image_path, img_size):
    """Load and preprocess a single frame."""
    frame = cv2.imread(image_path)
    frame = cv2.resize(frame, img_size) / 255.0  # Normalize to [0,1]
    return frame

# Load a pair of frames (frame1 and frame3)
frame1 = preprocess_frame("/mnt/c/Users/Ethan/github/kose2888/frameInterpolation/frames/familyGuy2/frame_00050.jpg", img_size)
frame3 = preprocess_frame("/mnt/c/Users/Ethan/github/kose2888/frameInterpolation/frames/familyGuy2/frame_00052.jpg", img_size)

# Stack and expand dimensions for the model
input_data = np.expand_dims(np.concatenate((frame1, frame3), axis=-1), axis=0)

# Call the model to predict the intermediate frame
predicted_frame = model.predict(input_data)[0]

# Postprocess the predicted frame (convert back to [0,255] range)
predicted_frame = (predicted_frame * 255).astype(np.uint8)

# Save the predicted frame to verify the result
cv2.imwrite("/mnt/c/Users/Ethan/github/kose2888/frameInterpolation/output/FIM_familyGuy_1hr/predicted_frame51.jpg", predicted_frame)
print("Intermediate frame saved as 'predicted_frame51.jpg'")
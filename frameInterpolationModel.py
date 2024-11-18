import tensorflow as tf
from tensorflow.keras import layers, models
import numpy as np
import os

# Model definition
def build_model(input_shape=(96, 160, 6)):
    model = models.Sequential([
        # Input scaling
        layers.Rescaling(1./255, input_shape=input_shape),
        
        # Encoder
        layers.Conv2D(16, (3, 3), padding='same', activation=None),
        layers.BatchNormalization(momentum=0.95),
        layers.Activation("relu"),
        layers.Conv2D(16, (3, 3), padding='same', activation=None),
        layers.BatchNormalization(momentum=0.95),
        layers.Activation("relu"),
        layers.MaxPooling2D(),
        layers.Dropout(0.1),
        
        layers.Conv2D(32, (3, 3), padding='same', activation=None),
        layers.BatchNormalization(momentum=0.95),
        layers.Activation("relu"),
        layers.Conv2D(32, (3, 3), padding='same', activation=None),
        layers.BatchNormalization(momentum=0.95),
        layers.Activation("relu"),
        layers.MaxPooling2D(),
        layers.Dropout(0.1),
        
        # Decoder
        layers.Conv2DTranspose(32, (4, 4), strides=2, padding='same', activation=None),
        layers.BatchNormalization(momentum=0.95),
        layers.Activation("relu"),
        layers.Conv2DTranspose(16, (4, 4), strides=2, padding='same', activation=None),
        layers.BatchNormalization(momentum=0.95),
        layers.Activation("relu"),
        
        # Output
        layers.Conv2D(3, (3, 3), padding='same', activation='sigmoid'),
    ])
    return model

# Build and compile model
model = build_model()
model.compile(optimizer='adam', loss='mae')

# Print model summary
model.summary()
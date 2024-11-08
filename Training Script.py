# Data Generator (from earlier)
class DataGenerator(tf.keras.utils.Sequence):
    def __init__(self, data_dir, batch_size=8, img_size=(96, 160), shuffle=True):
        self.data_dir = data_dir
        self.batch_size = batch_size
        self.img_size = img_size
        self.shuffle = shuffle
        self.file_paths = sorted(glob.glob(os.path.join(data_dir, "*.png")))
        self.indexes = np.arange(len(self.file_paths) - 2)
        self.on_epoch_end()
    
    def __len__(self):
        return int(np.floor(len(self.indexes) / self.batch_size))
    
    def __getitem__(self, index):
        batch_indexes = self.indexes[index*self.batch_size:(index+1)*self.batch_size]
        X, y = self.__data_generation(batch_indexes)
        return X, y
    
    def on_epoch_end(self):
        if self.shuffle:
            np.random.shuffle(self.indexes)
    
    def __data_generation(self, batch_indexes):
        X = []
        y = []
        for idx in batch_indexes:
            frame1 = cv2.imread(self.file_paths[idx])
            frame2 = cv2.imread(self.file_paths[idx + 1])
            frame3 = cv2.imread(self.file_paths[idx + 2])
            
            # Resize frames
            frame1 = cv2.resize(frame1, self.img_size)
            frame2 = cv2.resize(frame2, self.img_size)
            frame3 = cv2.resize(frame3, self.img_size)
            
            # Normalize frames to [0,1]
            frame1 = frame1 / 255.0
            frame2 = frame2 / 255.0
            frame3 = frame3 / 255.0
            
            # Stack frames 1 and 3 as input, frame 2 as target
            X.append(np.concatenate((frame1, frame3), axis=-1))
            y.append(frame2)
        
        return np.array(X), np.array(y)

# Training setup
data_dir = "path_to_frames"  # Replace with your dataset path
batch_size = 8
epochs = 10

# Create data generator
train_gen = DataGenerator(data_dir=data_dir, batch_size=batch_size)

# TensorBoard callback
log_dir = "logs/frame_interpolation"
tensorboard_callback = tf.keras.callbacks.TensorBoard(log_dir=log_dir, histogram_freq=1)

# Train model
history = model.fit(train_gen, epochs=epochs, callbacks=[tensorboard_callback])

# Save the model
model.save("Frame_Interpolation_Model")

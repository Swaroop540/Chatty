import os
import numpy as np
import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout, BatchNormalization
from sklearn.model_selection import train_test_split
from tensorflow.keras.callbacks import EarlyStopping
from tensorflow.keras.optimizers import Adam
from PIL import Image
from tqdm import tqdm  

dataset_dir = r'C:\Users\swaro\Desktop\Xray\MURA-v1.1'

img_size = (224, 224)  
batch_size = 16
epochs = 10 

def load_data(dataset_dir, img_size):
    images = []
    labels = []

    print("Loading and preprocessing images")
    for part in tqdm(os.listdir(os.path.join(dataset_dir, 'train')), desc="Body Part", unit="folder"):
        part_dir = os.path.join(dataset_dir, 'train', part)
        for patient in os.listdir(part_dir):
            patient_dir = os.path.join(part_dir, patient)
            for study in os.listdir(patient_dir):
                study_dir = os.path.join(patient_dir, study)
                label = 1 if 'positive' in study.lower() else 0
                for img_file in os.listdir(study_dir):
                    img_path = os.path.join(study_dir, img_file)
                    try:
                        img = Image.open(img_path).convert('L') 
                        img = img.resize(img_size)
                        img_array = np.array(img)
                        images.append(img_array)
                        labels.append(label)
                    except (IOError, OSError):
                        print(f"Skipping invalid image file: {img_path}")
    
    images = np.array(images)
    labels = np.array(labels)
    images = images / 255.0  


    images = images.reshape(-1, img_size[0], img_size[1], 1)
    
    print(f"Loaded {len(images)} images with {len(labels)} labels.")
    return images, labels


X, y = load_data(dataset_dir, img_size)

X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42)

datagen = ImageDataGenerator(
    rotation_range=10,
    zoom_range=0.1,
    width_shift_range=0.1,
    height_shift_range=0.1,
    horizontal_flip=True
)
datagen.fit(X_train)

model = Sequential([# resnet 50 , yolo , recursive network
    Conv2D(32, (3, 3), activation='relu', input_shape=(img_size[0], img_size[1], 1)),
    MaxPooling2D(pool_size=(2, 2)),
    BatchNormalization(),
    
    Conv2D(64, (3, 3), activation='relu'),
    MaxPooling2D(pool_size=(2, 2)),
    BatchNormalization(),
    
    Conv2D(128, (3, 3), activation='relu'),
    MaxPooling2D(pool_size=(2, 2)),
    BatchNormalization(),
    
    Flatten(),
    Dense(256, activation='relu'),
    Dropout(0.5),
    Dense(1, activation='sigmoid')  
])

model.compile(optimizer=Adam(learning_rate=0.001),
              loss='binary_crossentropy',
              metrics=['accuracy'])

class TrainingProgressCallback(tf.keras.callbacks.Callback):
    def on_epoch_begin(self, epoch, logs=None):
        print(f"Starting epoch {epoch + 1}/{epochs}")
    
    def on_epoch_end(self, epoch, logs=None):
        print(f"Finished epoch {epoch + 1}/{epochs}")
        print(f" - Loss: {logs['loss']:.4f} - Accuracy: {logs['accuracy']:.4f} - Val Loss: {logs['val_loss']:.4f} - Val Accuracy: {logs['val_accuracy']:.4f}")

print("Starting model training")
history = model.fit(datagen.flow(X_train, y_train, batch_size=batch_size),
                    validation_data=(X_val, y_val),
                    epochs=epochs,
                    callbacks=[TrainingProgressCallback()])

model.save(r'C:\Users\swaro\Desktop\Xray\mura_fracture_model2.h5')

print("Evaluating model on validation set.")
val_loss, val_acc = model.evaluate(X_val, y_val)
print(f'Validation Accuracy: {val_acc * 100:.2f}%')



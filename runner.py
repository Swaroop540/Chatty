# -*- coding: utf-8 -*-
"""
Created on Thu Oct 10 10:44:46 2024

@author: swaro
"""

import numpy as np
from tensorflow.keras.models import load_model
from PIL import Image

MODEL_PATH = r'C:\Users\swaro\Desktop\Xray\mura_fracture_model.h5'

model = load_model(MODEL_PATH)

def preprocess_image(image_path, target_size=(224, 224)):
    img = Image.open(image_path).convert('L')  
    img = img.resize(target_size)  
    img_array = np.array(img) / 255.0  
    img_array = img_array.reshape(1, target_size[0], target_size[1], 1)  
    return img_array

def predict_fracture(image_path):
    img = preprocess_image(image_path)
    prediction = model.predict(img)
    result = 'Fracture Detected' if prediction[0][0] > 0.5 else 'No Fracture Detected'
    return result

image_path = r"C:\Users\swaro\Desktop\Xray\MURA-v1.1\train\XR_FOREARM\patient00222\study1_negative\image2.png"

result = predict_fracture(image_path)
print(f"Prediction: {result}")

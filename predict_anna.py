import os
import glob
import numpy as np
import tensorflow as tf
from tensorflow.keras.preprocessing import image
from tensorflow.keras.applications.inception_v3 import preprocess_input
from tensorflow.keras.models import load_model

# Load the saved model
model = load_model('crop_disease_identification_model.h5')
base_path = r'C:\Users\Saundarya\Downloads\archive (2)\Train'
dir_list = os.listdir(base_path) 
   
print(dir_list)
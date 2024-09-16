import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.applications import MobileNetV2
from tensorflow.keras.layers import Dense, GlobalAveragePooling2D
from tensorflow.keras.models import Model
import numpy as np
import os

# Define paths
train_dir = "C:\Users\Saundarya\plant-data-set-for-mobilenetmodel\Train"
validation_dir = "C:\Users\Saundarya\plant-data-set-for-mobilenetmodel\valid"

# Image dimensions
img_width, img_height = 224, 224 #resize image 
batch_size = 32 #batch size 32 , might have to make it 8

# Data generators for loading images from directories
train_datagen = ImageDataGenerator(rescale=1./255, horizontal_flip=True, zoom_range=0.2)
validation_datagen = ImageDataGenerator(rescale=1./255)

train_generator = train_datagen.flow_from_directory(
    train_dir,
    target_size=(img_width, img_height),
    batch_size=batch_size,
    class_mode='binary'  # Binary classification: plant or non-plant
    #not category cuz its either this or that 1 or 0
)

validation_generator = validation_datagen.flow_from_directory(
    validation_dir,
    target_size=(img_width, img_height),
    batch_size=batch_size,
    class_mode='binary'
)

# Load pre-trained MobileNetV2 model without the top layer
base_model = MobileNetV2(weights='imagenet', include_top=False, input_shape=(img_width, img_height, 3))

# Freeze base model layers
base_model.trainable = False

# Add custom classification layers on top of MobileNetV2
x = base_model.output
x = GlobalAveragePooling2D()(x)  # Global average pooling layer
x = Dense(1024, activation='relu')(x)  # Fully connected layer
predictions = Dense(1, activation='sigmoid')(x)  # Output layer (1 unit for binary classification)

# Build the model
model = Model(inputs=base_model.input, outputs=predictions)

# Compile the model
model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

# Train the model
model.fit(
    train_generator,
    steps_per_epoch=train_generator.samples // batch_size,
    validation_data=validation_generator,
    validation_steps=validation_generator.samples // batch_size,
    epochs=10  # Adjust based on performance
)

# Save the trained model
model.save('plant_vs_non_plant_classifier.h5')

# Function to predict if the image is a plant or not
def check_if_plant(image_path):
    img = tf.keras.preprocessing.image.load_img(image_path, target_size=(img_width, img_height))
    img_array = tf.keras.preprocessing.image.img_to_array(img)
    img_array = np.expand_dims(img_array, axis=0)  # Create a batch
    img_array /= 255.0  # Normalize the image

    prediction = model.predict(img_array)

    if prediction < 0.5:
        print("Please take a plant photo.")
        return False
    else:
        print("Plant detected. Proceeding to disease prediction.")
        return True

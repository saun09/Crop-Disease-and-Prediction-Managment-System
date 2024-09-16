# Importing libraries
import tensorflow as tf
from tensorflow import keras
from keras import models

from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.layers import Input, Lambda, Dense , Flatten #input layer, Lambda for custom functions, dense for adding all layers, flatten for 1d
from tensorflow.keras.models import Model
from tensorflow.keras.applications.inception_v3 import InceptionV3
from tensorflow.keras.applications.inception_v3 import preprocess_input
from tensorflow.keras.preprocessing import image
from tensorflow.keras.preprocessing.image import ImageDataGenerator,load_img
from tensorflow.keras.models import Sequential
import numpy as np
from glob import glob #Return a list of paths matching a pathname pattern.
from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping


# Confirming if GPU is available
print("Num GPUs Available: ", len(tf.config.experimental.list_physical_devices('GPU')))
print(tf.__version__)

# Setting parameters
train_data_dir = r'C:\Users\Saundarya\Downloads\archive (2)\Train\wheat'

validation_data_dir = r'C:\Users\Saundarya\Downloads\archive (2)\Validation\wheat'
img_width, img_height = 299, 299
batch_size = 8 #8 or 4
epochs = 5

# We can add shear, rotate etc to make the model transform invariant but not needed for now.
train_datagen = ImageDataGenerator(rescale=1./255)
validation_datagen = ImageDataGenerator(rescale=1./255)

# Load images, resize, create batches of 32
train_generator = train_datagen.flow_from_directory(
    train_data_dir,
    target_size=(img_width, img_height),
    batch_size=batch_size,
    class_mode='categorical')

validation_generator = validation_datagen.flow_from_directory(
    validation_data_dir,
    target_size=(img_width, img_height),
    batch_size=batch_size,
    class_mode='categorical')

# Load model
inception = InceptionV3(input_shape=(img_width, img_height, 3), weights='imagenet', include_top=False)

X=Flatten()(inception.output) #flatteing output lasyer
prediction=Dense(42,activation='softmax')(X)
model=Model(inputs=inception.input,outputs=prediction)

# metrics can be set to val_loss, accuracy, etc....use accuracy for now. Can cause overfitting.
model.compile(
  loss='categorical_crossentropy', #loss function
  optimizer='adam', #optimizer
  metrics=['accuracy'] #read about
)

# Define the path where you want to save the model checkpoints
checkpoint_filepath = './checkpoints/wheat_train_disease_inceptionv3.keras'

# Create the ModelCheckpoint callback
checkpoint_callback = ModelCheckpoint(
    filepath=checkpoint_filepath,
    monitor='accuracy',  # Monitor accuracy
    save_best_only=True,  # Save only the best model
    save_weights_only=False,  # Save the entire model (architecture + weights)
    mode='max',  # We want to maximize accuracy
    verbose=1  # Print messages when saving the model
)

# Create the EarlyStopping callback
early_stopping_callback = EarlyStopping(
    monitor='accuracy',  # Monitor accuracy
    patience=5,  # Number of epochs with no improvement after which training will be stopped
    mode='max',  # We want to minimize the validation loss
    verbose=1  # Print messages when stopping training
)

model.fit(train_generator,
          steps_per_epoch=train_generator.samples // batch_size,
          epochs=epochs,
          validation_data=validation_generator,
          validation_steps=validation_generator.samples // batch_size,
          callbacks=[checkpoint_callback, early_stopping_callback])

model.save('wheat_train_disease_identification_model.keras')

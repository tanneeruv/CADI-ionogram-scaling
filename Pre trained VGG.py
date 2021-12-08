"""
Created on Thu Nov 19 14:58:16 2020

@author: TIFR/Balloon Facility/ibmcr
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from glob import glob

training_dir = '/home/ibmcr/TVR/cnnclass/Data/Train'
validation_dir = '/home/ibmcr/TVR/cnnclass/Data/Val'
test_dir = '/home/ibmcr/TVR/cnnclass/Data/Test'

image_files = glob(training_dir + '/*/*.jp*g')
valid_image_files = glob(validation_dir + '/*/*.jp*g')

folders = glob(training_dir + '/*')
num_classes = 8
print ('Total Classes = ' + str(num_classes))


from keras.models import Model
from keras.layers import Flatten, Dense
from keras.applications import VGG16

IMAGE_SIZE = [224, 224]  # You can increase the size for better results. 

# loading the weights of VGG16 without the top layer. These weights are trained on Imagenet dataset.
vgg = VGG16(input_shape = IMAGE_SIZE + [3], weights = 'imagenet', include_top = False)  # input_shape = (224,224,3) as required by VGG

# this will exclude the initial layers from training phase as there are already been trained.
for layer in vgg.layers:
    layer.trainable = False

x = Flatten()(vgg.output)
x = Dense(512, activation = 'relu')(x)   # we can add a new fully connected layer but it will increase the execution time.
x = Dense(512, activation = 'relu')(x)   # we can add a new fully connected layer but it will increase the execution time.
x = Dense(8, activation = 'softmax')(x)  # adding the output layer with softmax function as this is a multi label classification problem.

model = Model(inputs = vgg.input, outputs = x)

model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])


model.summary()



from keras.preprocessing.image import ImageDataGenerator
from keras.applications.vgg16 import preprocess_input

training_datagen = ImageDataGenerator(
                                    rescale=1./255,   # all pixel values will be between 0 an 1
                                    shear_range=0.2, 
                                    zoom_range=0.2,
                                    horizontal_flip=True,
                                    preprocessing_function=preprocess_input)

validation_datagen = ImageDataGenerator(rescale = 1./255, preprocessing_function=preprocess_input)

training_generator = training_datagen.flow_from_directory(training_dir, target_size = IMAGE_SIZE, batch_size = 200, class_mode = 'categorical')
validation_generator = validation_datagen.flow_from_directory(validation_dir, target_size = IMAGE_SIZE, batch_size = 200, class_mode = 'categorical')

# The labels are stored in class_indices in dictionary form. 
# checking the labels
training_generator.class_indices


training_images = 4000
validation_images = 400

history = model.fit_generator(training_generator,
                   steps_per_epoch = 4000,  # this should be equal to total number of images in training set. But to speed up the execution, I am only using 10000 images. Change this for better results. 
                   epochs = 10,  # change this for better results
                   validation_data = validation_generator,
                   validation_steps = 400)  # this should be equal to total number of images in validation set.

model.save("/home/ibmcr/TVR/cnnclass/vgg16_pre.h5")

model.save_weights('/home/ibmcr/TVR/cnnclass/vgg16_pre_wts.h5')


print ('Training Accuracy = ' + str(history.history['accuracy']))
print ('Validation Accuracy = ' + str(history.history['val_accuracy']))

import matplotlib.pyplot as plt

plt.plot(history.history["accuracy"], marker= 'o')
plt.plot(history.history['val_accuracy'], marker= '+')
plt.plot(history.history['loss'],marker= '^')
plt.plot(history.history['val_loss'],marker= '*')

plt.title("model accuracy")
plt.ylabel("Accuracy")
plt.xlabel("Epoch")
plt.legend(["Accuracy","Validation Accuracy","loss","Validation Loss"])
plt.show()

np.save('./home/ibmcr/TVR/cnnclass/npy', history.history)


from keras.preprocessing import image

img = image.load_img("/home/ibmcr/TVR/Paper 3/Images/Strong SF/4A040.png",target_size=(224,224))
#img = image.load_img("/home/ibmcr/TVR/Paper 3/Images/Blank/Mar 172015010.png",target_size=(224,224))
#img = image.load_img("/home/ibmcr/TVR/Paper 3/Images/Es/Apr 132015050.png",target_size=(224,224))
#img = image.load_img("/home/ibmcr/TVR/Paper 3/Images/others/Apr 122015110.png",target_size=(224,224))
img = np.asarray(img)
plt.imshow(img)
img = np.expand_dims(img, axis=0)
from keras.models import load_model
saved_model = load_model("/home/ibmcr/TVR/cnnclass/vgg16_pre.h5")
output = saved_model.predict(img)
print(output)



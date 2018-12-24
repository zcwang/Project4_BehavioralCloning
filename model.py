import os
import csv

# array to append row's elements from the .csv file
samples = []

with open('./data/driving_log.csv') as csvfile:
    reader = csv.reader(csvfile)

     # skip the first row due to headings
    next(reader, None)

    for line in reader:
        samples.append(line)

from sklearn.utils import shuffle
from sklearn.model_selection import train_test_split

train_samples, validation_samples = train_test_split(samples,test_size=0.25)

import cv2
import numpy as np
import sklearn
import matplotlib.pyplot as plt

def generator(samples, batch_size=32):
    num_samples = len(samples)
    while 1:
        shuffle(samples)
        for offset in range(0, num_samples, batch_size):
            batch_samples = samples[offset:offset+batch_size]
            images = []
            angles = []
            for batch_sample in batch_samples:
                    for i in range(0,3):

                        name = './data/IMG/'+batch_sample[i].split('/')[-1]
                        #sconvert to RGB for drive.py
                        center_image = cv2.cvtColor(cv2.imread(name), cv2.COLOR_BGR2RGB)
                        # measurement of steering angle
                        center_angle = float(batch_sample[3])
                        images.append(center_image)

                        # correction left and right images
                        # for left image, +0.2 of steering angle
                        # for right image, -0.2 of steering angle
                        if(i==0):
                            angles.append(center_angle)
                        elif(i==1):
                            angles.append(center_angle+0.2)
                        elif(i==2):
                            angles.append(center_angle-0.2)

                        # Code for Augmentation
                        # flip image and negate the measurement
                        images.append(cv2.flip(center_image,1))
                        if(i==0):
                            angles.append(center_angle*-1)
                        elif(i==1):
                            angles.append((center_angle+0.2)*-1)
                        elif(i==2):
                            angles.append((center_angle-0.2)*-1)
                        # generate 6 images from one

            X_train = np.array(images)
            y_train = np.array(angles)

            yield sklearn.utils.shuffle(X_train, y_train)

train_generator = generator(train_samples, batch_size=64)
validation_generator = generator(validation_samples, batch_size=64)

from keras.models import Sequential
from keras.layers.core import Dense, Flatten, Activation, Dropout
from keras.layers.convolutional import Convolution2D
from keras.layers import Lambda, Cropping2D

model = Sequential()

model.add(Lambda(lambda x: (x / 255.0) - 0.5, input_shape=(160,320,3)))
model.add(Cropping2D(cropping=((70,25),(0,0))))
model.add(Convolution2D(24,5,5,subsample=(2,2)))
model.add(Activation('elu'))
model.add(Convolution2D(36,5,5,subsample=(2,2)))
model.add(Activation('elu'))
model.add(Convolution2D(48,5,5,subsample=(2,2)))
model.add(Activation('elu'))
model.add(Convolution2D(64,3,3))
model.add(Activation('elu'))
model.add(Convolution2D(64,3,3))
model.add(Activation('elu'))
# flatten image
model.add(Flatten())
model.add(Dense(100))
model.add(Activation('elu'))
model.add(Dropout(0.25))
model.add(Dense(50))
model.add(Activation('elu'))
model.add(Dense(10))
model.add(Activation('elu'))
model.add(Dense(1))

model.compile(loss='mse',optimizer='rmsprop')
model.fit_generator(train_generator,
    samples_per_epoch= len(train_samples),
    validation_data=validation_generator,
    nb_val_samples=len(validation_samples),
    nb_epoch=2, verbose=1)
model.save('model.h5')
print('My model.h5 Saved.')

model.summary()




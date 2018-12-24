import csv
import cv2
import numpy as np
from PIL import Image

# create empty lists which will be filled with the network inputs and outputs
car_images = []
steering_angles = []

# load the data from the csv and image files into lists
with open('data/driving_log.csv', 'r') as csvfile:
  reader = csv.reader(csvfile)
  for line in reader:
    # this try block is to handle the exception on the first line of the csv
    # in which the column headers are read, instead of actual data
    try:
      steering_center = float(line[3])
    except ValueError:
      pass
    else:
      correction = 0.2
      steering_left = steering_center + correction
      steering_right = steering_center - correction
  
      current_path = 'data/'
      img_center = np.asarray(Image.open(current_path + line[0]))
      img_left = np.asarray(Image.open(current_path + line[0]))
      img_right = np.asarray(Image.open(current_path + line[0]))
  
      car_images.extend([img_center, img_left, img_right])
      steering_angles.extend([steering_center, steering_left, steering_right])

# convert the image and angle data into numpy arrays
X_train = np.array(car_images)
y_train = np.array(steering_angles)

from keras.models import Sequential
from keras.layers import Flatten, Dense, Lambda, Cropping2D
from keras.layers import Convolution2D
from keras.layers.pooling import MaxPooling2D

# This is a model derived by Nvidia to produce appropriate steering angles given
# image data as input
model = Sequential()
model.add(Lambda(lambda x: x / 255.0 - 0.5, input_shape=(160,320,3))) # normalize data
model.add(Cropping2D(cropping=((70,25),(0,0)))) # get rid of useless image data that distracts from the road
model.add(Convolution2D(24,5,5,subsample=(2,2),activation="relu"))
model.add(Convolution2D(36,5,5,subsample=(2,2),activation="relu"))
model.add(Convolution2D(48,5,5,subsample=(2,2),activation="relu"))
model.add(Convolution2D(64,3,3,activation="relu"))
model.add(Convolution2D(64,3,3,activation="relu"))
model.add(Flatten())
model.add(Dense(100))
model.add(Dense(50))
model.add(Dense(10))
model.add(Dense(1))

model.compile(loss='mse', optimizer='adam')
model.fit(X_train, y_train, validation_split=0.2, shuffle=True, epochs=2)

model.save('model.h5')
exit()
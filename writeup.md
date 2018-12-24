# **Behavioral Cloning** 

## Writeup Template

### You can use this file as a template for your writeup if you want to submit it as a markdown file, but feel free to use some other method and submit a pdf if you prefer.

---

**Behavioral Cloning Project**

The goals / steps of this project are the following:
* Use the simulator to collect data of good driving behavior
* Build, a convolution neural network in Keras that predicts steering angles from images
* Train and validate the model with a training and validation set
* Test that the model successfully drives around track one without leaving the road
* Summarize the results with a written report


[//]: # (Image References)

[image1]: ./writeup_images/center.jpg "Center Image"
[image2]: ./writeup_images/left.jpg "Recovery Image"
[image3]: ./writeup_images/right.jpg "Recovery Image"

## Rubric Points
### Here I will consider the [rubric points](https://review.udacity.com/#!/rubrics/432/view) individually and describe how I addressed each point in my implementation.  

---
### Files Submitted & Code Quality

#### 1. Submission includes all required files and can be used to run the simulator in autonomous mode

My project includes the following files:
* model.py containing the script to create and train the model
* drive.py for driving the car in autonomous mode
* model.h5 containing a trained convolution neural network 
* writeup_report.md or writeup_report.pdf summarizing the results

#### 2. Submission includes functional code
Using the Udacity provided simulator and my drive.py file, the car can be driven autonomously around the track by executing 
```sh
python drive.py model.h5
```

#### 3. Submission code is usable and readable

The model.py file contains the code for training and saving the convolution neural network. The file shows the pipeline I used for training and validating the model, and it contains comments to explain how the code works.

### Model Architecture and Training Strategy

#### 1. An appropriate model architecture has been employed

My model consists of a convolutional neural network with 5x5 and 3x3 filter sizes and depths between 32 and 128 (model.py lines 47-51) 

The model includes RELU layers to introduce nonlinearity (code line 20), and the data is normalized in the model using a Keras lambda layer (model.py lines 47-51). 

#### 2. Attempts to reduce overfitting in the model

The model was trained and validated on different data sets to ensure that the model was not overfitting (code line 59). The model was tested by running it through the simulator and ensuring that the vehicle could stay on the track.

#### 3. Model parameter tuning

The model used an adam optimizer, so the learning rate was not tuned manually (model.py line 58).

#### 4. Appropriate training data

Training data was chosen to keep the vehicle driving on the road. I used a combination of center lane driving, recovering from the left and right sides of the road, and driving around the track clockwise in order to augment the data set.

For details about how I created the training data, see the next section. 

### Model Architecture and Training Strategy

#### 1. Solution Design Approach

The overall strategy for deriving a model architecture was to have the ability to take in a large data set of good driving, but not overfit that data, so that the model would be able to react to different real-time images when driving autonomously in the simulation.

My first step was to use a convolution neural network model similar to the one derived by Nvidia. I thought this model might be appropriate because it has a good number of layers and activations.

In order to gauge how well the model was working, I split my image and steering angle data into a training and validation set (80% training, 20% validation). I found that my first model had a low mean squared error on the training set but a high mean squared error on the validation set. This implied that the model was overfitting. 

To combat the overfitting, I modified the model so that it had fewer epochs. This kept the validation loss low.

Then I made sure I had ample data for both good driving and recovery driving. I got recovery driving by getting recordings of the car going from the outside of the road to the center. I also created path-recovery data from the side cameras by applying a correction to the steering angle based on which side the camera data was taken from.

The final step was to run the simulator to see how well the car was driving around track one. There were a few spots where the vehicle fell off the track, especially right after the bridge. In order to fix this, I recorded a few more sample of me making that turn smoothly, and also made sure that I had data to recover if the car did not make the turn sharply enough (which was the issue I initially faced).

At the end of the process, the vehicle is able to drive autonomously around the track without leaving the road.

#### 2. Final Model Architecture

The final model architecture (model.py lines 18-24) consisted of a convolution neural network with the following layers and layer sizes:
The first layer normalizes the data using a Lambda function.
The second layer crops out the regions of the image that do not capture the road. This simplifies the images in order to not confuse the network.
The next 5 layers are 2D convoltuions, with increasing numbers of filters each layer (except the last 2, which are both 64 filters). The first 3 layers have 5x5 convolutions, and the last 2 layers have 3x3 convolutions. All 5 layers utilize the RELU activation function.


#### 3. Creation of the Training Set & Training Process

To capture good driving behavior, I first recorded two laps on track one using center lane driving. Here is an example image of center lane driving:

![alt text][image1]

I then recorded the vehicle recovering from the left side and right sides of the road back to center so that the vehicle would learn to get itself back to the center of the road (which is the type of images that the network knows how to handle the best) if the network accidentally steered the car away from center at some point. The following images were used to correlate stronger steering angles back to the center of the road when the car was off to either the right or the left.

![alt text][image2]
![alt text][image3]

To augment the data sat, I also initially flipped images and angles in order to mix up the data set so that the model would not overfit. I ended up not doing this approach in the final design, however, because I found that driving around the track clockwise provided a similar effect, and my model was driving the car perfectly around the track, so extra augmentation was not needed.


I finally randomly shuffled the data set and put 20% of the data into a validation set. 

I used this training data for training the model. The validation set helped determine if the model was over or under fitting. The ideal number of epochs was 2 as evidenced by a validation loss that stayed low. I used an adam optimizer so that manually training the learning rate wasn't necessary.

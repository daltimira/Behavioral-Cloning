# **Behavioral Cloning**

The goals / steps of this project are the following:
* Use the simulator to collect data of good driving behavior
* Build, a convolution neural network in Keras that predicts steering angles from images
* Train and validate the model with a training and validation set
* Test that the model successfully drives around track one without leaving the road
* Summarize the results with a written report


[//]: # (Image References)

[image1]: ./pictures/meanSquaredErrorLoss_v0_epoch5.png "Mean Squared Error Loss v0"
[image2]: ./pictures/center.jpg "Center camera"
[image3]: ./pictures/left.jpg "Left camera"
[image4]: ./pictures/right.jpg "Right camera"



---

The project includes the following files:
* model.py containing the script to create and train the model
* drive.py for driving the car in autonomous mode
* model.h5 containing a trained convolution neural network
* writeup_report.md summarizing the results

The model.py file contains the code for training and saving the convolution neural network. The file shows the pipeline I use for training and validating the model, and it contains comments to explain how the code works.

### Model Architecture and Training Strategy

I implement an arquitecture published by the autonomous vehicle team at NVIDIA (https://devblogs.nvidia.com/deep-learning-self-driving-cars/) in order to predict steering angle based on camera images placed on the car in a simulation environment. The training consists in taking images of the cameras while driving the car manually and taking measurments of the steering angles. The aim of the project is to train the network so that the car can run autonomously by predicting the correct steeering angles based on new images of the car while driving.

The network consists in:
- Normalization layer.
- Five convolution layers.
- Four fully connected layers.

After the normalization I also mean centering the image by substracting the value of 0.5.

The model includes RELU layers after each convolution to introduce nonlinearlity.

Also, I crop the image before the convolution layers in order to remove background scene and the front part of the car, which could affect negatively  the training of the model as it might be undesirable that the model pick some features from the background scene or parts of the car for the training (which is not useful for driving). The crop consists in removing the top 55 pixels and the bottom 23 pixels of the image.

The model in the submission has been obtained by training and validating by using images of the car while driving on the track 1. All the data set used consist in 8036 samples, where each sample has three images: left image, center image and right image (these are images taken at different locations of the car).

Example of an image taken from the center of the car:

![alt text][image2]

Example of an image taken from the left side of the car:

![alt text][image3]

Example of an image taken from the right side of the car:

![alt text][image4]

For each of the sample data I have a steering angle measurements. I assign this steering angle for the center camera, but I apply a correction factor of 0.2 for the right and left cameras in order to encourage the car to stay at the center.

In addition of using the right, center and left images, I also augment each of these image by flipping it horizontally and changing the sign of the steering measurement. So, in total, the number of data samples have been used for training the model is 8036x3x3 = 72324 samples.

From the total of these image, I split the training and validation samples (80% and 20% of the images respectively - line 83 in the model.py).

I also should note that, owing to the amount of images, in order to prevent storing the images in memory, I use generators. So I have a generator for the training samples (line 85) and a generator for the validation samples (line 86).

While training, I also used model checkpoint (line 114) in order to save the best model based on the validation loss, and an early stopper (line 115).

I train the network using 5 epochs, and I use an adam optimizer, so the learning rate is not tuned manually. The results obtained are the following ones (quite low mean squared error loss for the training and validation sets):

![alt text][image1]

The results obtained are quite good, and the testing on track 1 can be seen in the video: "./videos/video_track1.mp4" and on track 2 "./videos/video_track2.mp4)". For the first track, the car run correctly during all the track but for the track 2 the car easily fall outside of the track. This can be somehow expected as the model only have seen images from track one and probably cannot generalize enough to successfuly run on track 2.

In order to generalise more the model, different actions can be done: (1) train on track 2 (or another track) or (2) train the model while driving in reverse order.

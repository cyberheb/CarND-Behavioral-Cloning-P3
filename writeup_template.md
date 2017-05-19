#**Behavioral Cloning** 

---

**Behavioral Cloning Project**

The goals / steps of this project are the following:

* Use the simulator to collect data of good driving behavior
* Build, a convolution neural network in Keras that predicts steering angles from images
* Train and validate the model with a training and validation set
* Test that the model successfully drives around track one without leaving the road
* Summarize the results with a written report


[//]: # (Image References)

[image1]: ./examples/model_visualization.png "Model Visualization"
[image2]: ./examples/center_lane.jpg "Centar Lane Driving"

## Rubric Points
###Here I will consider the [rubric points](https://review.udacity.com/#!/rubrics/432/view) individually and describe how I addressed each point in my implementation.  

---
###Files Submitted & Code Quality

####1. Submission includes all required files and can be used to run the simulator in autonomous mode

My project includes the following files:
* model.py containing the script to create and train the model
* drive.py for driving the car in autonomous mode
* model.h5 containing a trained convolution neural network 
* writeup_report.md or writeup_report.pdf summarizing the results

####2. Submission includes functional code
Using the Udacity provided simulator and my drive.py file, the car can be driven autonomously around the track by executing 
```sh
python drive.py model.h5
```

####3. Submission code is usable and readable

The model.py file contains the code for training and saving the convolution neural network. The file shows the pipeline I used for training and validating the model, and it contains comments to explain how the code works.

###Model Architecture and Training Strategy

####1. An appropriate model architecture has been employed

My model consists of a convolution neural network with 3x3 filter and 5x5 filter sizes with depths between 24 and 64. The model adapted from NVIDIA architecture (line 90 - 106)

The model includes RELU layers to introduce nonlinearity, and the data is normalized in the model using a Keras lambda layer. 

####2. Attempts to reduce overfitting in the model

The model contains augmented data to reduce overfitting (line 54-60)

The model was trained and validated on different data sets to ensure that the model was not overfitting. I used a combination between data provided by Udacity, and also data taken from my restore driving on the track-1 + track-2. 

The model was tested by running it through the simulator and ensuring that the vehicle could stay on the track.

####3. Model parameter tuning

The model used an adam optimizer, so the learning rate was not tuned manually. 

####4. Appropriate training data

Training data was chosen to keep the vehicle driving on the road. I used a combination of center lane driving, recovering from the left and right sides of the road. 

For details about how I created the training data, see the next section. 

###Model Architecture and Training Strategy

####1. Solution Design Approach

The overall strategy for deriving a model architecture was to use NVIDIA architecture that was already proven in real self driving car scenario. 

My first step was to use a convolution neural network model similar to the LeNet. I thought this model might be appropriate because it will perform high level feature extraction from the training data that contain images.

In order to gauge how well the model was working, I split my image and steering angle data into a training and validation set. I found that my first model had a low mean squared error on the training set but a high mean squared error on the validation set. This implied that the model was overfitting. I found also the model doesn't perform well whenever car autonomously drive on the road where side lane lines missing.

To combat the overfitting, I modified the model so that it will perform better feature extraction including the road where side lines missing. I tried to use model from NVIDIA and it giving good result. However, for some reason if I perform augmented on track-2 data it is giving bad result for autonomous on track-1. In order to mitigate, I removed augmented data for track-2 (that is saying that no flipping images for track-2), and append them to augmented data of track-1 and udacity's data.

Then I also put cropping on the area of images to make training fater.

The final step was to run the simulator to see how well the car was driving around track one. The vehicle is able to drive autonomously around the track without leaving the road.

####2. Final Model Architecture

The final model architecture consisted of a convolution neural network with the following layers and layer sizes: 5 convolutional layers with RELU activation, and 4 flatten layer.

Here is a visualization of the architecture (note: visualizing the architecture is optional according to the project rubric)

![alt text][image1]

####3. Creation of the Training Set & Training Process

To capture good driving behavior, I first recorded two laps on track one using center lane driving. Here is an example image of center lane driving:

![alt text][image2]

I then recorded the vehicle recovering from the left side and right sides of the road back to center so that the vehicle would learn to move the streeing into correct position whenever it is about to cross the side lines (going outside the road). 

Then I repeated this process on track two in order to get more data points.

To augment the data sat, I also flipped images and angles thinking that this would encrich the training data since the training lane was bias to the left. 

After the collection process, I had 14,938 number of data points. I then preprocessed this data by augmenting the data for track-1 and leaving the data without augmentation for track-2. As a result, total samples data become almost 30k.

I finally randomly shuffled the data set and put 20% of the data into a validation set, so the training data samples is 20,913 and validation data samples is 8,963.

I used this training data for training the model. The validation set helped determine if the model was over or under fitting. The ideal number of epochs was 10 as evidenced by low validation loss getting stagnant within 10 epoch. I used an adam optimizer so that manually training the learning rate wasn't necessary.

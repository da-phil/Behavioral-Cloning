# Autonomously driving car in simulator by using Behavioral Cloning


The goals / steps of this project are the following:
* Use the simulator to collect data of good driving behavior
* Build, a convolution neural network in Keras that predicts steering angles from images
* Train and validate the model with a training and validation set
* Test that the model successfully drives around track one without leaving the road
* Summarize the results with a written report


[//]: # (Image References)
[example_image]: ./examples/placeholder_small.png "Normal Image"
[steering_signal]: ./examples/steering_signal_input.png "Steering signal input device comparison"
[using_multiple_cameras]: ./examples/carnd-using-multiple-cameras.png "Using all three cameras for steering prediction"

### Files explained

The project includes the following files:
* `model.py` containing the script to create and train the model
* `drive.py` for driving the car in autonomous mode
* `*model.h5` files containing trained convolution neural networks 
* `writeup_report.md` summarizing the results

Using the Udacity provided simulator, the provided model *commaAI* and my `drive.py` file, the car can be driven autonomously around the track by executing 
```sh
./drive.py commaAI-model.h5
```

### Model Architecture and Training Strategy

#### Model 

I tried out two models, the one which was published by comma.ai in their [paper "Learning a Driving Simulator" in 2016](https://arxiv.org/abs/1608.01230). The model can also be found on their github [research repsitory](https://github.com/commaai/research/blob/master/train_steering_model.py).

The second model I tried out was the one proposed in NVIDIAS [paper "End to End Learning for Self-Driving Cars" (2016)](http://images.nvidia.com/content/tegra/automotive/images/2016/solutions/pdf/end-to-end-dl-using-px.pdf).

Both models include ELU layers to introduce nonlinearity and the data is normalized in the models using a Keras lambda layer. 
Additionally I used dropout layer to prevent overfitting in both networks.

**Comma.AI Model**

| Layer         		     |     Description                         | 
|:---------------------:|:---------------------------------------------:| 
| input                 | input: 66x200x3 RGB image                     |
| Lambda normalization  | Normalizes RGB values to [-1, +1] value range   |
| Convolution           | kernel: 3x3, stride: 4x4, output: 16x50x16, activation: ELU    |
| Convolution           | kernel: 3x3, stride: 2x2, output: 7x24x32, activation: ELU     |
| Dropout               | keep_prob: 0.2                                |
| Convolution           | kernel: 3x3, stride: 2x2, output: 3x11x64, activation: ELU     |
| Flatten               | output: 2112                                  |
| Fully connected       | input: 2048, output: 512, activation: ELU     |
| Dropout               | keep_prob: 0.2                                |
| Fully connected out   | input: 512, output: 1                        | 


The model was trained and validated on different data sets to ensure that the model was not overfitting (code line 10-16). The model was tested by running it through the simulator and ensuring that the vehicle could stay on the track.

#### Model parameter tuning

I used an adam optimizer, with the learning rate set to `0.001` and also setting a learning rate decay of `0.0001` to make sure that I get a smoothly decreasing loss across training for 10 episodes. In the beginning I worked with only 5 episodes and a fixed default learning rate, but sometimes I could already see an oscilating loss value after 3 or 4 episodes.
In order to stop training after 3 episodes where the loss hasn't decreased I also added a `EarlyStopping` callback with a patience of `3` to the `fit_generator` model call in keras.


#### Training data

During training in the simulator I didn't so much attention to staying in the middle of the road, I just try to not get too close to the border of the street. After starting out with my keyboard as input device I noticed that I needed far more training data than with a mouse as input device. As can be seen in the following plot, where I drove the first couple of turns on track 2 with the keyboard first and then with the mouse, the mouse input loooks way smoother and more reasonable for training.

![steering_signal]

In order to stay in the lane I used the technique proposed in the NVIDIA paper to use all three camera images as shown in this image:

[using_multiple_cameras]


I tried recovering from the left and right sides of the road only a couple of times but figured that I'd need so many maneuvers for training that it's not worth it, using all three camera images already did the trick here.

For details about how I created the training data, see the next section. 

###Model Architecture and Training Strategy

####1. Solution Design Approach

The overall strategy for deriving a model architecture was to ...

My first step was to use a convolution neural network model similar to the ... I thought this model might be appropriate because ...

In order to gauge how well the model was working, I split my image and steering angle data into a training and validation set. I found that my first model had a low mean squared error on the training set but a high mean squared error on the validation set. This implied that the model was overfitting. 

To combat the overfitting, I modified the model so that ...


The final step was to run the simulator to see how well the car was driving around track one. There were a few spots where the vehicle fell off the track... to improve the driving behavior in these cases, I ....

At the end of the process, the vehicle is able to drive autonomously around the track without leaving the road.

####2. Final Model Architecture

The final model architecture (model.py lines 18-24) consisted of a convolution neural network with the following layers and layer sizes ...


####3. Creation of the Training Set & Training Process

To capture good driving behavior, I first recorded two laps on track one using center lane driving. Here is an example image of center lane driving:

I then recorded the vehicle recovering from the left side and right sides of the road back to center so that the vehicle would learn to .... These images show what a recovery looks like starting from ... :

Then I repeated this process on track two in order to get more data points.

To augment the data sat, I also flipped images and angles thinking that this would ... For example, here is an image that has then been flipped:

After the collection process, I had X number of data points. I then preprocessed this data by ...

I finally randomly shuffled the data set and put Y% of the data into a validation set. 

I used this training data for training the model. The validation set helped determine if the model was over or under fitting. The ideal number of epochs was Z as evidenced by ... I used an adam optimizer so that manually training the learning rate wasn't necessary.

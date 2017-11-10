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
[track1_gif]: ./examples/track1.gif
[track1_gif]: ./examples/track2.gif

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
Additionally I used dropout layers to prevent overfitting in both networks.

**comma.AI model**

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

**nvidia model**
tbd

The models were trained and validated on different data sets to ensure that the model was not overfitting.
model was tested by running it through the simulator and ensuring that the vehicle could stay on the track.

#### Model parameter tuning

I used an adam optimizer, with the learning rate set to `0.001` and also setting a learning rate decay of `0.0001` to make sure that I get a smoothly decreasing loss across training for 10 episodes. In the beginning I worked with only 5 episodes and a fixed default learning rate, but sometimes I could already see an oscilating loss value after 3 or 4 episodes.
In order to stop training after 3 episodes where the loss hasn't decreased I also added a `EarlyStopping` callback with a patience of `3` to the `fit_generator` model call in keras.


#### Training data

During training in the simulator I didn't put so much attention to staying in the middle of the road, I just try to not get too close to the border of the street. After starting out with my keyboard as input device I noticed that I needed far more training data than with a mouse as input device. As can be seen in the following plot, where I drove the first couple of turns on track 2 with the keyboard first and then with the mouse, the mouse input loooks way smoother and more reasonable for training.

![steering_signal]

I assume that - because the signal is not so peaky and more stable across frames - learning the end-to-end relation between pixels in the current frame to the steering angle is more robust, meaning that also the predictions get less jumpy.

In order to stay in the lane I used the technique proposed in the NVIDIA paper to use all three camera images as shown in this image:

![using_multiple_cameras]

Using a correction angle of `0.25` worked well for the recorded training sets.

I tried recovering from the left and right sides of the road only a couple of times for track 2, where the car would start steering aggressively to the left once started. Training with only 3 or 4 recovering maneuvers already helped of getting rid of this problem.

For mitigating a steering angle bias, as can be seen on track 1 where the car is mostly driving to the left I sometimes drove the tracks in the opposite direction.
Also randomly augmenting the data set by flipping images and angles helped with this issue.

I finally randomly shuffled the data set and put 20% of the data into a validation set. 

I used this training data for training the model. The validation set helped determine if the model was over or under fitting. The ideal number of epochs was Z as evidenced by


In order to gauge how well the model was working, I split my image and steering angle data into a training and validation set. I found that my first model had a low mean squared error on the training set but a high mean squared error on the validation set. 

Both models worked quite well, however the commaAI model seemed to need more training data to be as robust as the nvidia model, this might be due to the fact that it has approx. 1.1 million parameters whereas the nvidia model only has 250 thousand trainable parameters. This implied that the commaAI model was slightly overfitting. 

At the end of the process, the vehicle is able to drive autonomously around both tracks without leaving the road.
Driving track 1 autonomously with the nvidia model was already possible after training only with the Udacity supplied dataset, driving track 2 autonomously needed a lot of additional training data, approx. 34k (track1: 11k, track2: 32k) samples.


![track1_gif]

![track2_gif]

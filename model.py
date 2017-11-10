#!/usr/bin/env python

import os, sys, signal
import time
import csv
import numpy as np
from PIL import Image
import argparse
import matplotlib.pyplot as plt
import cv2

import tensorflow as tf
import keras
from keras.preprocessing.image import ImageDataGenerator
from keras.models import Sequential, load_model
from keras.layers import Dense, Dropout, Flatten
from keras.layers import Conv2D, MaxPooling2D, Lambda, Cropping2D
import sklearn
from sklearn.model_selection import train_test_split
from keras.callbacks import EarlyStopping, ModelCheckpoint


IMAGE_HEIGHT, IMAGE_WIDTH = 66, 200
INPUT_SHAPE = (IMAGE_HEIGHT, IMAGE_WIDTH, 3)

def commaAI_model(keep_prob=0.5):
    model = Sequential()
    model.add(Lambda(lambda x: (x / 255.0) - 0.5, input_shape=INPUT_SHAPE))
    model.add(Conv2D(16, kernel_size=(3, 3), strides=(4,4), activation="elu"))
    model.add(Conv2D(32, kernel_size=(3, 3), strides=(2,2), activation="elu"))
    model.add((Dropout(keep_prob)))
    model.add(Conv2D(64, kernel_size=(3, 3), strides=(2,2), activation="elu"))
    model.add(Flatten())
    model.add(Dense(512, activation="elu"))
    model.add(Dropout(keep_prob))
    model.add(Dense(1))
    return model
    

def nvidia_model(keep_prob=0.5):
    model = Sequential()
    model.add(Lambda(lambda x: (x / 255.0) - 0.5, input_shape=INPUT_SHAPE))
    model.add(Conv2D(24, kernel_size=(5, 5), use_bias=True, strides=(2,2), activation="elu"))
    model.add(Conv2D(36, kernel_size=(5, 5), strides=(2,2), activation="elu"))
    model.add(Conv2D(48, kernel_size=(5, 5), strides=(2,2), activation="elu"))
    model.add((Dropout(keep_prob)))
    model.add(Conv2D(64, kernel_size=(3, 3), strides=(1,1), activation="elu"))
    model.add(Conv2D(64, kernel_size=(3, 3), strides=(1,1), activation="elu"))
    model.add((Dropout(keep_prob)))
    model.add(Flatten())
    model.add(Dropout(keep_prob))
    model.add(Dense(100, activation="elu"))
    model.add(Dense(50, activation="elu"))
    model.add(Dense(10, activation="elu"))
    model.add(Dense(1))
    return model


def random_flip(image, steering_angle):
    """
    Randomly flip the image left <-> right, and adjust the steering angle.
    """
    if np.random.rand() < 0.5:
        image = cv2.flip(image, 1)
        steering_angle = -steering_angle
    return image, steering_angle


def preprocess_image(img):
    img = img.astype("float32")
    
    # Crop the image (removing the sky at the top and the car front at the bottom)
    img =  img [40:-25, :, :]

    # Resize image to fit to model architecture
    img = cv2.resize(img, (IMAGE_WIDTH, IMAGE_HEIGHT), cv2.INTER_AREA)
    return img


def augment_data(imgs, steering_angles):
    assert(len(imgs) == len(steering_angles))
    
    for i, (img, steering_angle) in enumerate(zip(imgs, steering_angles)):
        imgs[i], steering_angles[i] = random_flip(img, steering_angle)
        # TODO: maybe more augmentations    

    return imgs, steering_angles


def generator(samples, batch_size=32, steering_correction=0.2, is_training=False):
    num_samples = len(samples)
    while 1: # Loop forever so the generator never terminates
        sklearn.utils.shuffle(samples)
        for offset in range(0, num_samples, batch_size//3):
            batch_samples = samples[offset:offset+batch_size//3]
            images = []
            angles = []
            for batch_sample in batch_samples:
                current_path_center = training_set + "/IMG/" + batch_sample[0].split("/")[-1]
                current_path_left   = training_set + "/IMG/" + batch_sample[1].split("/")[-1]
                current_path_right  = training_set + "/IMG/" + batch_sample[2].split("/")[-1]

                center_angle = float(batch_sample[3])
                # create adjusted steering measurements for the side camera images
                left_angle  = center_angle + steering_correction
                right_angle = center_angle - steering_correction

                # read in images from center, left and right cameras
                img_center = preprocess_image(np.asarray(Image.open(current_path_center)))
                img_left   = preprocess_image(np.asarray(Image.open(current_path_left)))
                img_right  = preprocess_image(np.asarray(Image.open(current_path_right)))
                
                images.extend([img_left, img_center, img_right])
                angles.extend([left_angle, center_angle, right_angle])

                if is_training:
                    images, angles = augment_data(images, angles)

            X_train = np.array(images)
            y_train = np.array(angles)
            yield sklearn.utils.shuffle(X_train, y_train)




if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Remote Driving")
    parser.add_argument("trainingset",     type=int,    nargs="?",
                        default=0,         help="Choose trainingset")
    parser.add_argument("--steering_corr", type=float,  nargs="?",
                        default="0.2",     help="Steering correction")
    parser.add_argument("--epochs",        type=int,    nargs="?",
                        default="20",      help="Number of epochs for training")
    parser.add_argument("--batchsize",     type=int,    nargs="?",
                        default="256",     help="Batchsize for training")
    parser.add_argument("--model",         type=str,    nargs="?",
                        default="commaAI", help="Choose between models: commaAI, nvidia")
    parser.add_argument("--dropout",       type=float,  nargs="?",
                        default=0.2,       help="Dropout value for regulization during training")
    args = parser.parse_args()
    #print parameters
    print("=" * 40)
    print('Parameters')
    print("=" * 40)
    for key, value in vars(args).items():
        print('{:<20} = {}'.format(key, value))
    print("=" * 40)

    # Parse csv file
    training_set  = "../run%02d" % args.trainingset
    training_data = {"center_imgs": [], "left_imgs": [], "right_imgs": [],
                    "steering_angle_center": [], "steering_angle_left": [], "steering_angle_right": [],
                    "throttle": [], "brake": [], "speed": []}
    samples = []
    with open(training_set+"/driving_log.csv") as logfile:
        reader = csv.reader(logfile, delimiter=",")
        header = next(reader) # it's safe to skip first row in case it contains headings
        
        for line in reader:
            samples.append(line)
            # images
            training_data["center_imgs"].append(training_set + "/IMG/" + line[0].split("/")[-1])
            training_data["left_imgs"].append(training_set + "/IMG/" + line[1].split("/")[-1])
            training_data["right_imgs"].append(training_set + "/IMG/" + line[2].split("/")[-1])

            # car control signals
            # create adjusted steering measurements for the side camera images
            training_data["steering_angle_center"].append(float(line[3]))
            training_data["steering_angle_left"].append(float(line[3]) + args.steering_corr)
            training_data["steering_angle_right"].append(float(line[3]) - args.steering_corr)
            training_data["throttle"].append(line[4])
            training_data["brake"].append(line[5])
            training_data["speed"].append(line[6])


    weights_file = args.model + "-weights.h5"
    model_file   = args.model + "-model.h5"
    
    if os.path.isfile(model_file):
        model = load_model(model_file)
        print("Loaded pretrained %s keras model!" % args.model)
    else:
        if args.model == "commaAI":
            model = commaAI_model(args.dropout)
        elif args.model == "nvidia":
            model = nvidia_model(args.dropout)
        else:
            print("Model '%s' not known!" % args.model)
            sys.exit(-1)

        try:
            model.load_weights(weights_file)
            print("Loaded weights for %s keras model!" % args.model)
        except:
            print("Could not load weights for %s keras model from file %s!\n"
                  "Starting training from scratch!" % (args.model, weights_file))

    # define callbacks for the checkpoint saving and early stopping
    callbacks_list = [
        EarlyStopping(monitor="loss", patience=3, verbose=1),
        ModelCheckpoint(weights_file, monitor="loss", save_best_only=True, mode="auto", save_weights_only=True, verbose=1)
    ]

    sklearn.utils.shuffle(samples)
    train_samples, validation_samples = train_test_split(samples, test_size=0.2)
    nr_train_samples = len(train_samples)
    nr_valid_samples = len(validation_samples)
    print("Samples in training set:   %d" % nr_train_samples)
    print("Samples in validation set: %d" % nr_valid_samples)
    
    # check if batchsize is not too large for sample size (validation set might be too small)
    changed_batchsize = False
    while nr_valid_samples // args.batchsize == 0:
        args.batchsize = args.batchsize // 2
        changed_batchsize = True

    if changed_batchsize:
        print("Batchsize too high for dataset, changed batchsize to %d" % args.batchsize)

    sample_selection = range(nr_train_samples)
    
    # compile and train the model using the generator function
    train_generator = generator(train_samples, batch_size=args.batchsize,
                                steering_correction=args.steering_corr, is_training=True)
    validation_generator = generator(validation_samples, batch_size=args.batchsize,
                                steering_correction=args.steering_corr, is_training=False)

    #optimizer_choice = keras.optimizers.Adadelta(lr=0.5, decay=0.2)
    optimizer_choice = keras.optimizers.Adam(lr=0.001, decay=0.0001)
    # optimizer_choice = keras.optimizers.Adamax(lr=0.002,decay=0.1)

    model.compile(loss="mse", optimizer=optimizer_choice)    
    model.summary()
    model_history = model.fit_generator(train_generator, steps_per_epoch=nr_train_samples // args.batchsize, epochs=args.epochs,
                                         validation_data=validation_generator, validation_steps=nr_valid_samples // args.batchsize,
                                         callbacks=callbacks_list, verbose=1)

    ### plot the training and validation loss for each epoch
    fig = plt.figure()
    plt.plot(model_history.history['loss'])
    plt.plot(model_history.history['val_loss'])
    plt.title('model mean squared error loss')
    plt.ylabel('mean squared error loss')
    plt.xlabel('epoch')
    plt.legend(['training set', 'validation set'], loc='upper right')
    plt.show()

    model.save_weights(weights_file)
    model.save(model_file)

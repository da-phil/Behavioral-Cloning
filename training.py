#!/usr/bin/env python

import os, sys, signal
import time
import csv
import numpy as np
import tensorflow as tf
import pandas as pd
from PIL import Image
import argparse
import keras
import itertools
import matplotlib.pyplot as plt
from keras.preprocessing.image import ImageDataGenerator
from keras.models import Sequential, load_model
from keras.layers import Dense, Dropout, Flatten
from keras.layers import Conv2D, MaxPooling2D, Lambda, Cropping2D
import sklearn
from sklearn.model_selection import train_test_split
from keras.callbacks import EarlyStopping, ModelCheckpoint




def commaAI_model(keep_prob=0.5):
    model = Sequential()
    model.add(Lambda(lambda x: (x / 255.0) - 0.5, input_shape=(160,320,3)))
    model.add(Cropping2D(cropping=((50,20), (0,0)), input_shape=(3,160,320)))
    model.add(Conv2D(16, kernel_size=(3, 3), strides=(4,4), input_shape=(3,90,320), activation="elu"))
    model.add(Conv2D(32, kernel_size=(3, 3), strides=(2,2), activation="elu"))
    model.add((Dropout(keep_prob)))
    model.add(Conv2D(64, kernel_size=(3, 3), strides=(2,2), activation="elu"))
    model.add(Flatten())
    model.add(Dense(512, activation="elu"))
    model.add(Dropout(keep_prob))
    model.add(Dense(1))
    model.compile(loss="mse", optimizer="adam")
    return model
    

def nvidia_model(keep_prob=0.5):
    model = Sequential()
    model.add(Lambda(lambda x: (x / 255.0) - 0.5, input_shape=(160,320,3)))
    model.add(Cropping2D(cropping=((50,20), (0,0)), input_shape=(3,160,320)))
    model.add(Conv2D(16, kernel_size=(3, 3), strides=(4,4), input_shape=(3,90,320), activation="elu"))
    model.add(Conv2D(32, kernel_size=(3, 3), strides=(2,2), activation="elu"))
    model.add((Dropout(keep_prob)))
    model.add(Conv2D(64, kernel_size=(3, 3), strides=(2,2), activation="elu"))
    model.add(Flatten())
    model.add(Dense(512, activation="elu"))
    model.add(Dropout(keep_prob))
    model.add(Dense(1))
    model.compile(loss="mse", optimizer="adam")
    return model



def process_image(img):
    return img.astype("float32")


def generator(samples, batch_size=32):
    num_samples = len(samples)
    while 1: # Loop forever so the generator never terminates
        sklearn.utils.shuffle(samples)
        for offset in range(0, num_samples, batch_size):
            batch_samples = samples[offset:offset+batch_size]
            images = []
            angles = []
            for batch_sample in batch_samples:
                filename = batch_sample[0].split("/")[-1]
                current_path = training_set + "/IMG/" + filename
                center_image = np.asarray(Image.open(current_path))
                center_angle = float(batch_sample[3])
                images.append(center_image)
                angles.append(center_angle)

            # trim image to only see section with road
            X_train = np.array(images)
            y_train = np.array(angles)
            yield sklearn.utils.shuffle(X_train, y_train)




if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Remote Driving")
    parser.add_argument(
        "trainingset",
        type=int,
        nargs="?",
        default=0,
        help="Choose trainingset"
    )
    parser.add_argument(
        "--steering_corr",
        type=float,
        nargs="?",
        default="0.2",
        help="Steering correction related to left and right camera image evaluation in training"
    )
    parser.add_argument(
        "--epochs",
        type=int,
        nargs="?",
        default="20",
        help="Number of epochs for training"
    )
    parser.add_argument(
        "--batchsize",
        type=int,
        nargs="?",
        default="256",
        help="Batchsize for training"
    )
    parser.add_argument(
        "--model",
        type=str,
        nargs="?",
        default="commaAI",
        help="Choose between models: commaAI, nvidia"
    )
    
    args = parser.parse_args()
    
    weights_file    = args.model + "-weights.h5"
    model_file      = args.model + "-model.h5"

    # Parse csv file
    training_set  = "../run%02d" % args.trainingset
    training_data = {"center_imgs": [], "left_imgs": [], "right_imgs": [],
                    "steering_angle_center": [], "steering_angle_left": [], "steering_angle_right": [],
                    "throttle": [], "brake": [], "speed": []}
    samples = []
    with open(training_set+"/driving_log.csv") as logfile:
        reader = csv.reader(logfile, delimiter=",")
        header = next(reader) # it"s safe to skip first row in case it contains headings
        
        for line in reader:
            samples.append(line)
            """
            # images
            for i in range(3):
                filename = line[i].split("/")[-1]
                current_path = training_set + "/IMG/" + filename

                # read in images from center, left and right cameras
                img = process_image(np.asarray(Image.open(current_path)))

                if i == 0:
                    training_data["center_imgs"].append(img)
                if i == 1:
                    training_data["left_imgs"].append(img)
                if i == 2:
                    training_data["right_imgs"].append(img)
            """
            # car control signals
            for i in range(3,7):
                if i == 3:
                    # create adjusted steering measurements for the side camera images
                    training_data["steering_angle_center"].append(float(line[i]))
                    training_data["steering_angle_left"].append(float(line[i]) + args.steering_corr)
                    training_data["steering_angle_right"].append(float(line[i]) - args.steering_corr)
                if i == 4:
                    training_data["throttle"].append(line[i])
                if i == 5:
                    training_data["brake"].append(line[i])
                if i == 6:
                    training_data["speed"].append(line[i])



    try:
        model = load_model(model_file)
        print("Loaded pretrained %s keras model!" % args.model)
    except:        
        if args.model == "commaAI":
            model = commaAI_model(0.2)
        elif args.model == "nvidia":
            model = nvidia_model(0.2)
        else:
            print("Model '%s' not known!" % args.model)
            sys.exit(-1)
        try:
            model.load_weights(weights_file)
            print("Loaded weights for %s keras model!" % args.model)
        except:
            print("Could not load weights for %s keras model from file %s!\nStarting training from scratch!" %
                    (args.model, weights_file))

    # define callbacks for the checkpoint saving and early stopping
    callbacks_list = [
        EarlyStopping(monitor="loss", patience=3, verbose=1),
        ModelCheckpoint(weights_file, monitor="loss", save_best_only=True, mode="auto", save_weights_only=True, verbose=1)
    ]

    ### plot the training and validation loss for each epoch
    fig = plt.figure()
    plt.title("run%02d visualization" % args.trainingset)
    ax1 = plt.subplot(2,1,1)
    ax1.plot(training_data["steering_angle_center"])
    ax1.plot(training_data["throttle"])
    ax1.plot(training_data["brake"])
    ax1.legend(["steering", "throttle", "brake"])
    plt.grid()
    ax2 = plt.subplot(2,1,2, sharex=ax1)
    ax2.plot(training_data["speed"])
    ax2.legend(["speed"])
    plt.grid(), plt.show()
    
    
    train_samples, validation_samples = train_test_split(samples, test_size=0.2)
    nr_train_samples = len(train_samples)
    nr_valid_samples = len(validation_samples)
    print("Samples in training set:   %d" % nr_train_samples)
    print("Samples in validation set: %d" % nr_valid_samples)
    
    biased_index = []
    #biased_index.append(class_indexes_train[30][0])
    #biased_index.append(class_indexes_train[27][0])
    #biased_index = np.array(list(itertools.chain.from_iterable(biased_index)))

    sample_selection        = range(nr_train_samples) #balanced_index #biased_index
    #images_train            = np.array(training_data["center_imgs"])[sample_selection]
    #steering_angles_train   = np.array(training_data["steering_angle_center"])[sample_selection]
    
    # compile and train the model using the generator function
    train_generator = generator(train_samples, batch_size=args.batchsize)
    validation_generator = generator(validation_samples, batch_size=args.batchsize)
    
    model.summary()
    #model_history = model.fit(images_train, steering_angles_train, validation_split=0.2, shuffle=True,
    #                            batch_size=args.batchsize, epochs=args.epochs, verbose=1, callbacks=callbacks_list)
    model_history = model.fit_generator(train_generator, steps_per_epoch=nr_train_samples // args.batchsize, epochs=args.epochs,
                                         validation_data=validation_generator, validation_steps=nr_valid_samples // args.batchsize,
                                         callbacks=callbacks_list, verbose=1)

    ### plot the training and validation loss for each epoch
    plt.plot(model_history.history['loss'])
    plt.plot(model_history.history['val_loss'])
    plt.title('model mean squared error loss')
    plt.ylabel('mean squared error loss')
    plt.xlabel('epoch')
    plt.legend(['training set', 'validation set'], loc='upper right')
    plt.show()

    model.save_weights(weights_file)
    model.save(model_file)

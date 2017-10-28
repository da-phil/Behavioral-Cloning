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
from sklearn.model_selection import train_test_split
from keras.callbacks import EarlyStopping, ModelCheckpoint




def LeNet3_model(keep_prob=0.5):
    # really make sure it gets the dimensions right!

    model = Sequential()
    model.add(Lambda(lambda x: (x / 255.0) - 0.5, input_shape=(160,320,3)))
    model.add(Cropping2D(cropping=((50,20), (0,0)), input_shape=(3,160,320)))
    model.add(Conv2D(64, kernel_size=(3, 3), input_shape=(3,90,320), activation="elu"))
    model.add(MaxPooling2D((3,3)))
    model.add(Conv2D(128, (3, 3), activation="elu"))
    model.add(MaxPooling2D((3, 3)))
    model.add((Dropout(keep_prob)))
    model.add(Flatten())
    model.add(Dense(128, activation="elu"))
    model.add(Dropout(keep_prob))
    model.add(Dense(1))
    model.compile(loss="mse", optimizer=keras.optimizers.Adadelta()) #, metrics=["accuracy"])
    return model
    

def process_image(img):
    return img.astype("float32")



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
        "--weights_file",
        type=str,
        nargs="?",
        default="LeNet3_weights.h5",
        help="File used to store weight values of model"
    )
    parser.add_argument(
        "--model_file",
        type=str,
        nargs="?",
        default="LeNet3_model.h5",
        help="Batchsize for training"
    )
    
    args = parser.parse_args()
    steering_correction = 0.2
    lenet3_weights  = args.weights_file
    lenet3_model    = args.model_file
    nr_training_run = args.trainingset
    epochs          = args.epochs
    BATCH_SIZE      = args.batchsize
    
    # Parse csv file
    training_set  = "../run%02d" % nr_training_run
    training_data = {"center_imgs": [], "left_imgs": [], "right_imgs": [],
                    "steering_angle_center": [], "steering_angle_left": [], "steering_angle_right": [],
                    "throttle": [], "brake": [], "speed": []}
    training_samples = 0
    with open(training_set+"/driving_log.csv") as logfile:
        reader = csv.reader(logfile, delimiter=",")
        header = next(reader) # it"s safe to skip first row in case it contains headings
        for line in reader:
            training_samples += 1
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
            # car control signals
            for i in range(3,7):
                if i == 3:
                    # create adjusted steering measurements for the side camera images
                    training_data["steering_angle_center"].append(float(line[i]))
                    training_data["steering_angle_left"].append(float(line[i]) + steering_correction)
                    training_data["steering_angle_right"].append(float(line[i]) - steering_correction)
                if i == 4:
                    training_data["throttle"].append(line[i])
                if i == 5:
                    training_data["brake"].append(line[i])
                if i == 6:
                    training_data["speed"].append(line[i])

    """
    # preprocessing
    for i in range(training_samples):
        training_data["center_imgs"][i] = np.fliplr(training_data["center_imgs"][i])
        training_data["left_imgs"][i]   = np.fliplr(training_data["left_imgs"][i])
        training_data["right_imgs"][i]  = np.fliplr(training_data["right_imgs"][i])
        training_data["steering"][i]    = -training_data["steering_angle_center"]
    """
    
    try:
        LeNet3 = load_model(lenet3_model)
        print("Loaded pretrained LeNet3 keras model!")
    except:        
        LeNet3 = LeNet3_model(0.2)
        try:
            LeNet3.load_weights(lenet3_weights)
            print("Loaded weights for LeNet3 keras model!")
        except:
            print("Could not load weights for LeNet3 keras model, starting training from scratch!")


    # define callbacks for the checkpoint saving and early stopping
    callbacks_list = [
        EarlyStopping(monitor="loss", patience=2, verbose=1),
        ModelCheckpoint(lenet3_weights, monitor="loss", save_best_only=True, mode="auto", save_weights_only=True, verbose=1)
    ]

    biased_index = []
    #biased_index.append(class_indexes_train[30][0])
    #biased_index.append(class_indexes_train[27][0])
    #biased_index = np.array(list(itertools.chain.from_iterable(biased_index)))

    sample_selection        = range(training_samples) #balanced_index #biased_index
    images_train            = np.array(training_data["center_imgs"])[sample_selection]
    steering_angles_train   = np.array(training_data["steering_angle_center"])[sample_selection]

    # get rid of them for now (does not work anyway? garbage collector doesn't seem to clean up...)    
    del training_data["center_imgs"]
    del training_data["left_imgs"]
    del training_data["right_imgs"]

    LeNet3.summary()
    model_history = LeNet3.fit(images_train, steering_angles_train, validation_split=0.2, shuffle=True,
                                batch_size=BATCH_SIZE, epochs=epochs, verbose=1, callbacks=callbacks_list)

    ### plot the training and validation loss for each epoch
    plt.plot(model_history.history['loss'])
    plt.plot(model_history.history['val_loss'])
    plt.title('model mean squared error loss')
    plt.ylabel('mean squared error loss')
    plt.xlabel('epoch')
    plt.legend(['training set', 'validation set'], loc='upper right')
    plt.show()

    LeNet3.save_weights(lenet3_weights)
    LeNet3.save(lenet3_model)

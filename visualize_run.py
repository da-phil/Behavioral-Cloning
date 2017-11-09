#!/usr/bin/env python

import os, sys
import time
import csv
import numpy as np
import pandas as pd
import argparse
import itertools
import matplotlib.pyplot as plt
import scipy.signal


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Remote Driving")
    parser.add_argument("trainingset",     type=int,    nargs="?",
                        default=0,         help="Choose trainingset")
    parser.add_argument("--steering_corr", type=float,  nargs="?",
                        default="0.2",     help="Steering correction")
    args = parser.parse_args()
    fig = plt.figure()
    
    # Parse csv file
    training_set  = "../run%02d" % args.trainingset
    training_data = {"steering_angle_center": [], "steering_angle_left": [], "steering_angle_right": [],
                     "throttle": [], "brake": [], "speed": []}

    with open(training_set+"/driving_log.csv") as logfile:
        reader = csv.reader(logfile, delimiter=",")
        header = next(reader) # it"s safe to skip first row in case it contains headings
        
        for line in reader:
            # car control signals
            # create adjusted steering measurements for the side camera images
            training_data["steering_angle_center"].append(float(line[3]))
            training_data["steering_angle_left"].append(float(line[3]) + args.steering_corr)
            training_data["steering_angle_right"].append(float(line[3]) - args.steering_corr)
            training_data["throttle"].append(line[4])
            training_data["brake"].append(line[5])
            training_data["speed"].append(line[6])
            
    steering_filtered = scipy.signal.savgol_filter(training_data["steering_angle_center"], 9, 6,
                                                    deriv=0, delta=1.0, axis=-1, mode='interp', cval=0.0)

    nr_samples = len(training_data["speed"])
    print("Samples in dataset:   %d" % nr_samples)

    ### plot the training and validation loss for each epoch
    plt.title("run%02d visualization" % args.trainingset)
    ax1 = plt.subplot(2,1,1)
    ax1.plot(training_data["steering_angle_center"], 'r')
    ax1.plot(steering_filtered, 'r--')
    ax1.plot(training_data["throttle"])
    ax1.plot(training_data["brake"])
    ax1.legend(["steering", "steering_filtered", "throttle", "brake"])
    plt.grid()
    ax2 = plt.subplot(2,1,2, sharex=ax1)
    ax2.plot(training_data["speed"])
    ax2.legend(["speed"])
    plt.grid()
    plt.show()
    

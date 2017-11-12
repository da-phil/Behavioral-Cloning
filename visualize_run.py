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

    #reads CSV file into a single dataframe variable
    training_data_df = pd.read_csv(training_set + "/driving_log.csv",
                                   names=["center", "left", "right", "steering", "throttle", "brake", "speed"])

    steering_filtered = scipy.signal.savgol_filter(training_data_df["steering"].values, 9, 6,
                                                    deriv=0, delta=1.0, axis=-1, mode='interp', cval=0.0)

    nr_samples = len(training_data_df[["speed"]].values)
    print("Samples in dataset:   %d" % nr_samples)

    ### plot the training and validation loss for each epoch
    plt.title("run%02d visualization" % args.trainingset)
    ax1 = plt.subplot(2,1,1)
    ax1.plot(training_data_df["steering"].values, 'r')
    ax1.plot(steering_filtered, 'r--')
    ax1.plot(training_data_df["throttle"].values)
    ax1.plot(training_data_df["brake"].values)
    ax1.legend(["steering", "steering_filtered", "throttle", "brake"])
    plt.grid()
    ax2 = plt.subplot(2,1,2, sharex=ax1)
    ax2.plot(training_data_df["speed"].values)
    ax2.legend(["speed"])
    plt.grid()
    plt.show()

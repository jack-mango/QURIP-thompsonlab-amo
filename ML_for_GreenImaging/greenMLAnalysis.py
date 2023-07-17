from PipelineStages import *
import argparse
from tensorflow.keras import models
import logging
import numpy as np
import os
from scipy.io import loadmat

class greenMLAnalysis():

    def __init__(self, data, n_tweezers, n_loops):
        self.n_tweezers = n_tweezers
        self.n_loops = n_loops
        self.img_processor = self.make_img_processor(data)

        self.positions = None
        self.model = None

    def make_img_processor(self, data):
        if type(data) == str:
            stack, n_files = self.load_data(data)
        else:
            stack = data
        return ImageProcessing.ImageProcessor(stack, self.n_tweezers)

    def load_data(self, path):
        """
        Load .mat files from self.data_path if that directory exists.

        Returns:
        - stack: brightness values for each image stored in a numpy array. The first axis
                corresponds ot file number and the second image number within the file.
                The last two axes of this array hold the row and column pixel values for 
                each image.
        - n_files: the number of .mat files found at self.data_path
        """
        n_files = 0
        if os.path.isdir(path):
            stack = []
            for file in os.listdir(path):
                if file.endswith('.mat'):
                    data = loadmat(path + '/' + file)
                    stack.append(data['stack'])     
                    n_files += 1
            stack = np.concatenate(stack, axis=0)
        else:
            stack = loadmat(path)['stack']
            n_files += 1  
        return stack, n_files

    def find_tweezer_positins(self):
        self.positions = self.img_processor.find_tweezer_positions()
        self.img_processor.plot()
        return self.positions

    def make_labels(self):


    def train_model(self):
        nn_dist = self.img_processor.find_nn_dist()
        crops_1x1 = self.img_processor.crop_tweezers(1, nn_dist)
        crops_3x3 = self.img_processor.crop_tweezers(3, nn_dist)
        labels = self.make_labels(crops_1x1)
        return
    
    def make_crops(self):
        return
    
    def make_labels(self):
        fits, r_sq = labeler.bright_dark_fit()
        thresholds, plots = labeler.find_thresholds(fits)
        all_below_upper, all_above_lower = labeler.threshold_misfits(thresholds)
        labels = labeler.make_labels(thresholds)
        info = {
            "Histogram fits plot": plots,
            "Thresholds": thresholds,
            "Tweezers missing dark labels": all_above_lower,
            "Tweezers missing bright labels": all_below_upper,
            "R^2 Values": r_sq,
            "Image Value Distribution Fits": fits
            }
    return labels, info

    
    def fidelity_analysis(self, stack):
        return
    
    def get_occupancies(self, stack):
        return
import numpy as np
import matplotlib.pyplot as plt
from . import ImageProcessing, AutoGauss
import os

import logging
log = logging.getLogger(__name__)

class Labeler():

    """ 
    Task: take in crops and make labels
    Information out (but not needed for next steps): thresholds, fits and maybe classification graphs? -- store as attributes

    I take in crops and produce labels (and thresholds since they're needed for plotting) -- loading data is someone else's job
    What about fits for the plots? Parameters are needed for that right?
    """

    def __init__(self, crops, n_tweezers, n_loops, per_loop):
        self.n_tweezers = n_tweezers
        self.n_loops = n_loops
        self.per_loop = per_loop
        self.crop_brightness = self.find_crop_brightness(crops)

    def run(self):
        """ 
        Returns a dataset of cropped images that are labeled by which loop number and which image in the loop
        they're in. If keep_unknowns is set to False, then any crop with a label that correspondds to unknown is 
        discarded from the dataset. Images are labeled by occupancy of central site:
            [0, 1] -> bright
            [1, 0] -> dark
            [NaN, NaN] -> unknown
        Transients are any shots which correspond to jumps from the metastable state to the ground state, thus causing
        a short burst of dark shots surrounded by otherwise bright ones.
        """
        fits = self.bright_dark_fit()
        thresholds = self.find_thresholds(fits)
        labels = self.make_labels(thresholds)
        # return info
        return labels, self.crop_brightness, fits, thresholds
    
    def bright_dark_fit(self):
        """
        Fit a double gaussian function to the given data and return the parameters of the fit, with the lower mean Gaussian
        first and the higher mean one second. 
        """
        fits = np.empty((self.n_tweezers, 2, 3))
        for i in range(self.n_tweezers):
            model = AutoGauss.GMM(self.crop_brightness[i])
            fits[i] = model.fit()
        return fits
    
    def find_crop_brightness(self, crops):
        """
        Given an array of image crops corresponding to a single tweezer give the average pixel value for the image
        """
        return np.mean(crops, axis=(2, 3))
    
    def find_thresholds(self, fits, plot=False, z=4.753424308822899):
        """
        For each site in the lattice, find the pixel thresholds.
        """
        if plot:
            fig, axs = plt.subplots(
                self.n_tweezers // 5 + (self.n_tweezers % 5 > 0), 5, figsize=(16, self.n_tweezers // 4))
            plt.tight_layout(pad=2)
        thresholds = np.empty((self.n_tweezers, 2))
        for i, fit in enumerate(fits):
            lower_thresh = fit[0, 0] - fit[0, 1] * z
            upper_thresh = fit[1, 0] + fit[1, 1] * z
            thresholds[i] = np.array([lower_thresh, upper_thresh])
            #if plot:
            #    counts, bins, _ = axs[i // 5][i %
            #                                  5].hist(avg, bins=(self.per_loop // 4), density=True)
            #    x_vals = np.linspace(bins[0], bins[-1], self.per_loop)
            #    axs[i // 5][i %
            #                5].plot(x_vals, AutoGauss.double_gaussian(x_vals, *dark_fit, *bright_fit), 'k')
            #    axs[i // 5][i %
            #                5].axvline(lower_thresh, color='r', linestyle='--')
            #    axs[i // 5][i %
            #                5].axvline(upper_thresh, color='r', linestyle='--')
            #    axs[i // 5][i % 5].set_title(f"Tweezer {i}")
        return thresholds
    
    def make_labels(self, thresholds, plot=False):
        """
        Given an upper and lower threshold, classify whether tweezer sites are occupied. You can read more about the
        algorithm used for classification in the documentation.
        """
        labels = np.empty((self.n_tweezers, self.n_loops, self.per_loop))
        for i, tweezer_vals in enumerate(self.crop_brightness):
            for loop_num in range(self.n_loops):
                loop = tweezer_vals[loop_num * self.per_loop: (loop_num + 1) * self.per_loop]
                labels[i, loop_num] = self.slicer(loop, *thresholds[i])
        return labels.ravel()
    
    def slicer(self, arr, lower_thresh, upper_thresh):
        """
        Labels images as bright if they fall between two images that exceed bright thresholds, and similarly
        dark if image falls between two images exceeding dark threshold. If an image lies on the transition from
        a dark to a bright threshold it's labeled as unknown, enocded with an np.NaN.
        """
        labels = np.empty(arr.size)
        head = tail = 0
        bright = True
        for i, val in enumerate(arr):
            if val >= upper_thresh and bright:
                head = i + 1
            elif val >= upper_thresh and not bright:
                labels[head:i] = np.full(i - head, np.NaN)
                labels[tail:head] = np.zeros(head - tail)
                tail = i
                head = i  # + 1
                bright = True
            elif val <= lower_thresh and not bright:
                head = i + 1
            elif val <= lower_thresh and bright:
                labels[head:i] = np.full(i - head, np.NaN)
                labels[tail:head] = np.ones(head - tail)
                head = i  # + 1
                tail = i
                bright = False
        if bright:
            labels[tail:head] = np.ones(head - tail)
            labels[head:] = np.full(labels.size - head, np.NaN)
        else:
            labels[tail:] = np.zeros(labels.size - tail)
        return labels
    
    #def threshold_plot(self, tweezer_num):
    #    tweezer_vals = np.mean(self.crops[self.crop_index(tweezer_num, 0, 0): self.crop_index(tweezer_num + 1, 0, 0)], axis=(1, 2))
    #    tweezer_labels = self.labels[self.crop_index(tweezer_num, 0, 0): self.crop_index(tweezer_num + 1, 0, 0)]
#
    #    bright_mask = tweezer_labels[:, 1] == 1
    #    dark_mask = tweezer_labels[:, 0] == 1
    #    unknown_mask = np.isnan(tweezer_labels[:, 0])
#
    #    bright_indices = np.where(bright_mask)[0]
    #    bright_vals = tweezer_vals[bright_mask]
#
    #    dark_indices = np.where(dark_mask)[0]
    #    dark_vals = tweezer_vals[dark_mask]
#
    #    unknown_indices = np.where(unknown_mask)[0]
    #    unknown_vals = tweezer_vals[unknown_mask]
#
    #    print(len(unknown_vals))
    #    plt.figure(figsize=(20, 10))
    #    plt.plot(bright_indices, bright_vals, '.', label='bright')
    #    plt.plot(dark_indices, dark_vals, '.', label='dark')
    #    plt.plot(unknown_indices, unknown_vals, 'o', label='?')
    #    plt.axhline(self.thresholds[tweezer_num, 1], color='r', linestyle='--', label=f"Upper Threshold = {self.thresholds[tweezer_num, 1]:.3f}")
    #    plt.axhline(self.thresholds[tweezer_num, 0], color='g', linestyle='--', label=f"Lower Threshold = {self.thresholds[tweezer_num, 0]:.3f}")
    #    plt.legend()
    #    plt.title(f"Tweezer Number = {tweezer_num}")
    #    for i in range(self.n_loops):
    #        plt.axvline(i * self.per_loop, color='k', linestyle='--', label="Loop Separation")
    #    plt.show()
import numpy as np
import cv2
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

    def __init__(self, crops, n_tweezers, n_loops):
        self.n_tweezers = n_tweezers
        self.n_loops = n_loops
        self.per_loop = crops.shape[1] // n_loops
        self.img_vals = self.find_img_vals(crops)
        self.info = None

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
        fits, r_sq = self.bright_dark_fit()
        thresholds, plots = self.find_thresholds(fits)
        bad_thresholds = self.threshold_misfits(thresholds)
        labels = self.make_labels(thresholds)
        self.info = {"Histogram fits plot": plots,
                "Thresholds": thresholds,
                "Bad Thresholds": bad_thresholds,
                "R^2 Values": r_sq}
        return labels, self.info
    
    def bright_dark_fit(self):
        """
        Fit a double gaussian function to the given data and return the parameters of the fit, with the lower mean Gaussian
        first and the higher mean one second. 
        """
        fits = np.empty((self.n_tweezers, 2, 3))
        r_sq = np.empty(self.n_tweezers)
        for i in range(self.n_tweezers):
            model = AutoGauss.GaussianMixture(self.img_vals[i])
            fits[i], r_sq[i] = model.fit()
        return fits, r_sq
    
    def find_img_vals(self, crops):
        """
        Given an array of image crops corresponding to a single tweezer give the average pixel value for the image
        """
        return np.mean(crops, axis=(2, 3))
    
    def find_thresholds(self, fits, z=4.753424308822899):
        """
        For each site in the lattice, find the pixel thresholds.
        """
        fig, axs = plt.subplots(self.n_tweezers // 5 + (self.n_tweezers % 5 > 0), 5, figsize=(8.5, 11 * self.n_tweezers / 50))
        fig.tight_layout(h_pad=0.8)
        thresholds = np.empty((self.n_tweezers, 2))
        log.info("Making threshold plots...")
        for i, fit in enumerate(fits):
            lower_thresh = fit[1, 0] - fit[1, 1] * z
            upper_thresh = fit[0, 0] + fit[0, 1] * z
            thresholds[i] = np.array([lower_thresh, upper_thresh])
            counts, bins, _ = axs[i // 5][i % 5].hist(self.img_vals[i], bins=(self.per_loop // 4), density=True)
            x_vals = np.linspace(bins[0], bins[-1], self.per_loop)
            # Add in R^2 calculation here
            axs[i // 5][i % 5].plot(x_vals, AutoGauss.double_gaussian(x_vals, *fit[0], *fit[1]), 'k')
            axs[i // 5][i % 5].axvline(lower_thresh, color='r', linestyle='--')
            axs[i // 5][i % 5].axvline(upper_thresh, color='r', linestyle='--')
            axs[i // 5][i % 5].set_title(f"Tweezer {i}", fontsize=8)
            axs[i // 5][i % 5].tick_params(axis='both', labelsize=8)
        return thresholds, (fig, axs)
    
    def make_labels(self, thresholds, plot=False):
        """
        Given an upper and lower threshold, classify whether tweezer sites are occupied. You can read more about the
        algorithm used for classification in the documentation.
        """
        labels = np.empty((self.n_tweezers, self.n_loops, self.per_loop))
        for i, tweezer_vals in enumerate(self.img_vals):
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
    
    def threshold_misfits(self, thresholds):
        """ 
        Inspects the thresholds, reporting if any thresholds are lower/higher than all brightness values
        for that respective tweezer. Returns an n_tweezers x 2 array boolean array, where if the first
        (second) entry is True then all brightness values are above (below) the respective threshold.
        """
        labels = np.empty((self.n_tweezers, 2), dtype=bool)
        for i, thresh in enumerate(thresholds):
            all_below_upper = False
            all_above_lower = False
            if np.all(thresh[0] < self.img_vals):
                all_above_lower = True
                log.warning(f"All brightness values for tweezer {i} are above the lower threshold!\nNo images for this tweezer can be labeled dark!")
            if np.all(thresh[1] > self.img_vals):
                all_below_upper = True
                log.warning(f"All brightness values for tweezer {i} are below the upper threshold!\nNo images for this tweezer can be labeled bright!")
            labels[i] = np.array([all_above_lower, all_below_upper])
        return labels


    def threshold_plot(self, tweezer_num, labels):
        thresholds = self.info["Thresholds"]

        tweezer_vals = self.img_vals[tweezer_num]
        tweezer_labels = labels[tweezer_num * self.per_loop * self.n_loops:(tweezer_num + 1)* self.per_loop * self.n_loops:]

        bright_mask = tweezer_labels == 1
        dark_mask = tweezer_labels == 0
        unknown_mask = np.isnan(tweezer_labels)

        bright_indices = np.where(bright_mask)[0]
        bright_vals = tweezer_vals[bright_mask]

        dark_indices = np.where(dark_mask)[0]
        dark_vals = tweezer_vals[dark_mask]

        unknown_indices = np.where(unknown_mask)[0]
        unknown_vals = tweezer_vals[unknown_mask]

        print(unknown_mask)
        print(dark_vals.shape)

        fig = plt.figure(figsize=(20, 10))
        plt.plot(bright_indices, bright_vals, '.', label='bright')
        plt.plot(dark_indices, dark_vals, '.', label='dark')
        plt.plot(unknown_indices, unknown_vals, 'o', label='?')
        plt.axhline(thresholds[tweezer_num, 1], color='r', linestyle='--', label=f"Upper Threshold = {thresholds[tweezer_num, 1]:.3f}")
        plt.axhline(thresholds[tweezer_num, 0], color='g', linestyle='--', label=f"Lower Threshold = {thresholds[tweezer_num, 0]:.3f}")
        plt.legend(loc='upper right')
        plt.title(f"Tweezer Number = {tweezer_num}")
        for i in range(self.n_loops):
            plt.axvline(i * self.per_loop, color='k', linestyle='--', label="Loop Separation")
        return fig

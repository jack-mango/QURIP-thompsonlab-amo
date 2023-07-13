import numpy as np
import cv2
import matplotlib.pyplot as plt
from . import ImageProcessing, AutoGauss
import os

import logging
log = logging.getLogger(__name__)

class Labeler():

    """
    Contains the methods used in the labeling step of the training pipeline. Crop labels are generated
    based on the brightness of the central lattice site.

    Attributes:
    - n_tweezers: the number of tweezers in the stack images.
    - n_loops: number of loops in the image stack.
    - per_loop: the number of images in each loop of the stack.
    - img_vals: a single value that is attributed to each crop to be labeled.
    - info: the relevant information gathered when executing this pipeline stage as a
        dictionary.
    """

    def __init__(self, crops, n_tweezers, n_loops):
        self.n_tweezers = n_tweezers
        self.n_loops = n_loops
        self.per_loop = crops.shape[1] // n_loops
        self.img_vals = self.find_img_vals(crops)
        self.labels = None
        self.info = None

        print(self.img_vals.shape)

    def run(self):
        """
        Execute the necessary methods of this class to output information for
        the next pipeline stage.

        Returns:
        - labels: A one dimensional array of labels for each image value. 
         - info: a dictionary containing a plot of the thresholding fits for each tweezer, the thresholds,
                the tweezer number of any misfit thresholds, and the R^2 values for bimodal Gaussian fits to image values
                for each tweezer. 
        """
        fits, r_sq = self.bright_dark_fit()
        thresholds, plots = self.find_thresholds(fits)
        all_below_upper, all_above_lower = self.threshold_misfits(thresholds)
        self.labels = self.make_labels(thresholds)
        self.info = {"Histogram fits plot": plots,
                "Thresholds": thresholds,
                "Tweezers missing dark labels": all_above_lower,
                "Tweezers missing bright labels": all_below_upper,
                "R^2 Values": r_sq,
                "Image Value Distribution Fits": fits
                }
        return self.labels, self.info
    
    def bright_dark_fit(self):
        """
        Fit a bimodal Gaussian distribution to each self.img_vals for each individual tweezer. 

        Returns:
        - fits: a self.n_tweezers x 2 x 3 array, where the ith entry in the first axis corresponds to the ith tweezer,
                the first (second) entry on the second axis corresponds to Gaussian fit with the lower (higher) mean,
                and the final axis contains the mean, standard deviation, and relative amplitude in that order.
        - r_sq : the R^2 value of the fits contained in a self.n_tweezers long array. 
        """
        fits = np.empty((self.n_tweezers, 2, 3))
        r_sq = np.empty(self.n_tweezers)
        for i in range(self.n_tweezers):
            model = AutoGauss.GaussianMixture(self.img_vals[i])
            fits[i], r_sq[i] = model.fit()
        return fits, r_sq
    
    def find_img_vals(self, crops, r_atom=2.5):
        """
        Assign a brightness score to each image in crops. In the current implementation this is done
        by taking the average pixel value across each image in crops

        Parameters:
        - crops: a four dimensional array of images where the first axis corresponds to tweezer number, 
                the second corresponds to image number, and the final two correspond to the row and column
                numbers of the pixels within the image.

        Returns:
        - img_vals: values assigned to each crop according to the function method used by this method
        """
        #kernel = np.ravel(gaussian_kernel(crops.shape[-1], r_atom * 0.600600600601 / 2))
        #plt.imshow(gaussian_kernel(crops.shape[-1], r_atom * 0.600600600601 / 2))
        #plt.colorbar()
        #return np.average(np.reshape(crops, (*crops.shape[:2], -1)), weights=kernel, axis=2)
        return np.mean(crops, axis=(2, 3))
    
    def find_thresholds(self, fits, z=4.753424308822899):
        """
        Assuming img_vals follows a bimodal Gaussian distribution, calculates the image values that are
        z standard distributions above (below) the lower (upper) threshold. The particular value of z chosen
        corresponds to the number of standard deviations that 1 - CDF(x) = 1e-6 for a Gaussian distribution along
        whichever direction corresponds to the lower or upper thresholding.

        Parameters:
        - fits: parameters for a Gaussian fit. Must be a numpy array with shape (n_tweezers, 2, 3), with the first axis
                corresponding to tweezer number, the second corresponding to which Gaussian mode (lower mean first, higher 
                mean second), and the third containing the parameters corresponding to that Gaussian mode, ordered
                (mean, standard deviation, relative amplitude).
        - fig: a plot for each tweezer of its bimodal Gaussian fit, its image value histogram, and its thresholds.

        Returns:
        - sorted_vectors: an m x 2 array of the vectors sorted based on which tile they fall into. If
                          two vectors fall in the same tile, then the order is ambiguous.
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
        return thresholds, fig
    
    def make_labels(self, thresholds):
        """
        Create a label for each crop, corresponding to an image value. If the crop's image value falls below the tweezer's lower
        threshold or in a segment of image values that fall below the lower threshold and don't rise above the upper threshold, then 
        it is labeled as dark. Similarly if a crop's image value is above the tweezer's upper threshold or in a segment of
        image values above the upper theshold and don't fall below the lower threshold, then it is labeled as bright. If an image
        value is between the two thresholds and isn't in a segment of values all above or below one of the thresholds, it is labeled
        as unknown. Dark crops are labeled by 0, bright crops by 1, and unknown crops by NaNs. 

        Parameters:
        - thresholds: an n_tweezers x 2 array, with the first axis corresponding to the tweezer number and the second corresponding
                    to the threshold; lower threshold first and upper threshold second.
        
        Returns:
        - labels: A one dimensional array of labels for each image value. 
        """
        labels = np.empty((self.n_tweezers, self.n_loops, self.per_loop))
        for i, tweezer_vals in enumerate(self.img_vals):
            for loop_num in range(self.n_loops):
                loop = tweezer_vals[loop_num * self.per_loop: (loop_num + 1) * self.per_loop]
                labels[i, loop_num] = self.slicer(loop, *thresholds[i])
        return labels.ravel()
    
    def slicer(self, arr, lower_thresh, upper_thresh):
        """
        Finds where values of arr that are above (below) upper_thresh (lower_thresh). Indices of arr that lie between two
        indices that are both above or below the same threshold are labeled as bright (1) or dark (0). Indices of arr that
        are between the two thresholds are labeled as unknown (NaN).

        Parameters:
        - arr: a one dimensional array of image values.
        - lower_thresh: a single number to be used as the lower threshold.
        - upper_thresh: a single number to be used as the upper threshold.
    
        Returns:
        - labels: A one dimensional array of labels for each image value. 
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
                head = i
                bright = True
            elif val <= lower_thresh and not bright:
                head = i + 1
            elif val <= lower_thresh and bright:
                labels[head:i] = np.full(i - head, np.NaN)
                labels[tail:head] = np.ones(head - tail)
                head = i
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
        Note if any tweezer's thresholds are so high (low) such that all that tweezer's image values lie below (above) that
        threshold, leading to a lack of bright (dark) images for that tweezer.

        Parameters:
        - thresholds: an n_tweezers x 2 array, with the first axis corresponding to the tweezer number and the second corresponding
                    to the threshold; lower threshold first and upper threshold second.
    
        Returns:
        - all_below_upper: a list containing the indices for which all image values are below the upper threshold.
        - all_above_lower: a list containing the indices for which all image values are above the lower threshold.
        """
        all_below_upper = []
        all_above_lower = []
        for i, thresh in enumerate(thresholds):
            if np.all(thresh[0] < self.img_vals):
                all_above_lower.append(i)
                log.warning(f"All brightness values for tweezer {i} are above the lower threshold!\nNo images for this tweezer can be labeled dark!")
            if np.all(thresh[1] > self.img_vals):
                all_below_upper.append(i)
                log.warning(f"All brightness values for tweezer {i} are below the upper threshold!\nNo images for this tweezer can be labeled bright!")
        return all_below_upper, all_above_lower
    


    def threshold_plot(self, tweezer_num, show_unknowns=False):
        """
        Create a plot for an individual tweezer displaying its brightness values color coded according to their label. Also plotted
        are thresholds and loop number dividers. 

        Parameters:
        - tweezer_num: an integer indicating which tweezer's image values to plot.
    
        Returns:
        - fig: a matplotlib figure displaying the color coded image values, thresholds, and loop dividers.  
        """
        thresholds = self.info["Thresholds"]

        tweezer_vals = self.img_vals[tweezer_num]
        tweezer_labels = self.labels[tweezer_num * self.per_loop * self.n_loops:(tweezer_num + 1)* self.per_loop * self.n_loops:]

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
        if show_unknowns:
            plt.plot(unknown_indices, unknown_vals, 'o', label='?')
        plt.axhline(thresholds[tweezer_num, 1], color='r', linestyle='--', label=f"Upper Threshold = {thresholds[tweezer_num, 1]:.3f}")
        plt.axhline(thresholds[tweezer_num, 0], color='g', linestyle='--', label=f"Lower Threshold = {thresholds[tweezer_num, 0]:.3f}")
        plt.legend(loc='upper right')
        plt.title(f"Tweezer Number = {tweezer_num}")
        for i in range(self.n_loops):
            plt.axvline(i * self.per_loop, color='k', linestyle='--', label="Loop Separation")
        return fig 
    
def gaussian_kernel(k_size, std):
    kernel = cv2.getGaussianKernel(k_size, std)
    return np.matmul(kernel, kernel.T)
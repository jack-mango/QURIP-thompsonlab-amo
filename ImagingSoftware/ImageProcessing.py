import numpy as np
import cv2
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit
from scipy.special import ndtri

from AutoGauss import *


class ImageProcessor():
    """ 
        A base class that can be used to process lattice images and determine occupancy.
    """

    def __init__(self, stack, lattice_shape):
        """
        stack : An array of images, with the first axis being the image axis and the remaining two being pixel
            intensity values.
        lattice_shape : An array with a two entries, (m, n), where m denotes how many sites there are in the
            horizontal direction and n denotes how many in the vertical. 
        """
        self.stack = stack
        self.lattice_shape = lattice_shape
        self.n_tweezers = np.prod(lattice_shape)
        self.img_height, self.img_width = stack.shape[1], stack.shape[2]
        self.a0, self.a1, self.lattice_offset = self.lattice_characteristics_rect()
        self.tweezer_positions = self.tweezer_positions()

    def pixel(self, x):
        """ 
        Rounds x to the nearest integer, corresponding to the nearest pixel. x can be an array or a scalar.
        """
        return np.rint(x).astype(int)

    def lattice_characteristics_rect(self, plot=False):
        """ 
        Approximates the lattice constants a0 and a1, and lattice offset by fitting periodic Gaussians
        to the averaged stack array. This algorithm assumes that the lattice has a rectangular shape. More
        about the algorithm used can be found in the documentation.
        """
        xdata = np.mean(self.stack, axis=(0, 2))
        ydata = np.mean(self.stack, axis=(0, 1))
        # TODO: Make the gaussian no longer have to estimate standard deviation, offset and amplitude
        # rather use this information from what can be gathered about the lattice. Amplitude and standard deviation
        # can be gained from the averaged image but offset should probably come from background average.
        xparams, xcov = curve_fit(periodic_gaussian_1d(self.lattice_shape[0]), np.arange(len(xdata)), xdata,
                                  p0=self.guess(xdata), bounds=(self.lower_bounds(xdata), self.upper_bounds(xdata)))
        yparams, ycov = curve_fit(periodic_gaussian_1d(self.lattice_shape[1]), np.arange(len(ydata)), ydata,
                                  p0=xparams, maxfev=int(1e4), bounds=(self.lower_bounds(ydata), self.upper_bounds(ydata)))
        # The curve fitting works quite well for finding the lattice constant but not so mudch for the offset.
        # We try to shift the offset over in the x and y directions by one lattice constant at a time so
        # long as the error keeps decreasing.
        xerr, yerr = np.linalg.norm(
            np.diag(xcov)), np.linalg.norm(np.diag(ycov))
        right, right_err = self.find_offset(xdata, xparams, 1, xerr)
        left, left_err = self.find_offset(xdata, xparams, -1, xerr)
        up, up_err = self.find_offset(ydata, yparams, 1, yerr)
        down, down_err = self.find_offset(ydata, yparams, -1, yerr)
        if right_err > left_err:
            xparams = left
        else:
            xparams = right
        if up_err > down_err:
            yparams = down
        else:
            yparams = up
        if plot:
            fig, axs = plt.subplots(2, 1, figsize=(16, 8))
            x_vals = np.linspace(0, self.img_width, 1000)
            y_vals = np.linspace(0, self.img_height, 1000)
            xfunc, yfunc = periodic_gaussian_1d(
                self.lattice_shape[0]), periodic_gaussian_1d(self.lattice_shape[1])
            axs[0].plot(xdata, 'k.')
            axs[0].plot(x_vals, xfunc(x_vals, *xparams))
            axs[0].set_title("Flattened X Values")
            axs[1].plot(ydata, 'k.')
            axs[1].plot(y_vals, yfunc(y_vals, *yparams))
            axs[1].set_title("Flattened Y Values")
        return np.array([xparams[0], 0]), np.array([0, yparams[0]]), np.array([xparams[1], yparams[1]])

    def find_tweezer_positions(self, r=2):
        """ 
        Attempts to find locations of tweezers for any arbitrary layout. Does so by find brightest
        local maxima in the average of the stack, then setting regions of (2r + 1, 2r + 1) to zero in 
        order to prevent double counting bright pixels associated with the same tweezer. Since it's
        possible that this masking isn't large enough, if any "bright" pixels are adjacent to a mask,
        they're considered to be part of that tweezer and masked off as well.
        """
        positions = []
        img = np.mean(self.stack, axis=0)
        dilate = cv2.dilate(img, None)
        erode = cv2.erode(dilate, None)
        matching = np.where(erode == img, img, -255)
        i = 0
        d = 2 * r + 1
        while i < self.n_tweezers:
            pos = np.unravel_index(np.argmax(matching), img.shape)
            if any(i == 0 for i in pos) or (matching[pos[0] - 1: pos[0] + 1, pos[1] - 1: pos[1] + 1] == 0).any():
                matching[pos] = 0
            else:
                matching[pos[0] - r: pos[0] + r + 1, pos[1] - r: pos[1] + r + 1] = np.zeros((d, d))
                positions.append(pos)
                i += 1
        return positions

    def find_offset(self, data, guess, direction, min_err):
        """ 
        Used in finding the lattice constants. Given data to fit to an and initial guess, this method tries
        shifting the lattice over by one site at a time, continuing if the error of the fit generated by a shift
        if lower than before. If direction should either be a 1 or -1, denoting shifting to the right or left,
        respectively.
        """
        guess[1] += direction * guess[0]
        params, cov = curve_fit(periodic_gaussian_1d(self.lattice_shape[0]), np.arange(len(data)), data,
                                p0=guess)
        err = np.linalg.norm(np.diag(cov))
        if err < min_err:
            return self.find_offset(data, params, direction, err)
        else:
            guess[1] -= direction * guess[0]
            return guess, min_err

    def guess(self, data):
        """
        Generate a reasonable guess for the fitting used in the periodic Gaussian fitting based on
        the size of the image and typical lattice parameters.
        """
        lattice_constant = 5.7
        lattice_offset = 1
        std = 1.2
        scaling = data.max() - data.min()
        offset = data.min()
        return [lattice_constant, lattice_offset, std, scaling, offset]

    def lower_bounds(self, data):
        """
        Generate an absolute lower bound on reasonable fitting parameters based on the size of the 
        image and possible lattice parameters.
        """
        lattice_constant = 2.4
        lattice_offset = 0
        std = 0
        scaling = (data.max() - data.min()) / 4
        offset = 0
        return [lattice_constant, lattice_offset, std, scaling, offset]

    def upper_bounds(self, data):
        """
        Generate an absolute upper bound on reasonable fitting parameters based on the size of the 
        image and possible lattice parameters. 
        """
        lattice_constant = np.inf
        lattice_offset = np.inf
        std = self.img_width / (self.lattice_shape[0])
        scaling = np.inf
        offset = np.inf
        return [lattice_constant, lattice_offset, std, scaling, offset]

    def tweezer_positions(self):
        """
        Returns an array of the positions of the tweezers.
        """
        positions = []
        for i in range(self.n_tweezers):
            row = i // self.lattice_shape[0]
            col = np.mod(i, self.lattice_shape[0])
            positions.append(self.lattice_offset + col *
                             self.a0 + row * self.a1)
        return np.array(positions)

    def crop(self, x, y, h_border, v_border):
        """
        Returns the images from the stack corresponding to the pixels centered at (x, y),
        with horizontal and vertical borders of pixels corresponding to h_border and v_border.
        """
        return self.stack[:, self.pixel(x - h_border): self.pixel(x + h_border),
                          self.pixel(y - v_border): self.pixel(y + v_border), ]

    def crop_sites(self, n):
        """
        Given n, return an array containing a crop that includes n x n lattice sites centered on each site.
        """
        h_border, v_border = self.pixel(n * (self.a0 + self.a1) / 2)
        cropped_sites = []
        for position in self.tweezer_positions:
            cropped_sites.append(self.crop(*position, h_border, v_border))
        return np.concatenate(cropped_sites, axis=0)

    def dataset_index(self, tweezer_num, loop_num, img_num):
        """
        Given tweezer number, loop number, and image number, return the index that corresponds to the image
        in the crops array that would be returned by make_dataset().
        """
        return tweezer_num * self.n_loops * self.per_loop + loop_num * self.per_loop + img_num

    def make_dataset(self, n=3):
        """
        Returns cropped sites including n x n lattice sites for each image in the stack and each lattice site.
        """
        crops = self.crop_sites(n)
        return crops

    def plot(self, index=None):
        """
        Generate a plot of the lattice. If no index is provided the average of all images is taken and plotted.
        If a single index number is provided then corresponding picture of the entire lattice is plotted.
        """
        if index == None:
            img = self.stack.mean(axis=0)
        else:
            img = self.stack[index]
        plt.imshow(np.transpose(img), cmap='plasma')
        plt.colorbar()
        for position in self.tweezer_positions:
            plt.plot(*position, 'ws', fillstyle='none', alpha=0.8)
        plt.show()

    def mean_dark(self):
        return

    def mean_bright(self):
        return

    def background_noise(self):
        """
        Returns the average and standard deviation for pixels in the background region. The background
        region is considered to be everthing outside the bounding rectangle containing the lattice. 
        """
        lattice_region = np.matmul(
            np.array([self.a0, self.a1]), self.lattice_shape)
        avg = np.mean(self.stack, axis=0)
        x_low, y_low = self.pixel(self.lattice_offset)
        x_high, y_high = self.pixel(lattice_region)
        avg[x_low: x_high, y_low: y_high] = np.full(
            (x_high - x_low, y_high - y_low), np.NaN)
        return np.nanmean(avg), np.nanstd(avg)


class GreenImageProcessor(ImageProcessor):

    """
    A subclass of the ImageProcessor class, used specifically for processing green images. This class
    contains functionality to crop images to the correct shape and label them based on thresholding. 
    """

    def __init__(self, stack, lattice_shape, n_loops):
        """
        stack : An array of images, with the first axis being the image axis and the remaining two being pixel
            intensity values.
        lattice_shape : An array with a two entries, (m, n), where m denotes how many sites there are in the
            horizontal direction and n denotes how many in the vertical. 
        n_loops : The number of loops in the dataset
        """
        super().__init__(stack, lattice_shape)
        self.n_loops = n_loops
        self.per_loop = self.stack.shape[0] // n_loops

    def crop_index(self, tweezer_num, loop_num, img_num):
        """
        Given tweezer number, loop number, and image number, return the index that corresponds to the image
        in the crops array that would be returned by make_dataset.
        """
        return tweezer_num * self.n_loops * self.per_loop + loop_num * self.per_loop + img_num

    def make_dataset(self, n=3, keep_unknowns=False, keep_transients=False, plot=False):
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
        crops = self.crop_sites(n)
        labels, thresholds = self.make_labels_v2(plot=plot)
        if keep_unknowns:
            return crops, labels
        else:
            mask = ~ np.isnan(labels[:, 1])
            return crops[mask], labels[mask]

    def make_labels_v1(self, plot=False):
        """
        DEPRECATED! Given an upper and lower threshold, classify whether lattice sites are occupied.
        You can read more about the algorithm used for classification in the documentation. 
        """
        crops = self.crop_sites(1)
        thresholds = self.find_thresholds(crops, plot=plot)
        labels = np.empty(self.n_tweezers * self.n_loops * self.per_loop)
        for tweezer_num in range(self.n_tweezers):
            for loop_num in range(self.n_loops):
                # NOTE: Might need to change this to a weighted Gaussian instead of pure mean.
                loop = crops[self.crop_index(tweezer_num, loop_num, 0): self.crop_index(
                    tweezer_num, loop_num + 1, 0)]
                avg = np.mean(loop, axis=(1, 2))
                over_upper_thresh = np.where(
                    avg > thresholds[tweezer_num][1])[0]
                under_lower_thresh = np.where(
                    avg < thresholds[tweezer_num][0])[0]
                if over_upper_thresh.size == 0:
                    last_bright = 0
                else:
                    last_bright = over_upper_thresh[-1] + 1
                if under_lower_thresh.size == 0:
                    first_dark = self.per_loop
                else:
                    first_dark = under_lower_thresh[0]
                if last_bright > first_dark:
                    first_dark = last_bright
                    # To handle if there's a group of dark images between brights
                labels[self.crop_index(tweezer_num, loop_num, 0): self.crop_index(
                    tweezer_num, loop_num, last_bright)] = np.ones(last_bright)
                labels[self.crop_index(tweezer_num, loop_num, last_bright): self.crop_index(
                    tweezer_num, loop_num, first_dark)] = np.full(first_dark - last_bright, np.NaN)
                labels[self.crop_index(tweezer_num, loop_num, first_dark): self.crop_index(
                    tweezer_num, loop_num + 1, 0)] = np.zeros(self.per_loop - first_dark)
        labels = np.transpose(np.array([np.absolute(labels - 1), labels]))
        return labels, thresholds

    def make_labels_v2(self, plot=False):
        """
        Given an upper and lower threshold, classify whether lattice sites are occupied. You can read more about the
        algorithm used for classification in the documentation.
        """
        crops = self.crop_sites(1)
        thresholds = self.find_thresholds(crops, plot=plot)
        labels = np.empty(self.n_tweezers * self.n_loops * self.per_loop)
        for tweezer_num in range(self.n_tweezers):
            for loop_num in range(self.n_loops):
                loop = crops[self.crop_index(tweezer_num, loop_num, 0): self.crop_index(
                    tweezer_num, loop_num + 1, 0)]
                avg = np.mean(loop, axis=(1, 2))
                labels[self.crop_index(tweezer_num, loop_num, 0): self.crop_index(
                    tweezer_num, loop_num + 1, 0)] = self.slicer(avg, *thresholds[tweezer_num])
        labels = np.transpose(np.array([np.absolute(labels - 1), labels]))
        return labels, thresholds

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

    def find_thresholds(self, crops, plot=False, z=4.753424308822899):
        """
        For each site in the lattice, find the pixel thresholds using find_site_threshold.
        """
        if plot:
            fig, axs = plt.subplots(
                *self.lattice_shape, figsize=(4 * self.lattice_shape[0], 4 * self.lattice_shape[0]))
        thresholds = np.empty((self.n_tweezers, 2))
        for i in range(self.n_tweezers):
            avg = np.mean(crops[self.crop_index(i, 0, 0)                          : self.crop_index(i + 1, 0, 0)], axis=(1, 2))
            dark_fit, bright_fit = self.fit_gaussians(avg)
            lower_thresh = bright_fit[0] - bright_fit[1] * z
            upper_thresh = dark_fit[0] + dark_fit[1] * z
            thresholds[i] = np.array([lower_thresh, upper_thresh])
            if plot:
                counts, bins, _ = axs[i // self.lattice_shape[0]][i % self.lattice_shape[0]].hist(
                    avg, bins=(self.per_loop // 4), density=True)
                x_vals = np.linspace(bins[0], bins[-1], self.per_loop)
                axs[i // self.lattice_shape[0]][i % self.lattice_shape[0]
                                                ].plot(x_vals, double_gaussian(x_vals, *dark_fit, *bright_fit), 'k')
                axs[i // self.lattice_shape[0]][i % self.lattice_shape[0]
                                                ].axvline(lower_thresh, color='r', linestyle='--')
                axs[i // self.lattice_shape[0]][i % self.lattice_shape[0]
                                                ].axvline(upper_thresh, color='r', linestyle='--')
                axs[i // self.lattice_shape[0]][i %
                                                self.lattice_shape[0]].set_title(f"Tweezer {i}")
        return thresholds

    def fit_gaussians(self, data):
        """
        Find the pixel threshold of the possibly two Gaussian distributions in site_crops.
        The thresholds are selected to be z standard deviations above or below the mean for the 
        two distributions. z = 4.753424308822899 corresponds to a probability of 1e-6 that there exists
        a brighter crop than that corresponding to this z.
        """
        model = GMM(data)
        return model.fit()

    def plot_thresholds(self):
        return


class BlueImageProcessor(ImageProcessor):

    """
    A subclass of ImageProcessor used specifically for blue images. This class contains
    functionality for cropping images and labeling them in order to prepare datasets for neural
    network training and evaluation.
    """

    def __init__(self, stack, lattice_shape, labels):
        """
        stack : An array of images, with the first axis being the image axis and the remaining two being pixel
            intensity values.
        lattice_shape : An array with a two entries, (m, n), where m denotes how many sites there are in the
            horizontal direction and n denotes how many in the vertical. 
        labels : The occupancy of lattice sites in the images in the stack. 1 corresponds to bright, and 0 to dark.
        """
        super().__init__(stack, lattice_shape)
        self.labels = labels
        self.n_images = stack.shape[0]

    def crop_index(self, tweezer_num, img_num):
        """
        Given tweezer number, loop number, and image number, return the index that corresponds to the image
        in the crops array that would be returned by make_dataset.
        """
        return tweezer_num * self.n_images + img_num

    def make_dataset(self, n=3):
        """
        Returns cropped images that are labeled by the centeral site occupancy. The cropped images contain
        n x n lattice sites, centered on each lattice site for each image.
        """
        crops = self.crop_sites(n)
        labels = self.labels[:, self.n_tweezers // 3 + 1]
        return crops, labels


class TrainingImageProcessor(BlueImageProcessor):

    def make_dataset(self, n=3):
        return self.stack, self.labels[:, self.n_tweezers // n + 1]


def periodic_gaussian_1d(n_sites):
    def helper(x, lattice_constant, lattice_offset, std, scaling, offset):
        ans = 0
        for i in range(n_sites):
            ans += np.exp(-((x - i * lattice_constant -
                          lattice_offset) / std) ** 2 / 2)
        return scaling * ans + offset
    return helper


def periodic_gaussian_2d(n_sites, std, scaling, offset):
    # TODO
    def helper(r, a0_x, a0_y, a1_x, a1_y, offset_x, offset_y):
        ans = 0
        # for i in range()

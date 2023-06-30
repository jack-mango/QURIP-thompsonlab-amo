import numpy as np
import cv2
import matplotlib.pyplot as plt
from . import AutoGauss

import logging
log = logging.getLogger(__name__)

class ImageProcessor():
    """ 
        A base class that can be used to process lattice images and determine occupancy.
    """

    def __init__(self, stack, n_tweezers, n_loops, tweezer_positions=None):
        """
        stack : An array of images, with the first axis being the image axis and the remaining two being pixel
            intensity values.
        lattice_shape : An array with a two entries, (m, n), where m denotes how many sites there are in the
            horizontal direction and n denotes how many in the vertical. 
        """
        self.stack = stack
        self.n_tweezers = n_tweezers
        self.n_loops = n_loops
        self.per_loop = self.stack.shape[0] // n_loops
        self.img_height, self.img_width = stack.shape[1], stack.shape[2]
        if tweezer_positions is None:
            self.tweezer_positions = self.find_centroids()
        else:
            self.tweezer_positions = tweezer_positions
        self.nn_dist = self.find_nn_dist()
        self.labels = None
        self.crops = None
        self.thresholds = None

    def pixel(self, x):
        """ 
        Rounds x to the nearest integer, corresponding to the nearest pixel. x can be an array or a scalar.
        """
        return np.rint(x).astype(int)
    
    def find_tweezer_positions(self):
        img = np.mean(self.fractional_stack(8), axis=0)
        self.tweezer_positions = self.find_centroids()
        nn_dist = self.find_nn_dist()
        updated_positions = []
        guess = None
        for centroid in self.tweezer_positions:
            border = (nn_dist  + 1 )/ 2
            crop = img[self.pixel(centroid[0] - border):self.pixel(centroid[0] + border),
                        self.pixel(centroid[1] - border):self.pixel(centroid[1] + border)]
            model = AutoGauss.Gaussian2D(crop)
            params = model.fit(guess)
            if guess is None:
                guess = params
            center = np.array([self.pixel(centroid[0] - border), self.pixel(centroid[1] - border)]) + np.array(params[:2])
            updated_positions.append(center)
        return np.concatenate(updated_positions, axis=0)

    def find_centroids(self):
        """ 
        Attempt to find the locations of tweezers in an image using image processing methods
        and connected component analysis. You can find a more detailed description of the 
        algorithm used in the documentation.
        """
        frac_stack = self.fractional_stack(8)
        final = np.zeros(frac_stack.shape[1:], dtype='uint8')
        for img in frac_stack:
            img = np.copy(img).astype('uint8')
            img = cv2.GaussianBlur(img, (3, 3), 0)
            dark_params, bright_params = self.bright_dark_fit(img.flatten())
            # FIXME find a better default value to use
            thresh = -4 * np.maximum(dark_params[1], 0.6)
            img = cv2.adaptiveThreshold(
                img, 255, cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY, 9, thresh)
            n_sites, labeled_img, stats, centroids = cv2.connectedComponentsWithStats(
                img)
            for stat in stats:
                if stat[-1] <= 1:
                    img[stat[1]:stat[1] + stat[3], stat[0]:stat[0] +
                        stat[2]] = np.zeros((stat[3], stat[2]))
            final = np.maximum(img, final)
        n_sites, labeled_img, stats, centroids = cv2.connectedComponentsWithStats(
            final)
        centroids = centroids[np.argsort(stats[:, -1])]
        return centroids[-2:-self.n_tweezers - 2:-1, ::-1]

    def fractional_stack(self, n):
        """ Given n, return the first 1 / n images of each loop."""
        slices = np.empty((self.n_loops, self.img_width, self.img_height))
        for i in range(self.n_loops):
            slice_size = self.per_loop // n
            slices[i] = np.mean(
                self.stack[i * self.per_loop: i * self.per_loop + slice_size], axis=0)
        return slices

    def bright_dark_fit(self, data):
        """
        Fit a double gaussian function to the given data and return the parameters of the fit, with the lower mean Gaussian
        first and the higher mean one second. 
        """
        model = AutoGauss.GMM(data)
        return model.fit()

    def find_nn_dist(self):
        """
        Find the nearest neighbor distance given lattice positions. This can be found by finding the smallest normed distance
        between two tweezer positions, and identifying the horizontal/vertical distance between these tweezer distances. The
        nearest neighbor distance is taken to be the larger of horizontal and vertical nearest neighbor distances.
        """
        min_dist, closest_pair = self.closest_pair_distance(
            self.tweezer_positions)
        return np.max(np.absolute(np.diff(closest_pair, axis=0)))

    def closest_pair_bf(self, points):
        """ Find the two closest vectors in a list of vectors with complexity O(n^2). This code
        has been adapted from a response generated by ChatGPT.
        """
        min_distance = np.inf
        closest_pair = (None, None)
        for i in range(len(points) - 1):
            for j in range(i + 1, len(points)):
                dist = np.linalg.norm(
                    points[i] - points[j])
                if dist < min_distance:
                    min_distance = dist
                    closest_pair = (
                        points[i], points[j])
        return min_distance, closest_pair

    def closest_pair_distance(self, points):
        """
        Find the two closest vectors in a list of vectors with complexity O(n*lg(n)). This code
        has been adapted from a response generated by ChatGPT. 
        """
        if len(points) <= 3:
            return self.closest_pair_bf(points)
        points = points[points[:, 0].argsort(
        )]
        mid_point = len(points) // 2
        left_points = points[:mid_point]
        right_points = points[mid_point:]
        min_dist_left, closest_left = self.closest_pair_distance(left_points)
        min_dist_right, closest_right = self.closest_pair_distance(
            right_points)
        if min_dist_left < min_dist_right:
            min_distance = min_dist_left
            closest_pair = closest_left
        else:
            min_distance = min_dist_right
            closest_pair = closest_right
        strip_points = points[np.abs(
            points[:, 0] - points[mid_point, 0]) < min_distance]
        strip_points = strip_points[strip_points[:, 1].argsort()]
        min_dist_strip = min_distance
        closest_strip = (None, None)
        for i in range(len(strip_points)):
            j = i + 1
            while j < len(strip_points) and strip_points[j, 1] - strip_points[i, 1] < min_dist_strip:
                dist = np.linalg.norm(strip_points[i] - strip_points[j])
                if dist < min_dist_strip:
                    min_dist_strip = dist
                    closest_strip = (strip_points[i], strip_points[j])
                j += 1
        if min_dist_strip < min_distance:
            return min_dist_strip, closest_strip
        else:
            return min_distance, closest_pair

    def crop(self, x, y, h_border, v_border):
        """
        Returns the images from the stack corresponding to the pixels centered at (x, y),
        with horizontal and vertical borders of pixels corresponding to h_border and v_border.
        """
        return self.stack[:, self.pixel(x - h_border): self.pixel(x + h_border),
                          self.pixel(y - v_border): self.pixel(y + v_border), ]

    def crop_tweezer(self, n):
        """
        Given n, return an array containing a crop that includes n x n nearest neighbor distances of pixels centered
        on each tweezer in the image.
        """
        h_border = v_border = self.pixel(n * self.nn_dist / 2)
        cropped_tweezers = []
        for position in self.tweezer_positions:
            cropped_tweezers.append(self.crop(*position, h_border, v_border))
        return np.concatenate(cropped_tweezers, axis=0)

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
        crops = self.crop_tweezer(n)
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
        plt.imshow(img.T, cmap='viridis')
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
        return


class GreenImageProcessor(ImageProcessor):

    """
    A subclass of the ImageProcessor class, used specifically for processing green images. This class
    contains functionality to crop images to the correct shape and label them based on thresholding. 
    """

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
        crops = self.crop_tweezer(n)
        self.labels, self.thresholds = self.make_labels_v2(plot=plot)
        if keep_unknowns:
            return crops, self.labels
        else:
            unlabeled_images = np.isnan(self.labels[:, 1])
            log.info(f"{np.sum(unlabeled_images)} images were removed because they were unlabeled.")
            mask = ~ unlabeled_images
            return crops[mask], self.labels[mask]

    def make_labels_v1(self, plot=False):
        """
        DEPRECATED! Given an upper and lower threshold, classify whether lattice sites are occupied.
        You can read more about the algorithm used for classification in the documentation. 
        """
        crops = self.crop_tweezer(1)
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
        self.crops = self.crop_tweezer(1)
        thresholds = self.find_thresholds(self.crops, plot=plot)
        labels = np.empty(self.n_tweezers * self.n_loops * self.per_loop)
        for tweezer_num in range(self.n_tweezers):
            for loop_num in range(self.n_loops):
                loop = self.crops[self.crop_index(tweezer_num, loop_num, 0): self.crop_index(
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
                self.n_tweezers // 5 + (self.n_tweezers % 5 > 0), 5, figsize=(16, self.n_tweezers // 4))
            plt.tight_layout(pad=2)
        thresholds = np.empty((self.n_tweezers, 2))
        for i in range(self.n_tweezers):
            avg = np.mean(crops[self.crop_index(i, 0, 0): self.crop_index(i + 1, 0, 0)], axis=(1, 2))
            dark_fit, bright_fit = self.bright_dark_fit(avg)
            lower_thresh = bright_fit[0] - bright_fit[1] * z
            upper_thresh = dark_fit[0] + dark_fit[1] * z
            thresholds[i] = np.array([lower_thresh, upper_thresh])
            if plot:
                counts, bins, _ = axs[i // 5][i %
                                              5].hist(avg, bins=(self.per_loop // 4), density=True)
                x_vals = np.linspace(bins[0], bins[-1], self.per_loop)
                axs[i // 5][i %
                            5].plot(x_vals, AutoGauss.double_gaussian(x_vals, *dark_fit, *bright_fit), 'k')
                axs[i // 5][i %
                            5].axvline(lower_thresh, color='r', linestyle='--')
                axs[i // 5][i %
                            5].axvline(upper_thresh, color='r', linestyle='--')
                axs[i // 5][i % 5].set_title(f"Tweezer {i}")
        return thresholds
    
    def threshold_plot(self, tweezer_num):
        tweezer_vals = np.mean(self.crops[self.crop_index(tweezer_num, 0, 0): self.crop_index(tweezer_num + 1, 0, 0)], axis=(1, 2))
        tweezer_labels = self.labels[self.crop_index(tweezer_num, 0, 0): self.crop_index(tweezer_num + 1, 0, 0)]

        bright_mask = tweezer_labels[:, 1] == 1
        dark_mask = tweezer_labels[:, 0] == 1
        unknown_mask = np.isnan(tweezer_labels[:, 0])

        bright_indices = np.where(bright_mask)[0]
        bright_vals = tweezer_vals[bright_mask]

        dark_indices = np.where(dark_mask)[0]
        dark_vals = tweezer_vals[dark_mask]

        unknown_indices = np.where(unknown_mask)[0]
        unknown_vals = tweezer_vals[unknown_mask]

        plt.figure(figsize=(20, 10))
        plt.plot(bright_indices, bright_vals, '.', label='bright')
        plt.plot(dark_indices, dark_vals, '.', label='dark')
        plt.plot(unknown_indices, unknown_vals, 'o', label='?')
        plt.axhline(self.thresholds[tweezer_num, 1], color='r', linestyle='--', label=f"Upper Threshold = {self.thresholds[tweezer_num, 1]:.3f}")
        plt.axhline(self.thresholds[tweezer_num, 0], color='g', linestyle='--', label=f"Lower Threshold = {self.thresholds[tweezer_num, 0]:.3f}")
        plt.legend()
        plt.title(f"Tweezer Number = {tweezer_num}")
        for i in range(self.n_loops):
            plt.axvline(i * self.per_loop, color='k', linestyle='--', label="Loop Separation")
        plt.show()

class BlueImageProcessor(ImageProcessor):

    """
    A subclass of ImageProcessor used specifically for blue images. This class contains
    functionality for cropping images and labeling them in order to prepare datasets for neural
    network training and evaluation.
    """

    def __init__(self, stack, n_tweezers, n_loops, labels):
        """
        stack : An array of images, with the first axis being the image axis and the remaining two being pixel
            intensity values.
        lattice_shape : An array with a two entries, (m, n), where m denotes how many sites there are in the
            horizontal direction and n denotes how many in the vertical. 
        labels : The occupancy of lattice sites in the images in the stack. 1 corresponds to bright, and 0 to dark.
        """
        super().__init__(stack, n_tweezers, n_loops)
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
        crops = self.crop_tweezer(n)
        labels = self.labels[:, self.n_tweezers // 3 + 1]
        return crops, labels


class TrainingImageProcessor(BlueImageProcessor):

    def make_dataset(self, n=3):
        return self.stack, self.labels[:, self.n_tweezers // n + 1]

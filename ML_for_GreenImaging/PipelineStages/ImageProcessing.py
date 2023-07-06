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

    def __init__(self, stack, n_tweezers, n_loops):
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

    def run(self):
        positions = self.find_centroids()
        log.info(f"Found {positions.shape[0]} positions")
        nn_dist = self.find_nn_dist(positions)
        log.info(f"Found nearest neighbor distance to be {nn_dist:.3}")
        crops_3x3 = self.crop_tweezer(3, nn_dist, positions)
        crops_1x1 = self.crop_tweezer(1, nn_dist + 1, positions)
        info = {"Positions plot": self.plot(positions), "Positions": positions}
        return crops_3x3, crops_1x1, positions, info

    def pixel(self, x):
        """ 
        Rounds x to the nearest integer, corresponding to the nearest pixel. x can be an array or a scalar.
        """
        return np.rint(x).astype(int)
    
    def find_tweezer_positions(self):
        centroids = self.find_centroids()
        nn_dist = self.find_nn_dist(centroids)
        crops1x1 = self.crop_tweezer(1, nn_dist, centroids)
        positions = np.empty((self.n_tweezers, 2))
        weights = [] # need to get the size right; find lower right (maybe left?) corner correctly
        for i, tweezer in enumerate(crops1x1):
            first_per_loop = np.concatenate([tweezer[i:i + self.per_loop // 2] for i in range(0, self.n_loops, self.per_loop)])
            if i == 24:
                plt.imshow(np.mean(first_per_loop, axis=0))
            position, tweezer_weights = self.fit_gaussian_to_image(np.mean(tweezer, axis=0))
            positions[i] = position + centroids[i] - np.full(2, self.pixel(nn_dist / 2))
            weights.append(tweezer_weights)
        return positions

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
            model = AutoGauss.GMM(img.flatten())
            dark_params, bright_params = model.fit()
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

    def find_nn_dist(self, positions):
        """
        Find the nearest neighbor distance given lattice positions. This can be found by finding the smallest normed distance
        between two tweezer positions, and identifying the horizontal/vertical distance between these tweezer distances. The
        nearest neighbor distance is taken to be the larger of horizontal and vertical nearest neighbor distances.
        """
        min_dist, closest_pair = self.closest_pair_distance(positions)
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
        return self.stack[:, self.pixel(x - h_border): self.pixel(x + h_border) + 1,
                          self.pixel(y - v_border): self.pixel(y + v_border)+ 1]

    def crop_tweezer(self, n, nn_dist, centers):
        """
        Given n, return an array containing a crop that includes n x n nearest neighbor distances of pixels centered
        on each tweezer in the image.
        """
        h_border = v_border = self.pixel(n * nn_dist / 2)
        crops = []
        for center in centers:
            crops.append(self.crop(*center, h_border, v_border))
        return np.array(crops)

    def plot(self, positions):
        """
        Generate a plot of the lattice. If no index is provided the average of all images is taken and plotted.
        If a single index number is provided then corresponding picture of the entire lattice is plotted.
        """
        fig = plt.figure()
        img = self.stack.mean(axis=0)
        img = plt.imshow(img.T, cmap='viridis')
        plt.plot(*positions.T, 'r.')
        plt.colorbar()
        plt.title("Tweezer Positions")
        return fig

    def fit_gaussian_to_image(self, image_data):
        model = AutoGauss.Gaussian2D(image_data)
        params = model.fit()
        weights = model.func(model.make_coordinates(), 1, *params, 0)
        return np.array(params[:2]), weights

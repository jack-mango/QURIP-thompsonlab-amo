import numpy as np
import cv2
import matplotlib.pyplot as plt
from . import AutoGauss

import logging
log = logging.getLogger(__name__)

class ImageProcessor():
    """
        A class used for many image manipulation and tweezer array geometry reconstruction tasks.

    Attributes:
    - stack: brightness values for each image stored in a numpy array.
    - n_tweezers: the number of tweezers in the stack images.
    - n_loops: number of loops in the image stack.
    - per_loop: the number of images in each loop of the stack.
    - img_height: the height of a stack image in pixels.
    - img_width: the width of a stack image in pixels.
    - info: the relevant information gathered when executing this pipeline stage as a
            dictionary.
    """

    def __init__(self, stack, n_tweezers, n_loops, tweezer_positions=None):
        self.stack = stack
        self.n_tweezers = n_tweezers
        self.n_loops = n_loops
        self.per_loop = self.stack.shape[0] // n_loops
        self.img_height, self.img_width = stack.shape[-2], stack.shape[-1]
        self.positions = tweezer_positions
        self.info = {}

    def pixel(self, x):
        """
        Find the nearest pixel value(s) to x.

        Parameters:
        - x: pixel value(s). Can be a scalar or an array.

        Returns:
        The closest pixel value(s) to x.
        """
        return np.rint(x).astype(int)
    
    def find_tweezer_positions(self):
        """
        FIXME
        """
        self.positions = self.find_centroids()
        return self.positions

    def find_centroids(self):
        """
        Attempt to find the locations of tweezers in an image using image processing methods
        and connected component analysis. You can find a more detailed description of the 
        algorithm used in the documentation.

        Returns:
        The centroids associated with each tweezer in an self.n_tweezers x 2 array, where the ith 
        entry corresponds to the coordinates of the ith tweezer.
        """
        frac_stack = self.fractional_stack(8)
        frac_stack = self.to_uint8_img(frac_stack)
        final = np.zeros(frac_stack.shape[1:], dtype='uint8')
        for img in frac_stack:
            img = cv2.bilateralFilter(img, 5, 25, 5)
            model = AutoGauss.GaussianMixture(img.flatten())
            params, r_sq = model.fit()
            dark_params, bright_params = params
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
    
    def to_uint8_img(self, img):
        """
        Converts an image to the uint8 data type.

        Parameters:
            - img (numpy.ndarray): The input image to be converted.

        Returns:
            - numpy.ndarray: The converted image of uint8 data type.

        The function normalizes the input image by subtracting the minimum value and dividing by the range
        (maximum value minus minimum value). This ensures that the image values are scaled to the range [0, 1].
        Finally, the image is cast to the uint8 data type, where values are rounded and clipped to the range [0, 255].
        The resulting image has values of type uint8, suitable for image display or saving.

        Note that the input image is expected to be a numpy ndarray.
        """
        img = 255 * (img - img.min()) / (img.max() - img.min())
        return img.astype('uint8')

    def fractional_stack(self, n):
        """
        Separate the first 1/n images from the start of every loop, rounding up to the largest
        integer number of images

        Parameters:
        - n: the denomenator used in calculating what fraction of the stack
             extract.

        Returns:
        - slices: the first 1/n images from each loop in the stack. 
        """
        slices = np.empty((self.n_loops, self.img_width, self.img_height))
        slice_size = np.ceil(self.per_loop / n).astype(int)
        for i in range(self.n_loops):
            slices[i] = np.mean(
                self.stack[i * self.per_loop: i * self.per_loop + slice_size], axis=0)
        return slices

    def find_nn_dist(self):
        """
        Find the "nearest neighbor distance" given a list of coordinates. In this case this
        is taken as the larger of the two differences in x and y coordinates of the two closest points. 
        The the distance between the nearest neighbor points in the L2 distance. 

        Parameters:
        - positions: an self.n_tweezers x 2 array of coordinates corresponding to each tweezer, 
        where the ith entry corresponds to the coordinates of the ith tweezer.

        Returns:
        - nn_dist: the nearest neighbor distance based on the above criterion;
        """
        min_dist, closest_pair = self.closest_pair_distance(self.positions)
        return np.max(np.absolute(np.diff(closest_pair, axis=0)))

    def closest_pair_bf(self, points):
        """
        Given a list of points, find the two closest points using the L2 distance metric. This
        method implements a brute force approach which, if there are n points, has a cost of O(n^2).

        Parameters:
        - points: a m x 2 array of coordinates corresponding to each tweezer, 
        where the ith entry corresponds to the coordinates of the ith tweezer.

        Returns:
        - min_distance: the distance between the closest two points.
        - closest_pair: a tuple containing the coordinates of the two closest points. 
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
        Given a list of points, find the two closest points using the L2 distance metric. This
        method implements a recursive divide and conquer approach which, if there are n points, 
        has a cost of O(n log(n)).

        Parameters:
        - points: a m x 2 array of coordinates corresponding to each tweezer, 
        where the ith entry corresponds to the coordinates of the ith tweezer.

        Returns:
        - min_distance: the distance between the closest two points.
        - closest_pair: a tuple containing the coordinates of the two closest points. 
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
        Return the portion of the stack centered at the pixel corresponding to (x, y) with horizontal
        and vertical padding of pixel values corresponding to h_border and v_border respectively.

        Parameters:
        - x: x-coordinate of crop center, should be a float or integer.
        - y: y-coordinate of crop center, should be a float or integer.
        - h_border: number of pixels to add to the image on either side of the center, should be a float
                    or an integer
        - v_border: number of pixels to add to the image to the top and bottom the center, should be a float
                    or an integer

        Returns:
        - crop: an array of every crop from every image in the stack corresponding to the parameters given.
        """
        return self.stack[:, self.pixel(x - h_border): self.pixel(x + h_border) + 1,
                          self.pixel(y - v_border): self.pixel(y + v_border) + 1]

    def crop_tweezers(self, side_length, separation):
        """
        Create square crops for every tweezer in the stack, with each having a side length of 
        n x separation centered at each entry of centers

        Parameters:
        - side_length: how many nearest neighbor distances each side of the square crop should have. If
                        if the tweezers are in a rectangular lattice, then each crop would contain at most
                        side_length x side_length tweezers
        - separation: number of pixels per side_length unit

        Returns:
        - crops: an array of every crop of every tweezer. Consecutive crops in this array correspond to those 
                taken around the same center, and the same loop corresponding in the same center.
        """
        h_border = v_border = self.pixel(side_length * separation / 2)
        crops = []
        for pos in self.positions:
            crops.append(self.crop(*pos, h_border, v_border))
        return np.array(crops)

    def plot(self):
        """
        Plot the given position coordinates on top of stack averaged over all images.

        Returns:
        - fig: a matplotlib figure of the plot generated by this function
        """
        fig = plt.figure()
        img = self.stack.mean(axis=0)
        img = plt.imshow(img.T, cmap='viridis')
        plt.plot(*self.positions.T, 'ro', fillstyle='none')
        plt.colorbar()
        plt.title("Tweezer Positions")
        return fig

    def fit_gaussian_to_image(self, image_data):
        """
        Fit a two dimensional gaussian to a two dimensional array which represents pixel intensities
        for an image.

        Parameters:
        - image_data: a two dimensional array with entries represented by pixel values.

        Returns:
        - params: The parameters for the two dimensional gaussian fit
        - weights: a normalized array of the same size as the image data with values sampled calculated
                   from the fitted function, considering the pixel indicies as (x, y) coordinates.
        """
        model = AutoGauss.Gaussian2D(image_data)
        params = model.fit()
        weights = model.func(model.make_coordinates(), 1, *params, 0)
        return np.array(params[:2]), weights
    
    def position_tile_sort(self, tile_size):
        """
        Sort an array of position vectors based on which square tile they fall into, if all the vectors
        are contained within a rectangular region. Vectors in tiles closer to the origin are put in front of 
        those contained in tiles further from the origin.

        Parameters:
        - positions: an m x 2 array, containing the corrdinates of different position vectors.
        - tile_size: the side length of the square tile used to divide the rectangular region into tiles. 

        Returns:
        - sorted_vectors: an m x 2 array of the vectors sorted based on which tile they fall into. If
                          two vectors fall in the same tile, then the order is ambiguous.
        """
        num_tiles_x = self.img_width // tile_size
        tile_indices = np.floor(self.positions / tile_size).astype(int)
        tile_numbers = tile_indices[:, 1] * num_tiles_x + tile_indices[:, 0]
        sorted_indices = np.argsort(tile_numbers)
        sorted_vectors = self.positions[sorted_indices]
        self.positions = sorted_vectors
        return sorted_vectors
        
    
    

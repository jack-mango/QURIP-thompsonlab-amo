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
    """

    def __init__(self, stack, n_tweezers, n_loops):
        self.stack = stack
        self.n_tweezers = n_tweezers
        self.n_loops = n_loops
        self.per_loop = self.stack.shape[0] // n_loops
        self.img_height, self.img_width = stack.shape[1], stack.shape[2]
        self.info = None

    def run(self):
        positions = self.find_centroids()
        log.info(f"Found {positions.shape[0]} positions")
        nn_dist = self.find_nn_dist(positions)
        log.info(f"Found nearest neighbor distance to be {nn_dist:.3}")
        positions = self.position_tile_sort(positions, nn_dist)
        crops_3x3 = self.crop_tweezer(3, nn_dist, positions)
        crops_1x1 = self.crop_tweezer(1, nn_dist, positions)
        self.info = {"Positions plot": self.plot(positions), "Positions": positions}
        return crops_3x3, crops_1x1, positions, self.info

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
        Try to find the positions of the self.n_tweezers in the stack. The tweezer positions are
        the same for each image throughout the stack. You can find a more detailed description of the 
        algorithm used in the documentation.

        Returns:
        - positions: An self.n_tweezers x 2 array, where the ith entry corresponds to the coordinates of 
          the ith tweezer.
        """
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

        Returns:
        The centroids associated with each tweezer in an self.n_tweezers x 2 array, where the ith 
        entry corresponds to the coordinates of the ith tweezer.
        """
        frac_stack = self.fractional_stack(8)
        final = np.zeros(frac_stack.shape[1:], dtype='uint8')
        for img in frac_stack:
            img = np.copy(img).astype('uint8')
            img = cv2.GaussianBlur(img, (3, 3), 0)
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

    def fractional_stack(self, n):
        """
        Separate the first 1/n images from the start of every loop.

        Parameters:
        - n: the denomenator used in calculating what fraction of the stack
             extract.

        Returns:
        - slices: the first 1/n images from each loop in the stack. 
        """
        slices = np.empty((self.n_loops, self.img_width, self.img_height))
        for i in range(self.n_loops):
            slice_size = self.per_loop // n
            slices[i] = np.mean(
                self.stack[i * self.per_loop: i * self.per_loop + slice_size], axis=0)
        return slices

    def find_nn_dist(self, positions):
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
        min_dist, closest_pair = self.closest_pair_distance(positions)
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

    def crop_tweezer(self, side_length, separation, centers):
        """
        Create square crops for every tweezer in the stack, with each having a side length of 
        n x separation centered at each entry of centers

        Parameters:
        - side_length: how many nearest neighbor distances each side of the square crop should have. If
                        if the tweezers are in a rectangular lattice, then each crop would contain at most
                        side_length x side_length tweezers
        - separation: number of pixels per side_length unit
        - centers: an m x 2 array of (x, y) coordinates that specify where crops are taken from

        Returns:
        - crops: an array of every crop of every tweezer. Consecutive crops in this array correspond to those 
                taken around the same center, and the same loop corresponding in the same center.
        """
        h_border = v_border = self.pixel(side_length * separation / 2)
        crops = []
        for center in centers:
            crops.append(self.crop(*center, h_border, v_border))
        return np.array(crops)

    def plot(self, positions):
        """
        Plot the given position coordinates on top of stack averaged over all images.

        Parameters:
        - positions: an m x 2 array containing the coordinates (usually) of the tweezers.

        Returns:
        - fig: a matplotlib figure of the plot generated by this function
        """
        fig = plt.figure()
        img = self.stack.mean(axis=0)
        img = plt.imshow(img.T, cmap='viridis')
        plt.plot(*positions.T, 'r.')
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
    
    def position_tile_sort(self, positions, tile_size):
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
        num_tiles_y = self.img_height // tile_size
        tile_indices = np.floor(positions / tile_size).astype(int)
        tile_numbers = tile_indices[:, 1] * num_tiles_x + tile_indices[:, 0]
        sorted_indices = np.argsort(tile_numbers)
        sorted_vectors = positions[sorted_indices]

        return sorted_vectors
        
    
    

import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit


""" 
    A class that can be used to process lattice images and determine occupancy.
"""

class ImageProcessor():

    def __init__(self, stack, lattice_shape, n_loops, model=None):
        self.stack = stack
        self.lattice_shape = lattice_shape
        self.n_loops = n_loops
        self.per_loop = self.stack.shape[0] // n_loops
        self.n_tweezers = np.prod(lattice_shape)
        self.img_height, self.img_width = lattice_shape[0], lattice_shape[1]
        self.a0, self.a1, self.lattice_offset = self.lattice_characteristics_rect()
        self.lattice_site_positions = self.lattice_site_positions()
        self.model = model

    def pixel(self, x):
        """ Rounds scalar x to the nearest integer, corresponding to the nearest pixel."""
        return np.rint(x).astype(int)
    
    def lattice_characteristics_rect(self):
        """ Approximates the lattice constants a0 and a1, and lattice offset by fitting periodic Gaussians to the averaged stack array."""
        xdata = np.mean(self.stack, axis=(0, 1))
        ydata = np.mean(self.stack, axis=(0, 2))
        # TODO: Make the gaussian no longer have to estimate standard deviation, offset and amplitude
        # rather use this information from what can be gathered about the lattice. Amplitude and standard deviation
        # can be gained from the averaged image but offset should probably come from background average.
        guess = [self.stack.shape[1] / self.lattice_shape[0], 0, self.stack.shape[1] / self.lattice_shape[0], xdata.max() - xdata.min(), ydata.min()]
        xparams, xerr = curve_fit(periodic_gaussian_1d(self.lattice_shape[0]), np.arange(len(xdata)), xdata, p0=guess)
        yparams, yerr = curve_fit(periodic_gaussian_1d(self.lattice_shape[1]), np.arange(len(ydata)), ydata, p0=guess)
        return np.array([xparams[0], 0]), np.array([0, yparams[0]]), np.array([xparams[1], yparams[1]])

    def lattice_site_positions(self):
        """ Returns an array of the positions of the lattice sites. """
        positions = []
        # replace with np.meshgrid() for speedup
        for i in range(self.n_tweezers):
            row = i // self.lattice_shape[0]
            col = i - self.lattice_shape[0] * (i // self.lattice_shape[0])
            positions.append(self.lattice_offset + row * self.a0 + col * self.a1)
        return np.array(positions)

    def crop(self, x, y, h_border, v_border):
        """ Returns the image corresponding to the pixels centered at (x, y), with horizontal and vertical borders corresponding to
         h_border and v_border """
        return self.stack[:, self.pixel(x - h_border): self.pixel(x + h_border),
                    self.pixel(y - v_border): self.pixel(y + v_border)]
    
    def crop_sites(self, n):
        """ Given a n, the number of lattice sites to crop, return an array containing a crop centered at each lattice site in each image,
        that includes n x n lattice sites. """
        h_border, v_border = self.pixel(n * (self.a0 + self.a1) / 2)
        cropped_sites = []
        for position in self.lattice_site_positions:
            cropped_sites.append(self.crop(*position, h_border, v_border))
        return np.concatenate(tuple(cropped_sites))
    
    def crop_index(self, tweezer, loop, n):
        """ Given tweezer number, loop number, and image number (n), return the index that corresponds to the image
        in the crops array that would be returned by process. """
        return tweezer * self.n_loops * self.per_loop + loop * self.per_loop + n

    def crop_images(self):
        """ Returns a dataset of cropped images that are labeled by which loop number and which image in the loop they're in. 
            Images are labeled by [tweezer number, loop number, image number in loop]"""
        crops = self.crop_sites(3)
        labels = [np.array([i // (self.n_loops * self.per_loop),
                             np.mod(i // self.per_loop, self.n_loops),
                               np.mod(i, self.per_loop)]) for i in range(crops.shape[0])]
        return crops, labels
    
    def plot(self):
        plt.imshow(self.stack.mean(axis=0), cmap='magma')
        plt.colorbar()
        for position in self.lattice_site_positions:
            plt.plot(*position, 'ws', alpha=0.5, fillstyle='none', markersize=10)
        plt.show()
    
    def train(self):
        """ Train the neural network for this image processing using the stack. """
        return
    
    def evaluate(self):
        return
    
    def training_data(self, n_images, m):
        """ Create n_images of m x m groupings of lattice sites with the same background noise distribution, lattice constants,
          point spread function, and average number of bright and dark pixels per lattice site as the collection of images processed."""
        return
    
    def mean_dark(self):
        return
    
    def mean_bright(self):
        return
    
    def background_noise(self):
        """ Returns the average and standard deviation for pixels in the background region. The background
        region is considered to be everthing outside the bounding rectangle containing the lattice. """
        lattice_region = np.matmul(np.array([self.a0, self.a1]), self.lattice_shape)
        avg = np.mean(self.stack, axis=0)
        x_low, y_low = self.pixel(self.lattice_offset)
        x_high, y_high = self.pixel(lattice_region)
        avg[x_low: x_high, y_low: y_high] = np.full((x_high - x_low, y_high - y_low), np.NaN)
        return np.nanmean(avg), np.nanstd(avg)

    

def periodic_gaussian_1d(n_sites):
    def helper(x, lattice_constant, lattice_offset, std, scaling, offset):
        ans = 0
        for i in range(n_sites):
            ans += np.exp(-((x - i * lattice_constant - lattice_offset) / std) ** 2 / 2)
        return scaling * ans + offset
    return helper

def periodic_gaussian_2d(n_sites, std, scaling, offset):
    #TODO
    def helper(r, a0_x, a0_y, a1_x, a1_y, offset_x, offset_y):
        ans = 0
        #for i in range()

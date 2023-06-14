import numpy as np
from scipy.optimize import curve_fit

""" 
    A class that can be used to process and prepare collections of lattice images for training a neural network.
"""

class LatticeImage():

    def __init__(self, stack, lattice_shape, n_loops):
        self.stack = stack
        self.lattice_shape = lattice_shape
        self.n_loops = n_loops
        self.per_loop = self.stack.shape[0] // n_loops
        self.n_tweezers = np.prod(lattice_shape)
        self.img_height, self.img_width = lattice_shape[0], lattice_shape[1]
        self.a0, self.a1, self.lattice_offset = self.lattice_characteristics_rect()

    def pixel(self, x):
        """ Rounds px down to the nearest integer, corresponding to the nearest pixel."""
        return int(x)
    
    def lattice_characteristics_rect(self):
        """ Approximates the lattice constants a0 and a1, and lattice offset by fitting periodic Gaussians to the averaged stack array."""
        xdata = np.mean(self.stack, axis=(0, 1))
        ydata = np.mean(self.stack, axis=(0, 2))
        guess = [self.stack.shape[1] / self.lattice_shape[0], 0, self.stack.shape[1] / self.lattice_shape[0], xdata.max() - xdata.min(), ydata.min()]
        xparams, xerr = curve_fit(periodic_gaussian_1d(self.lattice_shape[0]), np.arange(len(xdata)), xdata, p0=guess)
        yparams, yerr = curve_fit(periodic_gaussian_1d(self.lattice_shape[1]), np.arange(len(ydata)), ydata, p0=guess)
        return np.array([xparams[0], 0]), np.array([0, yparams[0]]), np.array([xparams[1], yparams[1]])

    def crop(self, x, y, h_border, v_border):
        """ Returns the image corresponding to the pixels centered at (x, y), with horizontal and vertical borders corresponding to
         h_border and v_border """
        return self.stack[:, self.pixel(x - h_border): self.pixel(x + h_border),
                    self.pixel(y - v_border): self.pixel(y + v_border)]
    
    def crop_sites(self, n):
        """ Given a n, the number of lattice sites to crop, return an array containing a crop centered at each lattice site in each image,
        that includes n x n lattice sites. """
        h_border = self.pixel(n * (self.a0 + self.a1)[0] / 2)
        v_border = self.pixel(n * (self.a0 + self.a1)[1] / 2)
        cropped_sites = []
        # replace with np.meshgrid() for speedup
        for i in range(self.n_tweezers):
            row = i // self.lattice_shape[0]
            col = i - self.lattice_shape[0] * (i // self.lattice_shape[0])
            cropped_sites.append(self.crop(*(self.lattice_offset + row * self.a0 + col * self.a1), h_border, v_border))
        return np.concatenate(tuple(cropped_sites))

    def process(self):
        """ Returns a dataset of cropped images that are labeled by which loop number and which image in the loop they're in. 
            Images are labeled by [tweezer number, loop number, image number in loop]"""
        crops = self.crop_sites(3)
        labels = [np.array([i // (self.n_loops * self.per_loop), np.mod(i // self.per_loop, self.n_loops), np.mod(i, self.per_loop)]) for i in range(crops.shape[0])]
        return crops, labels


def periodic_gaussian_1d(n_sites):
    def helper(x, lattice_constant, lattice_offset, std, scaling, offset):
        ans = 0
        for i in range(n_sites):
            ans += np.exp(-((x - i * lattice_constant - lattice_offset) / std) ** 2 / 2)
        return scaling * ans + offset
    return helper

def periodic_gaussian_2d(n_sites):
    #TODO
    def helper(r, a0_x, a0_y, a1_x, a1_y, offset_x, offset_y, std, scaling, offset):
        ans = 0
        #for i in range()

def background_statistics(data, lattice_shape, a0, a1, lattice_offset):
    # Note -- doesn't have to be fast! Find a better way to crop the background. Assume the lattice isn't necessarily rectangular
    # Replace lattice region with NaNs and compute the nanMean in numpy
    bottom_buf = int(np.rint(lattice_offset[0] - lattice_constant[0]))
    left_buf = int(np.rint(lattice_offset[1] - lattice_constant[1]))
    top_buf = int(np.rint(lattice_offset[0] + lattice_shape[0] *  lattice_constant[0]))
    right_buf = int(np.rint(lattice_offset[1] + lattice_shape[1] *  lattice_constant[1]))
    bottom, left = data[:, :right_buf, :bottom_buf], data[:, right_buf:, :top_buf]
    top, right = data[:, left_buf:, top_buf:], data[:, :left_buf, bottom_buf:]
    background = np.concatenate((bottom, left, top, right), axis=None)
    return background.mean(), background.std()

def dark_statistics(data, occupancies, lattice_shape, lattice_constant, lattice_offset):
    return

def process():
    return
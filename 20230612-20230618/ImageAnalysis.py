import numpy as np
from scipy.optimize import curve_fit

def periodic_gaussian(n_sites):
    def helper(x, lattice_constant, lattice_offset, std, scaling, offset):
        ans = 0
        for i in range(n_sites):
            ans += np.exp(-((x - i * lattice_constant - lattice_offset) / std) ** 2 / 2)
        return scaling * ans + offset
    return helper

def lattice_characteristics(data, lattice_shape):
    #TODO: Split into two functions for the lattice constant and the lattice offset
    xdata = np.sum(data, axis=(0, 1)) / (data.shape[0] * data.shape[2])
    ydata = np.sum(data, axis=(0, 2)) / (data.shape[0] * data.shape[1])
    guess = [data.shape[1] / lattice_shape[0], 0, data.shape[1] / lattice_shape[0], xdata.max() - xdata.min(), ydata.min()]
    xparams, xerr = curve_fit(periodic_gaussian(lattice_shape[0]), np.arange(len(xdata)), xdata, p0=guess)
    yparams, yerr = curve_fit(periodic_gaussian(lattice_shape[1]), np.arange(len(ydata)), ydata, p0=guess)
    return np.array([xparams[0], yparams[0]]), np.array([xparams[1], yparams[1]])

def background_statistics(data, lattice_shape, lattice_constant, lattice_offset):
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

def crop(data, x, y, border):
    return data[:, int(np.rint(x - border)): int(np.rint(x + border)),
                int(np.rint(y - border)): int(np.rint(y + border))]

def crop_sites(n, data, lattice_constant, lattice_offset, lattice_shape):
    border = np.rint(n / 2 * lattice_constant.mean())
    lattice_constant = np.array([[lattice_constant[0], 0], [0, lattice_constant[1]]])
    lattice_offset = np.tile(lattice_offset, (np.prod(lattice_shape), 1))
    sites = np.array([np.array([i // lattice_shape[0], i - lattice_shape[0] * (i // lattice_shape[0])]) 
                          for i in range(np.prod(lattice_shape))])
    lattice_sites = np.matmul(sites, lattice_constant) + lattice_offset
    return np.concatenate(tuple([crop(data, site[0], site[1], border) for site in lattice_sites]))

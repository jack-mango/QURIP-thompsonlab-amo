import numpy as np
from scipy.optimize import curve_fit
from sklearn.mixture import GaussianMixture

class AutoGauss_v1():

    """ DEPRECATED """

    GAUSSIAN_BOUNDS = (
        [-np.inf, 0, -np.inf],
        [np.inf, np.inf, np.inf]
    )

    def __init__(self, x_data, y_data):
        self.x_data = x_data
        self.y_data = y_data

    def fwhm(self, x, y):
        """ Estimate the full-width a half maximum of the data. """
        half_max = x.max() / 2
        over_half_max = np.where(y < half_max)
        return x[over_half_max[0][1]] - x[over_half_max[0][0]]
    
    def fit(self):
        """ Try to fit two Gaussians to the data given in data. To do so first we try to fit the Gaussian to 
        each peak separately, then try to do a fit of both together. If the data is found to only contain
        one Gaussian, then the corresponding fit parameters are set to np.inf. """
        first_guess, second_guess = self.double_gaussians_guess()
        if second_guess[0] == np.inf:
            return first_guess, second_guess
        else:
            params, cov = curve_fit(double_gaussian, self.x_data, self.y_data,
                                     p0=[*first_guess, *second_guess])
            return params[:3], params[3:]
                   
    def double_gaussians_guess(self):
        """ Give a reasonable estimate for what the possible fit parameters could be for both Gaussians.
        Returns two arrays where the first is a guess for the taller Gaussian and the second the shorter.
        You can read the documentation for a detailed overview of how the algorithm works. """
        first_peak = self.y_data.argmax()
        intersection = self.n_consecutive_increases(self.y_data[first_peak:], 3)
        if not intersection:
            intersection = self.n_consecutive_increases(self.y_data[first_peak::-1], 3)
            if intersection:
                intersection *= -1
        if intersection:
            split = np.absolute(first_peak + intersection)
            x_first, x_second = self.x_data[:split], self.x_data[split:]
            y_first, y_second = self.y_data[:split], self.y_data[split:]
            first_std_guess = self.fwhm(x_first, y_first) / (2 * np.sqrt(2 * np.log(2)))
            second_std_guess = self.fwhm(x_second, y_second) / (2 * np.sqrt(2 * np.log(2)))
            first_params, first_cov = curve_fit(gaussian, x_first, y_first,
                                               p0=[x_first[y_first.argmax()], first_std_guess, y_first.mean()],
                                               bounds=self.GAUSSIAN_BOUNDS)
            second_params, second_cov = curve_fit(gaussian, x_second, y_second,
                                               p0=[x_second[y_second.argmax()], second_std_guess, y_second.mean()],
                                               bounds=self.GAUSSIAN_BOUNDS)
        else:
            first_params = np.full(3, np.inf)
            second_params = np.full(3, -np.inf)
        return first_params, second_params
    
    def find_local_min(self, array):
        """ Given an array, return the index of a local minimum. Does so by splitting array at maximal index,
        then searching for a minimum on either side using n_consecutive_increases. """

    def n_consecutive_increases(self, array, n):
        """ Return the index of the first occurence of n consecutive increases if one exists. If none exist
        then this method returns None."""
        for i in range(array.size - n):
            if all(array[i: i + n] == np.sort(array[i:i + n])) and np.size(np.unique(array[i: i + n])) == n:
                return i
        return
    
class GMM():

    def __init__(self, data):
        self.data = data

    def fit(self):
        model = GaussianMixture(2)
        model.fit(np.reshape(self.data, (-1, 1)))
        means = model.means_.flatten()
        stds = np.sqrt(model.covariances_.flatten())
        amplitudes = model.weights_.flatten() / (stds * np.sqrt(2 * np.pi))
        if means[0] < means[1]:
            return np.array([means[0], stds[0], amplitudes[0]]), np.array([means[1], stds[1], amplitudes[1]])
        else:
            return np.array([means[1], stds[1], amplitudes[1]]), np.array([means[0], stds[0], amplitudes[0]])
    
def gaussian(x, mean, std, a):
    return a * np.exp(- (x - mean) ** 2 / (2 * std ** 2))

def double_gaussian(x, mean_1, std_1, a_1, mean_2, std_2, a_2):
    return gaussian(x, mean_1, std_1, a_1) + gaussian(x, mean_2, std_2, a_2)
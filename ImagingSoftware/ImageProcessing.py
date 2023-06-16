import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit

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
        self.lattice_site_positions = self.lattice_site_positions()

    def pixel(self, x):
        """ Rounds x down to the nearest integer, corresponding to the nearest pixel.
          Note that x can be an array or a scalar. """
        return x.astype(int)
    
    def lattice_characteristics_rect(self):
        """ Approximates the lattice constants a0 and a1, and lattice offset by fitting periodic Gaussians
          to the averaged stack array. This algorithm assumes that the lattice has a rectangular shape. More
          about the algorithm used can be found in the documentation"""
        xdata = np.mean(self.stack, axis=(0, 1))
        ydata = np.mean(self.stack, axis=(0, 2))
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
        xerr, yerr = np.linalg.norm(np.diag(xcov)), np.linalg.norm(np.diag(ycov))
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
        return np.array([xparams[0], 0]), np.array([0, yparams[0]]), np.array([xparams[1], yparams[1]])
    
    def find_offset(self, data, guess, direction, min_err):
        """ Used in finding the lattice constants. Given data to fit to an and initial guess, this method tries
        shifting the lattice over by one site at a time, continuing if the error of the fit generated by a shift
        if lower than before. If direction should either be a 1 or -1, denoting shifting to the right or left,
        respectively. """
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
        """ Generate a reasonable guess for the fitting used in the periodic Gaussian fitting based on
        the size of the image and typical lattice parameters. """
        lattice_constant = 5.7
        lattice_offset = 1
        std = 1.2
        scaling = data.max() - data.min()
        offset = data.min()
        return [lattice_constant, lattice_offset, std, scaling, offset]
    
    def lower_bounds(self, data):
        """ Generate an absolute lower bound on reasonable fitting parameters based on the size of the 
        image and possible lattice parameters. """
        lattice_constant = 2.4
        lattice_offset = 0
        std = 0
        scaling = (data.max() - data.min()) / 4
        offset = 0
        return [lattice_constant, lattice_offset, std, scaling, offset]
    
    def upper_bounds(self, data):
        """ Generate an absolute upper bound on reasonable fitting parameters based on the size of the 
        image and possible lattice parameters. """
        lattice_constant = np.inf
        lattice_offset = np.inf
        std = self.img_width / (self.lattice_shape[0])
        scaling = np.inf
        offset = np.inf
        return [lattice_constant, lattice_offset, std, scaling, offset]


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
        """ Returns the image corresponding to the pixels centered at (x, y), with horizontal
          and vertical borders of pixels corresponding to h_border and v_border. """
        return self.stack[:, self.pixel(x - h_border): self.pixel(x + h_border),
                    self.pixel(y - v_border): self.pixel(y + v_border)]
    
    def crop_sites(self, n):
        """ Given n, return an array containing a crop that includes n x n lattice sites centered on each site. """
        h_border, v_border = self.pixel(n * (self.a0 + self.a1) / 2)
        cropped_sites = []
        print(n)
        for position in self.lattice_site_positions:
            cropped_sites.append(self.crop(*position, h_border, v_border))
        return np.concatenate(tuple(cropped_sites))
    
    def dataset_index(self, tweezer_num, loop_num, img_num):
        """ Given tweezer number, loop number, and image number, return the index that corresponds to the image
        in the crops array that would be returned by make_dataset(). """
        return tweezer_num * self.n_loops * self.per_loop + loop_num * self.per_loop + img_num

    def make_dataset(self, n=3):
        """ Returns cropped sites including n x n lattice sites for each image in the stack and each lattice site. """
        crops = self.crop_sites(n)
        return crops
    
    def plot(self, index=None):
        """ Generate a plot of the lattice. If no index is provided the average of all images is taken and plotted.
         If a single index number is provided then corresponding picture of the entire lattice is plotted. """
        if index == None:
            img = self.stack.mean(axis=0)
        else:
            img = self.stack[index]
        plt.imshow(img, cmap='magma')
        plt.colorbar()
        for position in self.lattice_site_positions:
            plt.plot(*position, 'bs', fillstyle='none')
        plt.show()

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

class GreenImageProcessor(ImageProcessor):

    """ A subclass of the ImageProcessor class, used specifically for processing green images. This class
    contains functionality to crop images to the correct shape and label them based on thresholding. """

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
        """ Given tweezer number, loop number, and image number, return the index that corresponds to the image
        in the crops array that would be returned by make_dataset. """
        return tweezer_num * self.n_loops * self.per_loop + loop_num * self.per_loop + img_num

    def make_dataset(self, lower, upper, n=3):
        """ Returns a dataset of cropped images that are labeled by which loop number and which image in the loop
          they're in. Images are labeled by occupancy of central site:
            1 -> occupied
            0 -> unoccupied
            NaN -> unknown. """
        crops = self.crop_sites(n)
        labels = self.make_labels(lower, upper)
        return crops, labels
    
    def make_labels(self, lower, upper):
        """ Given an upper and lower threshold, classify whether lattice sites are occupied. You can read more about the
        algorithm used for classification in the documentation. """
        crops = self.crop_sites(1)
        crops = np.reshape(crops, (self.n_tweezers, self.n_loops, self.per_loop, *crops.shape[-2:]))
        labels = np.empty(self.n_tweezers * self.n_loops * self.per_loop)
        for tweezer_num, tweezer in enumerate(crops):
            for loop_num, loop in enumerate(tweezer):
                # NOTE: Might need to change this to a weighted Gaussian instead of pure mean.
                avg = np.mean(loop, axis=(1, 2))
                last_bright = np.where(avg > upper)[0]
                first_dark = np.where(avg < lower)[0]
                
                first = self.crop_index(tweezer_num, loop_num, 0)
                last = self.crop_index(tweezer_num, loop_num + 1, 0)
                if last_bright.size == 0:
                    last_bright = first
                else:
                    last_bright = self.crop_index(tweezer_num, loop_num, last_bright[-1] + 1)
                if first_dark.size == 0:
                    first_dark = last
                else:
                    first_dark = self.crop_index(tweezer_num, loop_num, first_dark[0])
                if last_bright > first_dark:
                    first_dark = last_bright
                labels[first: last_bright] = np.ones(last_bright - first)
                labels[last_bright: first_dark] = np.full(first_dark - last_bright, None)
                labels[first_dark: last] = np.zeros(last - first_dark)
        return labels
                

    
class BlueImageProcessor(ImageProcessor):

    """ A subclass of ImageProcessor used specifically for blue images. This class contains
    functionality for cropping images and labeling them in order to prepare datasets for neural
    network training and evaluation. """

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
        """ Given tweezer number, loop number, and image number, return the index that corresponds to the image
        in the crops array that would be returned by make_dataset. """
        return tweezer_num * self.n_images + img_num

    def make_dataset(self, n=3):
        """ Returns cropped images that are labeled by the centeral site occupancy. The cropped images contain
          n x n lattice sites, centered on each lattice site for each image. """
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
            ans += np.exp(-((x - i * lattice_constant - lattice_offset) / std) ** 2 / 2)
        return scaling * ans + offset
    return helper

def periodic_gaussian_2d(n_sites, std, scaling, offset):
    #TODO
    def helper(r, a0_x, a0_y, a1_x, a1_y, offset_x, offset_y):
        ans = 0
        #for i in range()

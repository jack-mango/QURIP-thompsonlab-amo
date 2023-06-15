import numpy as np

"""
    A class to help with creating simulated images of lattices.
"""


class Image():

    def __init__(self, width, height, a0, a1, lattice_offset, lattice_shape, noise_mean, noise_spread,
                 n_dark, n_bright, site_spread):
        """
        width : width of image in px
        height : height of the image in px
        a0, a1 : spacing between lattice site centers in px
        lattice_offset : how much the first lattice site will be offset from (0, 0) in px
        lattice_shape : the size of the lattice -- a 4x3 lattice would correspond to (4, 3)
        noise_mean : the average background pixel value
        noise_spread : the standard deviation in the background pixel value
        #FIXME: Update
        """
        self.width, self.height = width, height
        self.a0, self.a1 = a0, a1
        self.lattice_offset = lattice_offset
        self.lattice_shape = lattice_shape
        self.noise_mean, self.noise_spread = noise_mean, noise_spread
        self.n_dark, self.n_bright = n_dark, n_bright
        self.site_spread = site_spread
        self.n_tweezers = np.prod(lattice_shape)
        self.img = np.random.normal(
            self.noise_mean, self.noise_spread, (self.width, self.height))

    def pixel(self, x):
        return int(np.rint(x))

    def add_photon(self, x, y):
        """ Add a photon to the pixel associated with the position (x, y). If the pixel corresponding to 
        the given position lies outside the image it is not included."""
        try:
            self.img[self.pixel(x), self.pixel(y)] += 1
        except IndexError:
            pass

    def populate_site(self, x, y, occupancy):
        """ Add a random number of photons around the lattice site centered at posiiton (x, y) according to a
        Poisson distribution with mean_photons as the average. The photon locations follow a Gaussian distribution
        with a standard deviation given by spread."""
        n_photons = np.random.poisson(self.n_dark + occupancy * self.n_bright)
        photons_x, photons_y = np.random.normal(
            x, self.site_spread, n_photons), np.random.normal(y, self.site_spread, n_photons)
        for x, y in zip(photons_x, photons_y):
            self.add_photon(x, y)

    def populate_image(self, site_occupancies):
        """ Add photons to each lattice site given by occupancy. The occupancy variable should be an array with the same dimensions
        as those given by the entries of lattice_shape with entries of either 0 or 1 corresponding to a filled or empty lattice site.
        Each lattice site is populated with n_dark or n_bright photons on average if it is either unoccupied or occupied respectively."""
        site_positions = []
        for i in range(self.n_tweezers):
            row = i // self.lattice_shape[0]
            col = i - self.lattice_shape[0] * (i // self.lattice_shape[0])
            site_positions.append(self.lattice_offset +
                                  np.array(row * self.a0 + col * self.a1))
        site_positions = np.array(site_positions)
        for site_pos, site_occ in zip(site_positions, site_occupancies.flatten()):
            self.populate_site(site_pos[0], site_pos[1], site_occ)
        return self.img

class ImageGenerator():
    
    def __init__(self, width, height, a0, a1, lattice_offset, lattice_shape, noise_mean, noise_spread,
                 n_dark, n_bright, site_spread):
        
        self.width, self.height = width, height
        self.a0, self.a1 = a0, a1
        self.lattice_offset = lattice_offset
        self.lattice_shape = lattice_shape
        self.noise_mean, self.noise_spread = noise_mean, noise_spread
        self.n_dark, self.n_bright = n_dark, n_bright
        self.site_spread = site_spread
        self.n_tweezers = np.prod(lattice_shape)
        

    def enumerate_occupancies(self):
        """ List all possible configurations of possible occupancies. """
        occupancies = []

        def helper(depth, occupancy):
            if depth == self.n_tweezers:
                occupancies.append(occupancy)
            else:
                empty, full = np.append(occupancy, 0), np.append(occupancy, 1)
                helper(depth + 1, empty)
                helper(depth + 1, full)
            return
        helper(0, np.empty(0))
        return np.array(occupancies)

    def make(self, n_reps):
        """ Create n_reps randomly generated images per possible occupancy configuration. """
        possible_occupancies = self.enumerate_occupancies()
        occupancies, n_images = np.tile(possible_occupancies, (n_reps, 1)), n_reps * possible_occupancies.shape[0]
        images = [Image(self.width, self.height, self.a0, self.a1, self.lattice_offset, self.lattice_shape,
                         self.noise_mean, self.noise_spread, self.n_dark, self.n_bright, self.site_spread)
                           for x in range(n_images)]
        return np.array([img.populate_image(occ) for img, occ in zip(images, occupancies)]), occupancies

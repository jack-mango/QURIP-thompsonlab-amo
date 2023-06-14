import numpy as np

"""
    A class to help with creating simulated images of lattices.
"""

class Image():

    def __init__(self, width, height, a0, a1, lattice_offset, lattice_shape, noise_mean, noise_spread):
        """
        width : width of image in px
        height : height of the image in px
        a0, a1 : spacing between lattice site centers in px
        lattice_offset : how much the first lattice site will be offset from (0, 0) in px
        lattice_shape : the size of the lattice -- a 4x3 lattice would correspond to (4, 3)
        noise_mean : the average background pixel value
        noise_spread : the standard deviation in the background pixel value
        """
        self._width = width
        self._height = height
        self._a0 = a0
        self._a1 = a1
        self._lattice_offset = lattice_offset
        self._lattice_shape = lattice_shape
        self._img = np.random.normal(noise_mean, noise_spread, (width, height))
        
    def pixel(self, px):
        return int(px)
    

    def add_photon(self, x, y):
        """ Add a photon to the pixel associated with the position (x, y). If the pixel corresponding to 
        the given position lies outside the image it is not included."""
        try:
            self._img[self.pixel(x), self.pixel(y)] += 1
        except IndexError:
            pass

    def populate_site(self, x, y, mean_photons, spread):
        """ Add a random number of photons around the lattice site centered at posiiton (x, y) according to a
        Poisson distribution with mean_photons as the average. The photon locations follow a Gaussian distribution
        with a standard deviation given by spread."""
        n_photons = np.random.poisson(mean_photons)
        photons_x, photons_y = np.random.normal(x, spread, n_photons), np.random.normal(y, spread, n_photons)
        for x, y in zip(photons_x, photons_y):
            self.add_photon(x, y)
    
    def populate_image(self, site_occupancies, spread, n_dark, n_bright):
        """ Add photons to each lattice site given by occupancy. The occupancy variable should be an array with the same dimensions
        as those given by the entries of lattice_shape with entries of either 0 or 1 corresponding to a filled or empty lattice site.
        Each lattice site is populated with n_dark or n_bright photons on average if it is either unoccupied or occupied respectively."""
        site_positions = []
        for i in range(np.prod(self._lattice_shape)):
            row = i // self._lattice_shape[0]
            col = i - self._lattice_shape[0] * (i // self._lattice_shape[0])
            site_positions.append(self._lattice_offset + np.array(row * self._a0 + col * self._a1))
        site_positions = np.array(site_positions)
        for site_pos, site_occ in zip(site_positions, site_occupancies.flatten()):
            self.populate_site(site_pos[0], site_pos[1], n_dark + site_occ * n_bright, spread)

    def get_img(self):
        return self._img
    
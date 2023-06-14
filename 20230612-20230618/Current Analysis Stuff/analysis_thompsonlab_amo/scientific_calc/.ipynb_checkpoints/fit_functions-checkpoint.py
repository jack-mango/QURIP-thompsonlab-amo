import numpy as np

from scipy.optimize import curve_fit


def linear(x, m, b):
    """
    Linear function, return m * x + b.
    """
    return m * x + b


def quadratic(x, a, x0, c):
    """
    Quadratic function, return a * (x-x0) ** 2 + c.
    """
    return a * (x - x0) ** 2 + c


def gaussian(x, a, x0, sigma, c):
    """
    Gaussian function with an offset, return a * np.exp( -(x-x0) ** 2 / (2 * sigma ** 2)) + c.
    """
    return a * np.exp(-(x - x0) ** 2 / (2 * sigma ** 2)) + c


def multi_gaussian(x, a, x0, sigma, c, n):
    """
    The sum of n Gaussian distributions. All parameters should be n-length array.
    """
    fit = 0
    for i in range(len(n)):
        fit += a[i] * np.exp(-2 * (x - x0[i]) ** 2 / (sigma[i] ** 2)) + c[i]
    return fit


def gaussian_2d(xy_mesh, amp, xc, yc, sigma_x, sigma_y, c):
    """
    Return a 2D Gaussian distribution. 'xy_mesh' should be a N-by-2 array or tuple.
    """
    (x, y) = xy_mesh  # unpack 1D list into 2D x and y coords
    gauss = amp * np.exp(-((x - xc) ** 2 / (2 * sigma_x ** 2) + (y - yc) ** 2 / (2 * sigma_y ** 2))) / (
            2 * np.pi * sigma_x * sigma_y) + c  # make the 2D Gaussian matrix
    return np.ravel(gauss)  # flatten the 2D Gaussian down to 1D


def lorentzian(x, a, x0, gamma, c):
    """
    Lorentzian line shape, return a * gamma / ((x-x0) ** 2 + (gamma/2) ** 2) + c.
    """
    return a * gamma / ((x - x0) ** 2 + (gamma / 2) ** 2) + c


def lorentzian_norm(x, a, x0, gamma, c):
    """
    Normalized Lorentzian lineshape, return a * (gamma / 2) ** 2 / ((x-x0) ** 2 + (gamma / 2) ** 2) + c.
    Thus, the center's height is a+c.
    """
    return a * (gamma / 2) ** 2 / ((x - x0) ** 2 + (gamma / 2) ** 2) + c


def cosine(t, a, f, c):
    """
    Normal sinusoidal function, return a * (cos(f * t * 2 * np.pi)) + c.
    """
    return a * (np.cos(f * t * 2 * np.pi)) + c


def rabi_oscillation(t, tau, a, f, phi, c):
    """
    Sinusoidal oscillation with an exponential decay.
    Return a * exp(-t / tau) * cos(f * t * 2 * np.pi + phi) + c
    """
    return a * np.exp(-t / tau) * (np.cos(f * t * 2 * np.pi + phi)) + c


def expon_decay(t, a, tau, c):
    return a * np.exp(-t / tau) + c


def gaussian_decay(t, a, tau, c):
    return a * np.exp(-(t / tau) ** 2) + c


def expon_decay_no_offset(t, a, tau):
    return a * np.exp(-t / tau)


def gaussian_decay_no_offset(t, a, tau):
    return a * np.exp(-(t / tau) ** 2)


def normalized_gaussian(x, x0, w, c=1):
    return np.exp(-2 * ((x[0] - x0[0]) ** 2 + (x[1] - x0[1]) ** 2) / ((c * w) ** 2))


def doubleGaussian(x, a1, a2, x1, x2, w1, w2):
    return a1 * np.exp(-2 * (x - x1) ** 2 / (w1 ** 2)) + a2 * np.exp(-2 * (x - x2) ** 2 / (w2 ** 2))


def gaussianDifference(x, fit):
    # used when solving for thresholds
    return gaussian(x, *fit[::2]) - gaussian(x, *fit[1::2])


def ramsey_fit_func(t, T2_s, delta, a, c):
    return a * alpha_td_amplitude(t, T2_s) * (np.cos(t * delta + k_phase_shift(t, T2_s))) + c


def alpha_td_amplitude(t, T2_s):
    return (1 + 0.95 * (t / T2_s) ** 2) ** (-1.5)


def k_phase_shift(t, T2_s):
    return -3 * np.arctan(0.97 * t / T2_s)


def saturation(p, a, psat, c):
    s = p / psat
    return a * s / (s + 1) + c


def bFieldShift(g, m, V, coil):
    coilFields = {'x': 7.5, 'y': 2.4, 'z': 4.3, 'q': 0.96}
    a = coilFields[coil]
    ampsPerVolt = 0.9
    I = V * ampsPerVolt
    b = a * 1.4 * (g / 2) * (m / 0.5) * I  # b field or gradient (G, G/cm)
    return b


def rabiTheory(f, f0, rabi_freq, pulse_length, rescale_factor):
    """
    Return the function of a Rabi spectrum (the concave one) given on https://en.wikipedia.org/wiki/Rabi_frequency.
    The unit of rabi_freq should be the same of f and f0 (resonance frequency). The unit of pulse length should be
    the [frequency]^-1.
    """
    
    Omega = 2 * np.pi * rabi_freq
    return (1 - (Omega ** 2 / (Omega ** 2 + (2 * np.pi * f - 2 * np.pi * f0) ** 2) *
            np.sin(0.5 * np.sqrt(Omega ** 2 + (2 * np.pi * f - 2 * np.pi * f0) ** 2) * pulse_length) ** 2)) * \
        rescale_factor


def rabiTheoryPos(f, f0, rabi_freq, pulse_length, rescale_factor):
    """
    Return the function of a Rabi spectrum (the convex one) given on https://en.wikipedia.org/wiki/Rabi_frequency.
    The unit of rabi_freq should be the same of f and f0 (resonance frequency). The unit of pulse length should be
    the [frequency]^-1.
    """
    Omega = 2 * np.pi * rabi_freq
    return ((Omega ** 2 / (Omega ** 2 + (2 * np.pi * f - 2 * np.pi * f0) ** 2) *
            np.sin(0.5 * np.sqrt(Omega ** 2 + (2 * np.pi * f - 2 * np.pi * f0) ** 2) * pulse_length) ** 2)) * \
        rescale_factor

def getR2(x, y, model, fit):
    '''
    Get the coefficient of determination R^2 of a fitting. 
    '''
    R2 = 1 - np.sum((model(x, *fit) - y)**2)/np.sum((y - np.mean(y))**2)
    return R2

def autoFitGaussian(x, y, args=None):
    x0 = x[np.argmax(y)]
    a = np.max(y) - np.min(y)
    ymin= np.min(y)
    s = np.sqrt(np.sum((y-ymin)*(x**2))/np.sum((y-ymin))-(np.sum((y-ymin)*x)/np.sum((y-ymin)))**2) # std of x if y is a PDF
    bounds=np.array([[0,np.min(x), 0, ymin-a], [2*a,np.max(x),np.inf, np.max(y)+a]])
    p0 = np.array([a, x0, s, ymin])
    if args is None:
        args = {'p0': p0, 'bounds': bounds}
    else:
        if 'p0' not in args:
            args['p0'] = p0
        if 'bounds' in args:
            args['bounds'] = [np.max([bounds[0], args['bounds'][0]], axis=0), np.min([bounds[1], args['bounds'][1]], axis=0)]
            args['p0'][args['p0'] < args['bounds'][0]] = args['bounds'][0][args['p0'] < args['bounds'][0]]
            args['p0'][args['p0'] > args['bounds'][1]] = args['bounds'][1][args['p0'] > args['bounds'][1]]
    fit, cov = curve_fit(gaussian, x, y, **args)
    return fit, cov

def autoFitGaussianNeg(x, y, args=None):
    U = [-1,1,1,-1]
    if args is not None:
        if 'p0' in args:
            args['p0'] = U*args['p0']
        if 'bounds' in args:
            args['bounds'] = np.array([[-args['bounds'][1][0], args['bounds'][0][1], args['bounds'][0][2], -args['bounds'][1][3]],
                                      [-args['bounds'][0][0], args['bounds'][1][1], args['bounds'][1][2], -args['bounds'][0][3]]])
    fit, cov = autoFitGaussian(x, -y, args=args)
    
    fit = U*fit
    U = np.diag(U)
    cov = U@cov@U
    return fit, cov
    
import numpy as np


def dbm2mw(dbm):
    """ Converting dBm to mW: mW = 10^(dBm/10)."""
    return 10 ** (dbm / 10.)


def mw2dbm(mw):
    """ Converting mW to dBm: dBm = 10*lg(mW)."""
    return 10 * np.log10(mw)


def smooth(x, num=50):
    """Return a continuous equal-spaced array between x.min() and x.max(), with total number of points = num."""
    return np.linspace(np.amin(x), np.amax(x), num)

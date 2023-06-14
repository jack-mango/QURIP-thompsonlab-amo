"""
An image stack is the images stored in one .mat file. Its shape is usually (nImagesPerLoop * nLoops, length_x,
length_y). For example, for a nLoops=20 normal green-imaging Rydberg experiment, there are three images per loop
(after loading, after rearrangement, after experiment) and the camera ROI is 120*20, then the shape of an image stack
is (60, 120, 20).

The images are in the sequence of {
1st image of the 1st loop,
2nd image of the 1st loop,
3rd image of the 1st loop,
1st image of the 2nd loop,
...
}
"""

import numpy as np
import matplotlib
import matplotlib.pyplot as plt
from skimage.feature import peak_local_max


def scaleImageStack(imageStack, cameraType='pvcam', scaleFactor=None, cameraBias=None):
    if scaleFactor is None and cameraBias is None:
        if cameraType == 'pvcam':
            scaleFactor = 0.25
            cameraBias = 100
        elif cameraType == 'nuvu':
            scaleFactor = 0.02
            cameraBias = 0
        else:
            raise ValueError('Unknow camera type: {:s}'.format(cameraType))
    photonFraction = scaleFactor
    return (imageStack.astype(int) - cameraBias) * photonFraction


def getCounts(imageStack, positions, rAtom=2, weights=None):
    """
    Return a list of counts at each position for each image. This function is often used when there's no
    rearrangement and one wants to acquire the thresholds of imaging.
    return shape: [#images, #positions]
    """
    if weights is not None:
        imageStack = imageStack * weights
    return np.array(
        [imageStack[:, p[0] - rAtom: p[0] + rAtom + 1, p[1] - rAtom: p[1] + rAtom + 1].sum(axis=(1, 2)) for p in
         positions]).T


def getBkgSubCounts(imageStack, positions, rAtom=2, weights=None, bgExcludeRegion=None):
    """
    Similar to getCounts, but instead return the background-subtracted counts,
    i.e. counts in the atom regions - average counts outside atoms' regions.
    bgExcludeRegion: [[x0, y0], [x1, y1]]
    return shape: [#images, #positions]
    """
    if weights is not None:
        imageStack = imageStack * weights
    if bgExcludeRegion is not None:
        mask = np.ones(imageStack.shape[1:], dtype=bool)
        bgr = bgExcludeRegion
        mask[bgr[0][0]:bgr[1][0], bgr[0][1]:bgr[1][1]] = False
        bg = imageStack[:, mask].mean()
    else:
        bg = 0
    nPixels = (2 * rAtom + 1) ** 2
    return getCounts(imageStack, positions, rAtom, weights) - bg * nPixels


def getCountsRegion(imageStack, region=None, weights=None):
    """
    Return the total counts of a given region for each image.
    region = [[x0, y0], [x1, y1]]
    return shape: [#images]
    """
    if weights is not None:
        imageStack = imageStack * weights
    total_counts = np.squeeze(np.array([imageStack.sum(axis=(1, 2))]))
    if region is None:
        return total_counts
    else:
        mask = np.ones(imageStack.shape[1:], dtype=bool)
        mask[region[0][0]:region[1][0], region[0][1]:region[1][1]] = False
        counts_excluded_region = np.sum(imageStack[:, mask], axis=(1, 2))
        return total_counts - counts_excluded_region


def getOccupancy(imageStack, thresholds, positions, rAtom=2, weights=None, averageOverAtoms=False):
    """
    Return the list of occupancy of atoms for each image in an image stack. This requires some given thresholds.
    thresholds: Can be either a list (of same length as positions) or a number (automatically converted to a list)
    return shape: [#images, #tweezers]
    """
    nTweezers = len(positions)
    try:  # In case the input thresholds is just a number, make it a list then.
        iter(thresholds)
    except TypeError:
        thresholds = [thresholds] * nTweezers
    counts = getCounts(imageStack, positions, rAtom, weights=weights)  # counts.shape = [#images, #positions]
    atoms = (counts > thresholds).astype(int)  # atoms.shape = [#images, #positions]
    if averageOverAtoms:
        return atoms.mean(axis=1)
    else:
        return atoms


def getOccupancyBkgSub(imageStack, thresholds, positions, rAtom=2, weights=None, averageOverAtoms=False,
                       bgExcludeRegion=None):
    """
    Return the list of occupancy of atoms for each image in an image stack. This requires some given thresholds.
    thresholds: Can be either a list (of same length as positions) or a number (automatically converted to a list)
    return shape: [#images, #tweezers]
    """
    nTweezers = len(positions)
    try:  # In case the input thresholds is just a number, make it a list then.
        iter(thresholds)
    except TypeError:
        thresholds = [thresholds] * nTweezers
    counts = getBkgSubCounts(imageStack, positions, rAtom, weights=weights, bgExcludeRegion=bgExcludeRegion)
    # [#images, #positions]
    atoms = (counts > thresholds).astype(int)  # atoms.shape = [#images, #positions]
    if averageOverAtoms:
        return atoms.mean(axis=1)
    else:
        return atoms


def getPatternOccupancy_binary(imageStack_binary, pattern, sumOverPositions=False):
    """
    Convert the occupancy (a binary image stack) into the occupancy of a pattern. The imageStack_binary should be
    of the shape [#Images, #Positions], and #Positions should be an integer times the length of pattern.
    """
    imer_len = len(pattern)
    if np.mod(np.shape(imageStack_binary)[1], imer_len) != 0:
        raise ValueError('The number of positions ({:.0f}) must be a multiple of the length of pattern ({:.0f}!'.
                         format(np.shape(imageStack_binary)[1], imer_len))
    else:
        imer_num = np.shape(imageStack_binary)[1] // imer_len
    imerStack = np.reshape(imageStack_binary, (np.shape(imageStack_binary)[0], imer_num, imer_len))
    # [#Images, #Imers, imer_len]
    patternOccupancy = (imerStack == pattern).all(axis=-1)  # [#Images, #Imers]
    if sumOverPositions:
        return np.sum(patternOccupancy, axis=-1)
    else:
        return patternOccupancy


def findTweezerPositions(imageStack, nTweezers, circleTweezers=1, rAtom=2):
    total = np.sum(imageStack, axis=0).copy()
    positions = peak_local_max(total, min_distance=1, num_peaks=nTweezers)
    positions = np.sort(positions, axis=0)
    plt.figure(figsize=(2, 4))
    plt.imshow(total, cmap='magma')
    plt.colorbar()
    ax = plt.gca()
    if circleTweezers:
        for p in positions:
            rect = matplotlib.patches.Rectangle((p[1] - rAtom - 0.5, p[0] - rAtom - 0.5), 2 * rAtom + 1, 2 * rAtom + 1,
                                                edgecolor='white', facecolor='none', linewidth=0.5)
            ax.add_patch(rect)
    ax.tick_params(axis='both', labelsize=8)
    return positions


def findTweezerPositions_v2(imageStack, nTweezers, rAtom=2, plot=True, transpose=True):
    total = np.sum(imageStack, axis=0).copy()
    total_copy = total.copy()
    positions = []
    for i in range(nTweezers):
        maximum = np.unravel_index(total.argmax(), total.shape)
        total[-rAtom + maximum[0]:rAtom + maximum[0] + 1, -rAtom + maximum[1]:rAtom + maximum[1] + 1] *= 0
        positions.append(maximum)
    positions = np.array(positions)
    positions = positions[np.argsort(positions[:, 0])]
    if plot:
        if transpose:
            total_copy = total_copy.T
        plt.figure(figsize=(2, 4))
        plt.imshow(total_copy, cmap='magma')
        plt.colorbar()
        ax = plt.gca()
        for x, y in positions:  # Real (x, y) on the camera's GUI.
            if transpose:
                x, y = y, x
            rect = matplotlib.patches.Rectangle((y - rAtom - 0.5, x - rAtom - 0.5), 2 * rAtom + 1, 2 * rAtom + 1,
                                                edgecolor='white', facecolor='none', linewidth=0.5)
            ax.add_patch(rect)
        ax.tick_params(axis='both', labelsize=8)
    return positions

def sortTweezerPositions(positions, ny, plot=False):
    nx = len(positions)/ny
    idx_y = np.argsort(positions[:, 1])
    pos_sorted = positions[idx_y]
    for row in range(ny):
        idx_x = np.argsort(pos_sorted[row*10:(row+1)*10, 0])
        pos_sorted[row*10:(row+1)*10, :] = pos_sorted[row*10+idx_x, :]
    if plot:
        plt.figure(figsize=(5,5))
        plt.plot(pos_sorted[:,0], pos_sorted[:,1],'-+')
        plt.plot(pos_sorted[0,0], pos_sorted[0,1],'o',color='red')
        plt.show()
    return pos_sorted
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
from matplotlib import  colors
from .imagestack_functions import scaleImageStack, getOccupancy, getBkgSubCounts, getOccupancyBkgSub
from .imagefile_functions_binary import getSubImageStack_binary


def convertToBinary(file, args=None):
    if file == 'resultType':
        return 'raw'
    if args['cameraType'] == 'pvcam':
        imageStack = scaleImageStack(file.load(), cameraType='pvcam')
        binary_pic = getOccupancy(imageStack, args['thresholds'], args['positions'], rAtom=args['rAtom'],
                                  weights=args['weights'], averageOverAtoms=False)
        return binary_pic.astype('int')
    elif args['cameraType'] == 'nuvu':
        imageStack = scaleImageStack(file.load(), cameraType='nuvu')
        binary_pic = getOccupancyBkgSub(imageStack, args['thresholds'], args['positions'][args['activeIdx']],
                                        rAtom=args['rAtom'], weights=None, averageOverAtoms=False,
                                        bgExcludeRegion=args['bgExcludeRegion'])
        return binary_pic.astype('int')


def getSubImageStack(file, imagesPerSequence, whichImage, cameraType):
    """
    This function help you grab the desired image stack, and will automatically help you filter out the image you want.
    If whichImage==None, then it will grab all the images. Otherwise, it will first check if the input imagesPerSequence
    is reasonable (that it can divide the number of total images), then grab the ones you want.

    whichImage can also be a list, then the function will return all a list of image stacks, indicated by the elements
    in whichImage.
    """
    imageStack = scaleImageStack(file.load(), cameraType=cameraType)
    if whichImage is not None:  # If whichImage==None, then it will sum all the images.
        try:
            iter(whichImage)
        except TypeError:
            totalImages = np.shape(imageStack)[0]
            if np.mod(totalImages, imagesPerSequence) != 0:
                raise ValueError(
                    'The image stack has {:.0f} images, while we expect {:.0f} images per sequence!'.format(
                        totalImages, imagesPerSequence))
            imageStack = imageStack[whichImage::imagesPerSequence, :, :]
        else:
            return list(map(lambda img_idx: getSubImageStack(file, imagesPerSequence, img_idx), whichImage))
    return imageStack


def cameraCounts(file, args=None):
    """
    The original name of this function was atomCounts, which is due to unpleasant historical reason. It returns the
    total counts of the images around the given positions.
    args:{
    'cameraType': 'pvcam' or 'nuvu',
    'positions': positions of tweezers,
    'rAtom': radius of the circle for each atom,
    'imagesPerSequence': the number of images per sequence,
    'whichImage': the image one want to measure (start from 0),
    'averageOverPositions': whether average over all the positions or return site-resolved results (Bool),
    'bgExcludeRegion': the region one is interested in (all other places will be considered as background),
    }
    """
    if file == 'resultType':
        return 'average'
    imageStack = getSubImageStack(file, args['imagesPerSequence'], args['whichImage'], cameraType=args['cameraType'])
    counts = getBkgSubCounts(imageStack, positions=args['positions'], rAtom=args['rAtom'], weights=None,
                             bgExcludeRegion=args['bgExcludeRegion'])
    if args['averageOverPositions']:
        return np.mean(counts)  # Averaged over both loops and positions.
    else:
        return np.mean(counts, axis=0)  # Averaged over loops, but not positions.


def cameraCountsContinuousImageBackgroundConditioned(file, args=None):
    """

    """
    if file == 'resultType':
        return 'average'
    positions, nLoops, threshold, bgExcludeRegion, rAtom = args['positions'], args['nLoops'], args['threshold'], args[
        'bgExcludeRegion'], args['rAtom']
    imageStack = scaleImageStack(file.load(), cameraType=args['cameraType'])
    nTweezers = len(positions)
    # [nLoops * #ConseqImgs, nTweezers]:
    counts = getBkgSubCounts(imageStack, positions=positions, rAtom=rAtom, bgExcludeRegion=bgExcludeRegion)

    counts = np.reshape(counts, (nLoops, -1, nTweezers))  # [nLoops, #ConseqImgs, nTweezers]
    counts_sited = []
    for tweezer in range(nTweezers):
        counts_cond = []
        for loop in range(nLoops):
            if counts[loop, 0, tweezer] > threshold:
                counts_cond.append(counts[loop, :, tweezer])
        if not counts_cond:  # If counts_cond == [], there's no atom loaded in that tweezer for all loops.
            counts_sited.append(np.nan * np.ones(np.shape(counts)[1]))
        else:
            counts_sited.append(np.mean(counts_cond, axis=0))
    counts_sited = np.array(counts_sited)

    if args['averageOverPositions']:
        return np.nanmean(counts_sited, axis=0)  # [#ConseqImgs]
    else:
        return counts_sited  # [nTweezers, #ConseqImgs]


def showPositionsExperiment(file, args):
    """
    This function helps to plot out the pictures.
     {
     'cameraType': 'pvcam' or 'nuvu',
     'positions': positions of atoms,
     'rAtom': radius of the circle for each atom,
     'imagesPerSequence': the number of images per sequence,
     'whichImage': the images one want to condition on (start from 0),
     'binaryPics': If one wants to plot the binary picture along with the actual picture (need to addBinaryPics first),
     'circle': If one wants to circle out the tweezer positions,
     'transpose': Whether transpose the picture,
     'averaged': Whether average the picture over loops,
     'ROI': [x0, y0, x1, y1],
     'plotOptions': Plot options for the actual picture (the binary one is unchangable)
    """
    positions, rAtom, averaged = args['positions'], args['rAtom'], args['averaged']
    imageStack = getSubImageStack(file, args['imagesPerSequence'], args['whichImage'], cameraType=args['cameraType'])
    nLoops = imageStack.shape[0]
    if args['ROI'] is None:
        ROI = (0, 0, imageStack.shape[1], imageStack.shape[2])
    else:
        ROI = args['ROI']
    # trim off area outside ROI
    summed = imageStack[:, ROI[0]:ROI[2], ROI[1]:ROI[3]]
    if args['transpose']:
        # first axis is loop number--leave that fixed. swap next two, which rotates the image
        summed = np.transpose(summed, (0, 2, 1))
        # now transpose ROI as well, so we can use it below to draw the circles
        ROI = [ROI[1], ROI[0], ROI[3], ROI[2]]
    # dimensions of the trimmed and transposed image
    final_dims = np.shape(summed)[1:]
    if averaged is False:
        # if we want to show singles, then reshape the array so the different shots appear along the first (row) axis
        summed = summed.reshape(nLoops * (np.shape(summed)[1]), np.shape(summed)[2])
    else:
        # if we want to show average, average over experiments
        summed = summed.sum(axis=0) / nLoops
    ratio = np.shape(summed)[0] / np.shape(summed)[1]
    if args['binaryPics']:
        binaryImageStack = getSubImageStack_binary(file, args['imagesPerSequence'], args['whichImage'])
        if not args['transpose']:
            binaryImageStack = np.reshape(binaryImageStack, (-1, 1))
        if averaged:
            binaryImageStack = np.reshape(np.mean(binaryImageStack, axis=0), (1, -1))
        fig, ax = plt.subplots(1, 2, figsize=(6, 3 * ratio))
        ax_realpic = ax[0]
        ax_binarypic = ax[1]
        ax_binarypic.imshow(binaryImageStack, vmin=0, vmax=1, cmap='Greys_r')
        ax_binarypic.set_xticks(ticks=np.arange(binaryImageStack.shape[1] + 1) - 0.5)
        ax_binarypic.set_xticklabels([])
        ax_binarypic.set_yticks(ticks=np.arange(binaryImageStack.shape[0] + 1) - 0.5)
        ax_binarypic.set_yticklabels([])
        ax_binarypic.grid(color='red', linestyle='-', linewidth=2)
        ax_binarypic.set_xlabel('Tweezers')
    else:
        fig, ax = plt.subplots(figsize=(3, 3 * ratio))
        ax_realpic = ax
    ax_realpic.imshow(summed, **args['plotOptions'])
    if args['circle']:
        # if we are not averaging, need to tile in the row direction
        reps = 1 if averaged else nLoops
        for rr in range(reps):
            for p in positions:
                xypos = (p[1] - rAtom - 0.5 - ROI[0] + rr * final_dims[0], p[0] - rAtom - 0.5 - ROI[1])
                if args['transpose']:
                    xypos = np.flip(xypos)
                rect = matplotlib.patches.Rectangle(xypos, 2 * rAtom + 1, 2 * rAtom + 1, edgecolor='white',
                                                    facecolor='none', linewidth=1)
                ax_realpic.add_patch(rect)
        ax_realpic.tick_params(axis='both', labelsize=8)
    plt.tight_layout()
    plt.show()

    return summed


def showPositionsCWImage(file, args):
    """
    This function helps to plot out the averaged picture of a CW imaging experiment.
     {
     'cameraType': 'pvcam' (or 'nuvu', extremely rare),
     'positions': positions of atoms,
     'rAtom': radius of the circle for each atom,
     'transpose': Whether transpose the picture,
     'plotOptions': Plot options for the actual picture.
    """
    positions, rAtom = args['positions'], args['rAtom']
    imageStack = scaleImageStack(file.load(), cameraType=args['cameraType'])
    averaged = imageStack.mean(axis=0)
    if args['transpose']:
        averaged = averaged.T
    ratio = np.shape(averaged)[0] / np.shape(averaged)[1]
    plt.figure(figsize=(7, 7 * ratio))
    plt.imshow(averaged, **args['plotOptions'])
    ax = plt.gca()
    for p in positions:
        xypos = (p[1] - rAtom - 0.5, p[0] - rAtom - 0.5)
        if args['transpose']:
            xypos = np.flip(xypos)
        if args['circle']:
            rect = matplotlib.patches.Rectangle(xypos, 2 * rAtom + 1, 2 * rAtom + 1, edgecolor='white', facecolor='none',
                                                linewidth=1)
            ax.add_patch(rect)
    plt.colorbar()
    ax.tick_params(axis='both', labelsize=8)

    return averaged

def showBinaryPicturesCompared(file, args):
    """
    This function helps to plot out the pictures.
     {
     'cameraType': 'pvcam' or 'nuvu',
     'imagesPerSequence': the number of images per sequence,
     'conditionalImage': the images one want to condition on (start from 0),
     'measuredImage': the images one want to condition on (start from 0),
     'transpose': Whether transpose the picture,
     'averaged': Whether average the picture over loops,
     'plotOptions': Plot options for the actual picture (the binary one is unchangable)
    """
    imageStack_cond = getSubImageStack_binary(file, args['imagesPerSequence'], args['conditionalImage'])
    imageStack_meas = getSubImageStack_binary(file, args['imagesPerSequence'], args['measuredImage'])
    nLoops = imageStack_cond.shape[0]
    if not args['transpose']:
        # first axis is loop number--leave that fixed. swap next two, which rotates the image
        imageStack_cond = np.transpose(imageStack_cond, (-1, 1))
        imageStack_meas = np.transpose(imageStack_meas, (-1, 1))
    imageStack = (imageStack_cond * imageStack_meas + imageStack_cond) / 2  # 0 = No atom, 0.5 = loss, 1 = survive
    ratio = np.shape(imageStack_cond)[0] / np.shape(imageStack_cond)[1]
    fig, ax = plt.subplots(figsize=(3, 3 * ratio))
    cmap = colors.ListedColormap(['black', 'white', 'green1'])
    ax.imshow(imageStack, vmin=0, vmax=1, cmap=cmap)
    ax.set_xticks(ticks=np.arange(imageStack.shape[1] + 1) - 0.5)
    ax.set_xticklabels([])
    ax.set_yticks(ticks=np.arange(imageStack.shape[0] + 1) - 0.5)
    ax.set_yticklabels([])
    ax.grid(color='red', linestyle='-', linewidth=2)
    ax.set_xlabel('Tweezers')
    plt.tight_layout()
    plt.show()

    return imageStack

# ---------------------------------------------Don't delete------------------------------------------------------------

# def atomCounts(file, args=None):
#     """
#     The original name of this function was atomCounts, which is due to unpleasant historical reason. It returns the
#     total counts of the images around the given positions.
#     args:{
#     'positions': positions of tweezers,
#     'thresholds': thresholds for imaging (can be an array or a number),
#     'weights': weights one applies before thresholding,
#     'rAtom': radius of the circle for each atom,
#     'imagesPerSequence': the number of images per sequence,
#     'whichImage': the image one want to measure (start from 0),
#     'sumOverPositions': whether average over all the atoms or return site-resolved results (Bool),
#     }
#     P.S. There is no 'bgExcludeRegion' yet, maybe useful to add it.
#     """
#     if file == 'resultType':
#         return 'sum'
#     imageStack = getSubImageStack(file, args['imagesPerSequence'], args['whichImage'])
#     occupancy = getOccupancy(imageStack, args['thresholds'], args['positions'], rAtom=args['rAtom'],
#                              weights=args['weights'], averageOverAtoms=False)
#     if args['sumOverPositions']:
#         return np.sum(occupancy)  # Summed over both loops and positions.
#     else:
#         return np.sum(occupancy, axis=0)  # Summed over loops, but not positions.
#
#
# def survivalProbability(file, args=None):
#     """
#     One of the most common used functions, which will return the survival probability of atoms p=n1/n0. Here, the value
#     n1 is total atoms of the measured image, and n0 is of the conditioned image. It will return both n1 and n0, and in
#     master_structure.measurement.measure they will be converted to survival rates and error bars.
#     args:{
#     'tweezerIdx': tweezer index that we are actually going to use,
#     'imagesPerSequence': the number of images per sequence,
#     'conditionedImage': the images one want to condition on (start from 0),
#     'pattern': the additional condition that specify the imer pattern of the conditioned image
#         (i.e., [1]=single-atom, [1, 1]=dimer.)
#     'measuredImage': the image one want to measure (start from 0),
#     'averageOverPositions': whether average over all the positions or return site-resolved results (Bool),
#     }
#     """
#     if file == 'resultType':
#         return 'survival'
#     imageStack_cond, imageStack_meas = getSubImageStack(file, args['imagesPerSequence'],
#                                                      [args['conditionedImage'], args['measuredImage']])
#     atomsCounts_cond = getOccupancy(imageStack_cond, args['thresholds'], args['positions'][args['tweezerIdx']],
#                                     rAtom=args['rAtom'], weights=args['weights'], averageOverAtoms=False)
#     atomsCounts_meas = getOccupancy(imageStack_meas, args['thresholds'], args['positions'][args['tweezerIdx']],
#                                     rAtom=args['rAtom'], weights=args['weights'], averageOverAtoms=False)
#
#     # imer_len = len(args['pattern'])
#     # if np.mod(len(args['tweezerIdx']), imer_len) != 0:
#     #     raise ValueError('The length of tweezerIdx ({:.0f}) must be a multiple of the length of pattern ({:.0f}!'.
#     #                      format(len(args['tweezerIdx']), imer_len))
#     # else:
#     #     atomsCounts_cond = np.reshape(atomsCounts_cond, (atomsCounts_cond.shape[0], -1, imer_len))
#     #     atomsCounts_meas = np.reshape(atomsCounts_meas, (atomsCounts_meas.shape[0], -1, imer_len))
#     #     pattern_condition = (atomsCounts_cond == args['pattern']).all(axis=2)
#     #     atomsCounts_cond = atomsCounts_cond[pattern_condition]
#     #     atomsCounts_meas = atomsCounts_meas[pattern_condition]
#
#     if args['averageOverPositions']:  # Summed over both loops and positions.
#         atomsCounts_cond = np.sum(atomsCounts_cond)
#         atomsCounts_meas = np.sum(atomsCounts_meas)
#     else:  # Summed over loops, but not positions.
#         atomsCounts_cond = np.sum(atomsCounts_cond, axis=0)
#         atomsCounts_meas = np.sum(atomsCounts_meas, axis=0)
#     return atomsCounts_meas, atomsCounts_cond

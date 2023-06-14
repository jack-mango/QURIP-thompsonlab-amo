import numpy as np


def getSubImageStack_binary(file, imagesPerSequence, whichImage):
    """
    This function help you grab the desired image stack, and will automatically help you filter out the image you want.
    If whichImage==None, then it will grab all the images. Otherwise, it will first check if the input imagesPerSequence
    is reasonable (that it can divide the number of total images), then grab the ones you want.

    whichImage can also be a list, then the function will return all a list of image stacks, indicated by the elements
    in whichImage.
    """
    binaryImageStack = file.loadBinary()
    if whichImage is not None:  # If whichImage==None, then it will sum all the images.
        try:
            iter(whichImage)
        except TypeError:
            totalImages = np.shape(binaryImageStack)[0]
            if np.mod(totalImages, imagesPerSequence) != 0:
                raise ValueError(
                    'The image stack has {:.0f} images, while we expect {:.0f} images per sequence!'.format(
                        totalImages, imagesPerSequence))
            binaryImageStack = binaryImageStack[whichImage::imagesPerSequence, :]
        else:
            return list(map(lambda img_idx: getSubImageStack_binary(file, imagesPerSequence, img_idx), whichImage))
    return binaryImageStack


def getRawOccupancy_binary(file, args=None):
    if file == 'resultType':
        return 'raw'
    return getSubImageStack_binary(file, args['imagesPerSequence'], args['whichImage'])[:, args['tweezerIdx']]


def atomCounts_binary(file, args=None):
    """
    The original name of this function was "atomCounts", which is due to unpleasant historical reason. It returns the
    total counts of the images around the given positions.
    args:{
    'imagesPerSequence': the number of images per sequence,
    'whichImage': the image one want to measure (start from 0),
    'sumOverPositions': whether average over all the atoms or return site-resolved results (Bool),
    }
    """
    if file == 'resultType':
        return 'sum'
    occupancy = getSubImageStack_binary(file, args['imagesPerSequence'], args['whichImage'])
    if args['sumOverPositions']:
        return np.sum(occupancy)  # Summed over both loops and positions.
    else:
        return np.sum(occupancy, axis=0)  # Summed over loops, but not positions.


def atomOccupancy_binary(file, args=None):
    """
    The original name of this function was "loadingProbability". It returns the number of atoms over the total number
    of tweezers.
    args:{
    'imagesPerSequence': the number of images per sequence,
    'whichImage': the image one want to measure (start from 0),
    'averageOverPositions': whether average over all the atoms or return site-resolved results (Bool),
    }
    """
    if file == 'resultType':
        return 'survival'
    occupancy = getSubImageStack_binary(file, args['imagesPerSequence'], args['whichImage'])
    nLoops = np.shape(occupancy)[0]
    nTweezers = np.shape(occupancy)[1]
    if args['averageOverPositions']:
        return np.sum(occupancy), nLoops * nTweezers  # Summed over both loops and positions.
    else:
        return np.sum(occupancy, axis=0), nLoops  # Summed over loops, but not positions.


def survivalProbability_binary(file, args=None):
    """
    One of the most common used functions, which will return the survival probability of atoms p=n1/n0. Here, the value
    n1 is total atoms of the measured image, and n0 is of the conditioned image. It will return both n1 and n0, and in
    master_structure.measurement.measure they will be converted to survival rates and error bars.
    args:{
    'tweezerIdx': tweezer index that we are actually going to use,
    'imagesPerSequence': the number of images per sequence,
    'conditionedImage': the images one want to condition on (start from 0),
    'pattern': the additional condition that specify the imer pattern of the conditioned image
        (i.e., [1]=single-atom, [1, 1]=dimer, [1, 0, 1]=trimer with the middle one empty.)
    'measImage': the image one want to measure (start from 0),
    'averageOverPositions': whether average over all the positions or return site-resolved results (Bool),
    }
    """
    if file == 'resultType':
        return 'survival'
    atomsCounts_cond, atomsCounts_meas = getSubImageStack_binary(file, args['imagesPerSequence'],
                                                                 [args['conditionedImage'], args['measuredImage']])
    atomsCounts_cond = atomsCounts_cond[:, args['tweezerIdx']]  # [#Images, #atoms]
    atomsCounts_meas = atomsCounts_meas[:, args['tweezerIdx']]  # [#Images, #atoms]

    imer_len = len(args['pattern'])
    if np.mod(len(args['tweezerIdx']), imer_len) != 0:
        raise ValueError('The length of tweezerIdx ({:.0f}) must be a multiple of the length of pattern ({:.0f}!'.
                         format(len(args['tweezerIdx']), imer_len))

    atomsCounts_cond = np.reshape(atomsCounts_cond, (atomsCounts_cond.shape[0], -1, imer_len))
    # [#Images, #Imers, imer_len]
    atomsCounts_meas = np.reshape(atomsCounts_meas, (atomsCounts_meas.shape[0], -1, imer_len))
    # [#Images, #Imers, imer_len]
    pattern_condition = (atomsCounts_cond == args['pattern']).all(axis=2)  # [#Images, #Imers]
    atomsCounts_cond = atomsCounts_cond.sum(axis=-1) * pattern_condition.astype(int)  # [#Images, #Imers]
    atomsCounts_meas = atomsCounts_meas.sum(axis=-1) * pattern_condition.astype(int)  # [#Images, #Imers]
    if args['averageOverPositions']:  # Summed over both loops and positions.
        atomsCounts_cond = np.sum(atomsCounts_cond)
        atomsCounts_meas = np.sum(atomsCounts_meas)
    else:  # Summed over loops, but not positions.
        atomsCounts_cond = np.sum(atomsCounts_cond, axis=0)
        atomsCounts_meas = np.sum(atomsCounts_meas, axis=0)
    return atomsCounts_meas, atomsCounts_cond


def patternProbability_binary(file, args=None):
    """
    This function is used usually when one are working with dimers or more. The
    args:{
    'tweezerIdx': tweezer index that we are actually going to use,
    'imagesPerSequence': the number of images per sequence,
    'conditionedImage': the images one want to condition on (start from 0),
    'measImage': the image one want to measure (start from 0),
    'conditionedPattern': the additional condition that specify the imer pattern of the conditioned image
        (i.e., [1]=single-atom, [1, 1]=dimer, [1, 0, 1]=trimer with the middle one empty),
    'measurededPattern': Similar to 'conditionedPattern', but applied on the measured image (can be a list of list,
        i.e. [[1, 0], [0, 1]] will return the probability for [1, 0] and [0, 1] respectively),
    'averageOverPositions': whether average over all the positions or return site-resolved results (Bool),
    }
    """
    if file == 'resultType':
        return 'survival'
    atomsCounts_cond, atomsCounts_meas = getSubImageStack_binary(file, args['imagesPerSequence'],
                                                                 [args['conditionedImage'], args['measuredImage']])
    atomsCounts_cond = atomsCounts_cond[:, args['tweezerIdx']]  # [#Images, #atoms]
    atomsCounts_meas = atomsCounts_meas[:, args['tweezerIdx']]  # [#Images, #atoms]

    imer_len = len(args['conditionedPattern'])
    if np.mod(len(args['tweezerIdx']), imer_len) != 0:
        raise ValueError('The length of tweezerIdx ({:.0f}) must be a multiple of the length of pattern ({:.0f}!'.
                         format(len(args['tweezerIdx']), imer_len))
    atomsCounts_cond = np.reshape(atomsCounts_cond, (atomsCounts_cond.shape[0], -1, imer_len))
    # [#Images, #Imers, imer_len]
    atomsCounts_meas = np.reshape(atomsCounts_meas, (atomsCounts_meas.shape[0], -1, imer_len))
    # [#Images, #Imers, imer_len]
    patternCounts_cond = (atomsCounts_cond == args['conditionedPattern']).all(axis=2).astype(int)  # [#Images, #Imers]
    if np.shape(args['measuredPattern']) == (imer_len,):
        patternCounts_meas = (atomsCounts_meas == args['measuredPattern']).all(axis=2).astype(int) * patternCounts_cond
        # [#Images, #Imers]
    else:
        patternCounts_meas = np.array([(atomsCounts_meas == measuredPattern).all(axis=2).astype(int)
                                       * patternCounts_cond for measuredPattern in
                                       args['measuredPattern']])  # [#Patterns, #Images, #Imers]

    if args['averageOverPositions']:  # Summed over both loops and positions.
        patternCounts_cond = np.sum(patternCounts_cond, axis=(-2, -1))
        patternCounts_meas = np.sum(patternCounts_meas, axis=(-2, -1))
    else:  # Summed over loops, but not positions.
        patternCounts_cond = np.sum(patternCounts_cond, axis=-2)
        patternCounts_meas = np.sum(patternCounts_meas, axis=-2)
    return patternCounts_meas, patternCounts_cond


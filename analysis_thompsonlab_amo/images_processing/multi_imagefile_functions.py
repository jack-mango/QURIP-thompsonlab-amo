import numpy as np
from .imagestack_functions import getPatternOccupancy_binary
from .imagefile_functions_binary import getSubImageStack_binary


def conditionalOccupancy_binary(file_list, args=None):
    """
    This function is designed for blue (399 nm) imaging, and can also be used in other places (however, in principle
    it should be applied to only binary pictures). The "file_list" should be a list of files, with the first one being
    measured, and all the other being conditions. For example, if the first file is a blue imaging, and the second file
    is the green imaging (after rearrangement), this can give the atom survival probability between these two images.
    args:{
    'tweezerIdxList': a list of the tweezer indices one want to use for each file (len = #files),
    'imagesPerSequenceList': a list of the images per sequence for each file (len = #files),
    'whichImageList': a list of which image being considered for each file (len = #files),
    'condFuncList': a list of function that will be applied on each condition, default to be None (len = #files - 1),
    'conditionLogic': either 'and' or 'or', the logic to combine all the conditions,
    'averageOverPositions': Boolean, if set to False will return site-resolved results.
    }
    """
    if file_list == 'resultType':
        return 'survival'
    for key in ['tweezerIdxList', 'imagesPerSequenceList', 'whichImageList']:
        if len(args[key]) != len(file_list):
            raise ValueError('The length of {:s} (len={:.0f}) is different from the number of '
                             'files {:.0f}!'.format(key, len(args[key]), len(file_list)))
    tweezerIdxList, imagesPerSequenceList, whichImageList = args['tweezerIdxList'], \
        args['imagesPerSequenceList'], args['whichImageList']
    file_meas = file_list[0]
    file_cond = file_list[1:]
    measurement = getSubImageStack_binary(file_meas, imagesPerSequenceList[0], whichImageList[0])[:, tweezerIdxList[0]]
    # [#Images, #Tweezers]
    conditions = [getSubImageStack_binary(file, imagesPerSequence, whichImage)[:, tweezerIdx]
                  for file, tweezerIdx, imagesPerSequence, whichImage in
                  zip(file_cond, tweezerIdxList[1:], imagesPerSequenceList[1:], whichImageList[1:])]
    # [#Conditions, #Images, #Tweezers]

    condFunc = args['condFuncList']
    if condFunc is None:
        condFunc = [lambda x: x] * len(conditions)
    if len(condFunc) != len(file_cond):
        raise ValueError('The length of condFuncList ({:.0f}) is different from number of condition files ({:.0f}), '
                         'which should be (#Files - 1)!'.format(len(condFunc), len(file_cond)))
    condition_list = []
    for i in range(len(conditions)):
        condition = np.array(condFunc[i](conditions[i]))
        if np.shape(condition) != np.shape(measurement):
            raise ValueError('The shape of condition #{:.0f} after conversion {:s} is different from the shape of the '
                             'measurement {:s}!'.format(i, str(np.shape(condition)), str(np.shape(measurement))))
        else:
            condition_list.append(condition)
    if args['conditionLogic'] == 'and':
        overall_condition = np.all(condition_list, axis=0)
    elif args['conditionLogic'] == 'or':
        overall_condition = np.any(condition_list, axis=0)
    else:
        raise ValueError('Unknown logic for conditions {:s}!'.format(args['conditionLogic']))

    measurement_conditioned = measurement * overall_condition
    if args['averageOverPositions']:
        return np.sum(measurement_conditioned), np.sum(overall_condition)
    else:
        return np.sum(measurement_conditioned, axis=-2), np.sum(overall_condition, axis=-2)


def conditionalPatternOccupancy_binary(file_list, args=None):
    """
    Similar to the function "conditionalOccupancy_binary", but instead of directly measuring the number of atoms, it
    will measure the times specific patterns appears.
    args:{
    'tweezerIdxList': a list of the tweezer indices one want to use for each file (len = #files),
    'imagesPerSequenceList': a list of the images per sequence for each file (len = #files),
    'whichImageList': a list of which image being considered for each file (len = #files),
    'patternList': a list of patterns one want to filter out for each file (len = #files), the length of each pattern
        needs to be the length of the tweezerIdx of that file divided by the same integer (i.e. the number of imers),
    'condFuncList': a list of function that will be applied on each condition, default to be None (len = #files - 1),
    'conditionLogic': either 'and' or 'or', the logic to combine all the conditions,
    'averageOverPositions': Boolean, if set to False will return site-resolved results.
    }
    """
    if file_list == 'resultType':
        return 'survival'
    for key in ['tweezerIdxList', 'imagesPerSequenceList', 'whichImageList', 'patternList']:
        if len(args[key]) != len(file_list):
            raise ValueError('The length of {:s} (len={:.0f}) is different from the number of '
                             'files {:.0f}!'.format(key, len(args[key]), len(file_list)))
    tweezerIdxList, imagesPerSequenceList, whichImageList, patternList = args['tweezerIdxList'], \
        args['imagesPerSequenceList'], args['whichImageList'], args['patternList']
    file_meas = file_list[0]
    file_cond = file_list[1:]
    measurementPic = getSubImageStack_binary(file_meas, imagesPerSequenceList[0], whichImageList[0])[:,
                     tweezerIdxList[0]]  # [#Images, #Tweezers]
    conditionPics = [getSubImageStack_binary(file, imagesPerSequence, whichImage)[:, tweezerIdx]
                     for file, tweezerIdx, imagesPerSequence, whichImage in
                     zip(file_cond, tweezerIdxList[1:], imagesPerSequenceList[1:], whichImageList[1:])]
    # [#Conditions, #Images, #Tweezers]
    measurement = getPatternOccupancy_binary(measurementPic, pattern=patternList[0])
    conditions = [getPatternOccupancy_binary(conditionPic, pattern=pattern) for conditionPic, pattern in
                  zip(conditionPics, patternList[1:])]

    condFunc = args['condFuncList']
    if condFunc is None:
        condFunc = [lambda x: x] * len(conditions)
    if len(condFunc) != len(file_cond):
        raise ValueError('The length of condFuncList ({:.0f}) is different from number of condition files ({:.0f}), '
                         'which should be (#Files - 1)!'.format(len(condFunc), len(file_cond)))
    condition_list = []
    for i in range(len(conditions)):
        condition = np.array(condFunc[i](conditions[i]))
        if np.shape(condition) != np.shape(measurement):
            raise ValueError('The shape of condition #{:.0f} after conversion {:s} is different from the shape of the '
                             'measurement {:s}!'.format(i, str(np.shape(condition)), str(np.shape(measurement))))
        else:
            condition_list.append(condition)
    if args['conditionLogic'] == 'and':
        overall_condition = np.all(condition_list, axis=0)
    elif args['conditionLogic'] == 'or':
        overall_condition = np.any(condition_list, axis=0)
    else:
        raise ValueError('Unknown logic for conditions {:s}!'.format(args['conditionLogic']))

    measurement_conditioned = measurement * overall_condition
    if args['averageOverPositions']:
        return np.sum(measurement_conditioned), np.sum(overall_condition)
    else:
        return np.sum(measurement_conditioned, axis=-2), np.sum(overall_condition, axis=-2)


def conditionalMultiImagePatternOccupancy_binary(file_list, args=None):
    """
    """
    if file_list == 'resultType':
        return 'survival'
    for key in ['tweezerIdxList', 'whichImageList', 'patternList']:
        if len(args[key]) != len(file_list):
            raise ValueError('The length of {:s} (len={:.0f}) is different from the number of '
                             'files {:.0f}!'.format(key, len(args[key]), len(file_list)))
    tweezerIdxList, whichImageList, patternList = args['tweezerIdxList'], args['whichImageList'], args['patternList']
    nLoops = args['nLoops']

    imagesPerSequenceList = []
    for i, file in enumerate(file_list):
        nImages = np.shape(file.loadBinary())[0]
        if np.mod(nImages, nLoops) != 0:
            raise ValueError('The number of images per image stack for file #{:.0f} ({:.0f}) cannot be divided by'
                             'nLoops ({:.0f})'.format(i, nImages, nLoops))
        imagesPerSequenceList.append(nImages // nLoops)

    def image_to_logics(file, imagesPerSequence, whichImage, tweezerIdx, pattern, filterLogic):
        if whichImage is None:
            return None
        elif whichImage == 'all':
            imgIncluded = list(range(imagesPerSequence))
        else:
            imgIncluded = whichImage
        imageFilteredList = []
        for img_idx in imgIncluded:
            imageStack = getSubImageStack_binary(file, imagesPerSequence, img_idx)[:, tweezerIdx]
            imageFiltered = getPatternOccupancy_binary(imageStack, pattern=pattern)
            imageFilteredList.append(imageFiltered)
        if filterLogic == 'and':
            return np.all(imageFilteredList, axis=0)
        elif filterLogic == 'or':
            return np.any(imageFilteredList, axis=0)
        else:
            raise ValueError('Unknown logic for conditions {:s}!'.format(args['conditionLogic']))

    measurement = image_to_logics(file_list[0], imagesPerSequenceList[0], whichImageList[0], tweezerIdxList[0],
                                  patternList[0], filterLogic=args['measurementLogic'])
    conditions = [image_to_logics(file_list[i], imagesPerSequenceList[i], whichImageList[i], tweezerIdxList[i],
                                  patternList[i], filterLogic=args['conditionLogic']) for i in range(1, len(file_list))]
    overall_condition = 1
    for i in range(len(conditions)):
        condition = conditions[i]
        if np.shape(condition) != np.shape(measurement):
            raise ValueError('The shape of condition #{:.0f} after conversion {:s} is different from the shape of the '
                             'measurement {:s}!'.format(i, str(np.shape(condition)), str(np.shape(measurement))))
        elif args['conditionLogic'] == 'and':
            overall_condition *= condition
        elif args['conditionLogic'] == 'or':
            overall_condition |= condition
        else:
            raise ValueError('Unknown logic for conditions {:s}!'.format(args['conditionLogic']))
    measurement_conditioned = measurement * overall_condition
    if args['averageOverPositions']:
        return np.sum(measurement_conditioned), np.sum(overall_condition)
    else:
        return np.sum(measurement_conditioned, axis=-2), np.sum(overall_condition, axis=-2)


def arbFuncConditionalMeasurement_binary(file_list, args=None):
    if file_list == 'resultType':
        return 'survival'

    if len(args['funcList']) != len(file_list):
        raise ValueError('The length of \'funcList\' (len={:.0f}) is different from the number of '
                         'files {:.0f}!'.format(len(args['funcList']), len(file_list)))
    file_meas = file_list[0]
    file_cond = file_list[1:]
    measurement = args['funcList'][0](file_meas)  # [?, #Tweezers]
    conditions = [func(file) for file, func in zip(file_cond, args['funcList'][1:])]  # [#Conditions, ?, #Tweezers]

    condition_list = []
    for i in range(len(conditions)):
        condition = conditions[i]
        if np.shape(condition) != np.shape(measurement):
            raise ValueError('The shape of condition #{:.0f} after conversion {:s} is different from the shape of the '
                             'measurement {:s}!'.format(i, str(np.shape(condition)), str(np.shape(measurement))))
        else:
            condition_list.append(condition)
    if args['conditionLogic'] == 'and':
        overall_condition = np.all(condition_list, axis=0)
    elif args['conditionLogic'] == 'or':
        overall_condition = np.any(condition_list, axis=0)
    else:
        raise ValueError('Unknown logic for conditions {:s}!'.format(args['conditionLogic']))

    measurement_conditioned = measurement * overall_condition

    if args['averageOverPositions']:
        return np.sum(measurement_conditioned), np.sum(overall_condition)
    else:
        return np.sum(measurement_conditioned, axis=-2), np.sum(overall_condition, axis=-2)

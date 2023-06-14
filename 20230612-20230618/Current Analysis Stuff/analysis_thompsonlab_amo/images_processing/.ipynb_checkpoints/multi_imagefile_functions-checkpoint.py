import numpy as np
from .imagestack_functions import getPatternOccupancy_binary
from .imagefile_functions_binary import getSubImageStack_binary


def conditionalOccupancy_binary(file_list, args=None):
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

    overall_condition = 1
    for i in range(len(conditions)):
        condition = np.array(condFunc[i](conditions[i]))
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


def conditionalPatternOccupancy_binary(file_list, args=None):
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

    overall_condition = 1
    for i in range(len(conditions)):
        condition = np.array(condFunc[i](conditions[i]))
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

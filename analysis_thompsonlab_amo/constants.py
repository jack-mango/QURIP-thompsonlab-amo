# for W/cm^2

import numpy as np

pos485_30 = np.array([[18, 12], [25, 12], [29, 12], [35, 12], [41, 12], [47, 12], [53, 11], [59, 11], [64, 11],
                      [70, 11], [76, 11], [82, 11], [88, 11], [94, 11], [99, 11], [105, 11], [111, 11], [117, 11],
                      [123, 11], [129, 11], [134, 11], [139, 11], [145, 11], [151, 10], [157, 10], [163, 10], [169, 10],
                      [174, 10], [180, 10], [186, 10]])
pos485_30[:, 1] -= 2
pos485_30[:, 0] += 1

pos485_15 = np.array([[16, 6],
                      [22, 6],
                      [27, 6],
                      [33, 6],
                      [39, 6],
                      [45, 6],
                      [51, 6],
                      [57, 6],
                      [62, 6],
                      [68, 6],
                      [74, 6],
                      [80, 6],
                      [86, 6],
                      [92, 6],
                      [97, 6]])

pos485_15_v2 = np.array([[16, 6],
                         [22, 6],
                         [26, 6],
                         [33, 6],
                         [39, 6],
                         [45, 6],
                         [51, 6],
                         [57, 6],
                         [62, 6],
                         [68, 6],
                         [74, 6],
                         [80, 6],
                         [86, 6],
                         [92, 6],
                         [97, 6]])

pos485_30v = np.array([
    [9, 19],
    [9, 25],
    [9, 31],
    [9, 37],
    [9, 43],
    [9, 49],
    [9, 54],
    [9, 60],
    [9, 66],
    [9, 72],
    [9, 78],
    [9, 84],
    [9, 90],
    [9, 95],
    [9, 101],
    [9, 107],
    [9, 113],
    [9, 119],
    [9, 125],
    [9, 131],
    [9, 137],
    [9, 143],
    [9, 149],
    [9, 155],
    [9, 161],
    [9, 167],
    [9, 173],
    [9, 179], [9, 184], [9, 190]])


pos487_15_recovered = np.array([[15, 11],
                               [20, 11],
                               [26, 11],
                               [32, 11],
                               [38, 11],
                               [43, 11],
                               [49, 11],
                               [55, 11],
                               [61, 11],
                               [66, 10],
                               [72, 11],
                               [78, 10],
                               [83, 10],
                               [89, 10],
                               [95, 10]])

pos487_15nuvu_old = np.array([[7, 8],
                              [13, 8],
                              [19, 8],
                              [25, 7],
                              [30, 7],
                              [36, 7],
                              [43, 9],
                              [48, 9],
                              [55, 9],
                              [60, 9],
                              [66, 9],
                              [72, 9],
                              [78, 9],
                              [84, 9],
                              [90, 9]])

# pos487_15_recovered = np.array([[16, 12],
#                                 [22, 12],
#                                 [27, 12],
#                                 [33, 12],
#                                 [39, 12],
#                                 [45, 11],
#                                 [51, 11],
#                                 [56, 11],
#                                 [62, 11],
#                                 [67, 11],
#                                 [73, 11],
#                                 [79, 11],
#                                 [84, 11],
#                                 [90, 11],
#                                 [96, 11]])

pos487_15nuvu_smallROI = np.array([[7, 8],
                                   [13, 8],
                                   [19, 8],
                                   [25, 7],
                                   [30, 7],
                                   [37, 7],
                                   [42, 7],
                                   [48, 7],
                                   [55, 7],
                                   [60, 7],
                                   [66, 7],
                                   [72, 7],
                                   [78, 7],
                                   [84, 7],
                                   [90, 7]])

pos487_15nuvu = np.array([[107, 8],
                          [113, 8],
                          [119, 8],
                          [125, 7],
                          [130, 7],
                          [137, 7],
                          [142, 7],
                          [148, 7],
                          [155, 7],
                          [160, 7],
                          [166, 7],
                          [172, 7],
                          [178, 7],
                          [184, 7],
                          [190, 7]])

pos487_6nuvu_9MHz = np.array([[237, 9],
                              [290, 7],
                              [343, 6],
                              [396, 6],
                              [449, 5],
                              [501, 4]])

pos487_6nuvu_9MHz_mROI = np.array([[5,   9],
                                   [15,   7],
                                   [25,   6],
                                   [35,   6],
                                   [45,   5],
                                   [55,   4]])

pos487_5nuvu_9MHz_mROI = np.array([[ 5,  8],
       [15,  7],
       [25,  7],
       [37,  6],
       [48,  6]])

# positions_10x10 = np.array([[14, 52],
#        [14, 35],
#        [14, 12],
#        [14, 29],
#        [14, 46],
#        [15, 41],
#        [15, 64],
#        [15, 24],
#        [15, 58],
#        [15, 18],
#        [20, 46],
#        [20, 41],
#        [20, 58],
#        [20, 12],
#        [20, 18],
#        [20, 35],
#        [20, 29],
#        [20, 23],
#        [20, 52],
#        [20, 64],
#        [26, 46],
#        [26, 18],
#        [26, 23],
#        [26, 12],
#        [26, 52],
#        [26, 58],
#        [26, 35],
#        [26, 64],
#        [26, 29],
#        [26, 41],
#        [31, 46],
#        [32, 23],
#        [32, 29],
#        [32, 64],
#        [32, 41],
#        [32, 58],
#        [32, 35],
#        [32, 52],
#        [32, 12],
#        [32, 18],
#        [37, 23],
#        [37, 35],
#        [37, 12],
#        [37, 58],
#        [37, 17],
#        [37, 40],
#        [37, 29],
#        [37, 63],
#        [37, 52],
#        [38, 46],
#        [43, 17],
#        [43, 35],
#        [43, 63],
#        [43, 52],
#        [43, 40],
#        [43, 46],
#        [43, 29],
#        [43, 23],
#        [43, 58],
#        [43, 12],
#        [49, 63],
#        [49, 29],
#        [49, 46],
#        [49, 12],
#        [49, 52],
#        [49, 40],
#        [49, 23],
#        [49, 17],
#        [49, 35],
#        [49, 58],
#        [54, 11],
#        [54, 34],
#        [54, 17],
#        [54, 63],
#        [54, 57],
#        [55, 52],
#        [55, 40],
#        [55, 23],
#        [55, 29],
#        [55, 46],
#        [60, 23],
#        [60, 40],
#        [60, 46],
#        [60, 34],
#        [60, 11],
#        [60, 17],
#        [60, 52],
#        [60, 57],
#        [61, 63],
#        [61, 29],
#        [66, 23],
#        [66, 17],
#        [66, 52],
#        [66, 29],
#        [66, 11],
#        [66, 40],
#        [66, 46],
#        [66, 57],
#        [66, 34],
#        [66, 63]])

positions_10x10 = np.array([[17, 12],
       [23, 12],
       [28, 12],
       [34, 12],
       [40, 12],
       [46, 12],
       [51, 12],
       [57, 12],
       [63, 12],
       [69, 11],
       [17, 18],
       [23, 18],
       [28, 18],
       [34, 18],
       [40, 18],
       [46, 18],
       [51, 18],
       [57, 17],
       [63, 17],
       [68, 17],
       [17, 24],
       [23, 24],
       [29, 23],
       [34, 24],
       [40, 23],
       [46, 23],
       [51, 23],
       [57, 23],
       [63, 23],
       [69, 23],
       [17, 29],
       [23, 30],
       [28, 29],
       [34, 29],
       [40, 29],
       [46, 29],
       [51, 29],
       [57, 29],
       [63, 29],
       [68, 29],
       [17, 35],
       [23, 35],
       [29, 35],
       [34, 35],
       [40, 35],
       [46, 35],
       [51, 35],
       [57, 35],
       [63, 34],
       [69, 34],
       [17, 41],
       [23, 41],
       [28, 41],
       [34, 41],
       [40, 41],
       [46, 41],
       [51, 41],
       [57, 40],
       [63, 40],
       [69, 40],
       [17, 47],
       [23, 47],
       [29, 46],
       [34, 46],
       [40, 46],
       [46, 46],
       [51, 46],
       [57, 46],
       [63, 46],
       [69, 46],
       [17, 53],
       [23, 52],
       [29, 52],
       [34, 52],
       [40, 52],
       [46, 52],
       [51, 52],
       [57, 52],
       [63, 52],
       [69, 51],
       [17, 58],
       [23, 58],
       [29, 58],
       [34, 58],
       [40, 58],
       [46, 58],
       [51, 58],
       [57, 58],
       [63, 57],
       [69, 57],
       [17, 64],
       [23, 64],
       [28, 64],
       [34, 64],
       [40, 64],
       [46, 63],
       [51, 64],
       [57, 63],
       [63, 63],
       [68, 63]])

bgExcludeRegion485_30 = [[7, 2], [195, 15]]
bgExcludeRegion485_20 = [[10, 5], [125, 15]]
bgExcludeRegion487_15 = [[10, 5], [100, 15]]
Ifactor = 1e3 * (np.pi * 15e-4 ** 2) / 2
scaling = 1e2 * 2
Intensity = '(W/cm$^2$)'
import os
import shutil
import numpy as np
import pickle


def deleteNuvuGlitchedImages(d_meas, nLoops, run_path, nLoops_func=None, force_remove=False, only_warnings=False):
    """
    Because we have this issue with the NUVU camera that sometimes it would simply skip an image, this function is
    to help one delete the glitched PVcam & NUVU pictures. The deleted pictures are moved to a folder named
    "trashcan" in the same run. After deleting the pictures, one need to reload the data, the function does not
    help reload it.
    
    New functions April 4th, 2023: Now the function will automatically check if nLoops seems to be given correctly,
    to avoid the trouble one might have if one gives the wrong nLoops. If |#loops of picture - nLoops| <= 3, then the
    picture will be removed to the trashcan. Otherwise, it will warn the user that the nLoops given seems to be
    wrong and one need to set force_remove=True to forcefully remove these files.

    New functions April 7th, 2023: Now the function will also be used for the case where the number fo images is
    different from file to file. One can pass a function through "nLoops_func" (i.e. lambda f: f.params['runPhaseIndex']
    * 10), which will help calculate the expected nLoops for each individual file. To avoid possible confusion, either
    one of nLoops or nLoops_func needs to be None, otherwise, it will raise an error. Moreover, I add a new parameter
    named "only_warnings". If it's set to True, the function will only warn the user if the number of images doesn't
    match the expectancy, but not actually make any deletion.
    """

    if nLoops_func is not None and nLoops is not None:
        raise ValueError('Either one of nLoops or nLoops_func needs to be None. To use the nLoops_func, '
                         'one is required to set nLoops=None to avoid possible confusion.')

    for f in d_meas.files:
        if nLoops_func is not None:
            nLoops = int(nLoops_func(f))

        if np.shape(f.load())[0] != nLoops:
            if not os.path.exists(os.path.join(run_path, 'trashcan')):
                os.makedirs(os.path.join(run_path, 'trashcan'))
            nuvu_filename = f.filename
            pvcam_filename = 'camera' + nuvu_filename[7:]
            if not os.path.exists(os.path.join(run_path, 'nuvu', nuvu_filename)):
                print('File fIdx={:.0f} only has {:.0f} images but is no longer found in the folder (the given nLoops '
                      'is {:.0f}), possibly already been deleted or moved to the trash can!'
                      .format(f.params['fIdx'], np.shape(f.load())[0], nLoops))
            else:
                if np.abs(np.shape(f.load())[0] - nLoops) <= 3 or force_remove:
                    if not only_warnings:
                        shutil.move(os.path.join(run_path, 'nuvu', nuvu_filename), os.path.join(run_path, 'trashcan'))
                        shutil.move(os.path.join(run_path, 'pvcam', pvcam_filename), os.path.join(run_path, 'trashcan'))
                    print('File fIdx={:.0f} only has {:.0f} images while the given nLoops is {:.0f}, moving the '
                          'corresponding files to the trash can!'.format(f.params['fIdx'], np.shape(f.load())[0],
                                                                         nLoops))
                else:
                    print('File fIdx={:.0f} has {:.0f} images, which is significantly different from the nLoops given '
                          '({:.0f}). Please double check whether the nLoops is given correctly, '
                          'or set force_remove=True to forcefully remove the file to the trash.'
                          .format(f.params['fIdx'], np.shape(f.load())[0], nLoops))


def deleteNuvuGlitchedImagesBool(d_meas, delete_condition, run_path, only_warnings=False):
    """
    Similar to the function "deleteNuvuGlitchedImages". However, here the condition for deletion is given by the
    function passed through "delete_condition". Give a file f in d_meas, if delete_condition(f) == True, than the
    corresponding files will be deleted. After deleting the pictures, one need to reload the data, the function does not
    help reload it.
    """
    for f in d_meas.files:
        if delete_condition(f):
            if not os.path.exists(os.path.join(run_path, 'trashcan')):
                os.makedirs(os.path.join(run_path, 'trashcan'))
            nuvu_filename = f.filename
            pvcam_filename = 'camera' + nuvu_filename[7:]
            if not os.path.exists(os.path.join(run_path, 'nuvu', nuvu_filename)):
                print('File fIdx={:.0f} (nImages = {:.0f}) satisfies the condition but cannot be found, possibly '
                      'already been deleted or moved to the trash can!'.format(f.params['fIdx'], np.shape(f.load())[0]))
            else:
                if not only_warnings:
                    shutil.move(os.path.join(run_path, 'nuvu', nuvu_filename), os.path.join(run_path, 'trashcan'))
                    shutil.move(os.path.join(run_path, 'pvcam', pvcam_filename), os.path.join(run_path, 'trashcan'))
                print('File fIdx={:.0f} (nImages = {:.0f}) satisfies the condition, moving the '
                      'corresponding files to the trash can!'.format(f.params['fIdx'], np.shape(f.load())[0]))


def getRBdepth(filename):
    with open(filename, 'rb') as file:
        SRBseq = pickle.load(file)
    depth_list = []
    for seq in SRBseq:
        depth = 0
        for phase in seq:
            if phase == 'sven_gate':
                depth += 1
        depth_list.append(depth)
    return np.array(depth_list)


def group(x, y, yerr):
    unique_x, inv_ndx = np.unique(x, return_inverse=True)
    sum_y = np.bincount(inv_ndx, weights=y)
    sum_y2 = np.bincount(inv_ndx, weights=y ** 2)
    sum_yerr2 = np.bincount(inv_ndx, weights=yerr ** 2)
    n_y = np.bincount(inv_ndx)
    if n_y[-1] <= 2:
        unique_x = unique_x[:-1]
        sum_y = sum_y[:-1]
        sum_y2 = sum_y2[:-1]
        sum_yerr2 = sum_yerr2[:-1]
        n_y = n_y[:-1]
    mean_y = sum_y / n_y
    std_y = np.sqrt((sum_y2 / n_y - mean_y ** 2) / (n_y - 1))
    err_y = np.sqrt(sum_yerr2) / n_y
    return np.array(unique_x), np.array(mean_y), np.array(np.sqrt(std_y ** 2 + err_y ** 2))


    
import os
import re
import glob
import pickle
import numpy as np
import scipy.io as sio
from tqdm.notebook import tqdm
from .images_processing.imagefile_functions import convertToBinary


class file:

    def __init__(self, filename, path=''):
        self.filename = filename
        self.filename_no_extension = os.path.splitext(self.filename)[0]

        self.path = path
        self.binary_path = os.path.join(path, 'binary_pics')

        self.fullname = os.path.join(self.path, self.filename)
        self.fullname_no_extension = os.path.splitext(self.fullname)[0]

        self.binary_fullname = os.path.join(self.binary_path, self.filename_no_extension + '.npy')
        self.loadedDict = 0
        self.params = dict([(y[0], float(y[1])) for y in re.findall('(\w+)=([\-?\d.]+)', self.fullname_no_extension)])
        self.params['fIdx'] = int(self.params['fIdx'])
        self.params['all'] = True  # add dummy parameter to allow grabbing all datasets
        self.rawdata = None
        self.binarydata = None

    def load(self):
        """ load the data, but only if we haven't already """
        if self.rawdata is None:
            # print(sio.loadmat(self.fullname))
            #             self.rawdata = sio.loadmat(self.fullname)['images']
            self.rawdata = sio.loadmat(self.fullname)['stack']
        return self.rawdata

    def loadBinary(self):
        """ load the binary data, but only if we haven't already """
        if self.binarydata is None:
            self.binarydata = np.load(self.binary_fullname)
        return self.binarydata

    def matchParams(self, keys, vals):
        """ check if params specified in keys match vals """
        for k, v in zip(keys, vals):
            if self.params[k] != v:
                return False
        return True


class dataset:
    """ dataset is a container that holds all the data files relevant to some experiment.
    It can load entire directories or individual files, and will not reload/duplicate files it already has."""

    def __init__(self, relpath, prefix=''):
        self.files = []
        self.allParamsDict = []
        self.paramsLoaded = []
        self.addDir(relpath, prefix)

    def addDir(self, relpath, prefix=''):
        """
        add all files in a directory, specified wrt. a prefix.
        Should be able to call addDir repeatedly on a directory that is being added to, and only new files will
        be updated...
        """
        path = os.path.join(prefix, relpath)
        os.chdir(path)
        fileList = sorted(glob.glob('*.mat'))
        for f in fileList:
            self.addFile(f, path)
        os.chdir(prefix)

    def addFile(self, filename, path=''):
        """ add a single file, checking if we already have it """
        matches = list(filter(lambda x: x.path == path and x.filename == filename, self.files))
        if len(matches) == 0:
            # if not, add it
            self.files.append(file(filename, path=path))

    def addParams(self, relpath, prefix=''):
        path = os.path.join(prefix, relpath)
        os.chdir(path + '\\paramsFolder')
        for f in glob.glob('*.mat'):
            if f not in self.paramsLoaded:
                self.allParamsDict.append(sio.loadmat(f))
                self.paramsLoaded.append(f)
        os.chdir(prefix)
        return None

    def saveBinaryPics(self, binaryConversionArgs, relpath, prefix='', convert_all=False):
        binary_path = os.path.join(prefix, relpath, 'binary_pics')
        if not os.path.exists(binary_path):
            os.makedirs(binary_path)
        os.chdir(binary_path)
        binary_pic_list = sorted(glob.glob('*.npy'))
        pbar = tqdm(total=len(self.files), desc='Checking and converting all pictures into binary')
        for f in self.files:
            pbar.update()
            binary_filename = f.filename_no_extension + '.npy'
            if binary_filename not in binary_pic_list or convert_all:
                binary_pic = convertToBinary(f, binaryConversionArgs)
                np.save(binary_filename, binary_pic)
        pbar.close()
        with open(binary_path + '\\thresholds_params', 'wb') as f_args:
            pickle.dump(dict(binaryConversionArgs), f_args)

    def getfIdxList(self, includefn=lambda x: True):
        fIdx_list = []
        for f in self.files:
            if includefn(f):
                fIdx_list.append(f.params['fIdx'])
        return fIdx_list
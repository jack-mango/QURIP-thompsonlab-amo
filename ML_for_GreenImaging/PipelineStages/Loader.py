import os
import numpy as np
from tensorflow.keras import models
from scipy.io import loadmat

import logging
log = logging.getLogger(__name__)

class Loader():

    """
    Loads data files and potential model files to train a neural network or
    classify image data.

    Attributes:
    - n_tweezers: number of tweezers in the image.
    - n_loops: number of loops per image file.
    - data_path: absolute path of a folder containing some image data.
    - model_path: absolute path of a folder containing the model to be used
     later in the pipeline. If no .h5 file for the model exists at model_path,
     then a new one is created later in the pipeline and saved to model_path.
    """


    def __init__(self, n_tweezers, n_loops, data_path, model_path):
        self.n_tweezers = n_tweezers
        self.n_loops = n_loops
        self.data_path = data_path
        self.model_path = model_path

    def run(self):
        """
        Execute this class's methods to return information used later in the pipeline.
        
        Returns:
        - stack: brightness values for each image stored in a numpy array.
        - tot_loops: the total number of loops across all data files. For example
                     if there are 20 loops in each file and four files, then the tot_loops
                     is 80. 
        - model: an instance of a tensorflow.keras sequential neural network.
        """
        stack, n_files = self.load_data()
        log.info(f"Found {n_files} data files in {self.data_path}")
        tot_loops = self.n_loops * n_files
        model = self.load_model()
        if not self.model_path is None:
            log.info(f"Found model at {self.model_path}")
        return stack, tot_loops, model

    def load_data(self):
        """
        Load .mat files from self.data_path if that directory exists.

        Returns:
        - stack: brightness values for each image stored in a numpy array.
                 The last two axes of this array hold the row and column pixel
                 values for each image.
        - n_files: the number of .mat files found at self.data_path
        """
        n_files = 0
        if os.path.isdir(self.data_path):
            stack = []
            for i, file in enumerate(os.listdir(self.data_path)):
                if file.endswith('.mat'):
                    data = loadmat(self.data_path + '/' + file)
                    stack.append(data['stack'])     
                    n_files += 1
            stack = np.concatenate(stack)
        else:
            stack = loadmat(self.data_path)['stack']
            n_files += 1  
        return stack, n_files
        
    def load_model(self):
        """
        Load an .h5 model from self.model_path if one exists there.

        Returns:
        - model: an instance of a tensorflow.keras sequential neural network.
        """
        if os.path.exists(self.model_path):
            model = models.load_model(self.model_path)
        else:
            model = None
        return model
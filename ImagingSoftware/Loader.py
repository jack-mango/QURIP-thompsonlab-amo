import os
import numpy as np
from tensorflow.keras import models
from scipy import loadmat

class Loader():

    """
    I take in the path and I make a dataset for training. I can also report statistics about the dataset.
    """

    def __init__(self, n_loops, n_tweezers, data_path, model_path=None):
        self.data_path = data_path
        self.model_path = model_path
        self.n_tweezers = n_tweezers
        self.n_loops = n_loops

    def run(self):
        stack, n_files = self.load_data()
        tot_loops = self.n_loops * n_files
        model = self.load_model()
        return stack, tot_loops, model

    def load_data(self):
        if os.path.isdir(self.data_path):
            stack = []
            for i, file in enumerate(os.listdir(self.data_path)):
                if file.endswith('.mat'):
                    stack.append(loadmat(self.data_path))
            return np.concatenate(stack), i + 1
        else:
            return loadmat(self.data_path), 1
        
    def load_model(self):
        if self.model_path is None:
            return
        else:
            return models.load_model(self.model_path)
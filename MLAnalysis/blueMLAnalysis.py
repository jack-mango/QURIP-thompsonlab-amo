from PipelineStages import *
import matplotlib.pyplot as plt
from tensorflow.keras import models
import logging
import numpy as np
import os
import cv2
from scipy.io import loadmat

class blueMLAnalysis():

    DEFAULT = {
        'validation_split': 0.2,
        'epochs': 8
    }

    def __init__(self, n_tweezers, n_loops, tweezer_positions=None, make_plots=True):
        self.n_tweezers = n_tweezers
        self.n_loops = n_loops
        self.img_processor = ImageProcessing.ImageProcessor(n_tweezers, n_loops,
                                                             tweezer_positions=tweezer_positions, make_plots=make_plots)
        self.make_plots = make_plots
        self.model = None

    def find_tweezers(self, pv_stack, nuvu_stack):
        nuvu_positions = self.img_processor.transform_positions(pv_stack, nuvu_stack)
        self.img_processor.find_nn_dist()
        return nuvu_positions

    def train_model(self, data_path, green_model_path, blue_model_path, training_kwargs=DEFAULT):
        nuvu_stack, pv_stack = self.load_training_data(data_path)
        green_model = models.load_model(green_model_path)
        self.model = models.load_model(blue_model_path)
        self.find_tweezers(pv_stack, nuvu_stack)
        labels = self.make_labels(pv_stack)
        crops = self.make_crops()
        history = self.model.fit(crops, labels, **training_kwargs)
        self.model.save(blue_model_path)
        return history
    
    def get_occupancies(self, stack):
        crops = self.crop_images(stack, 3)
        occupancies = self.model.predict(crops)
        return np.argmax(occupancies, axis=1)
    
    def fidelity_analysis(self, stack):
        occupancies = self.get_occupancies(stack)
        occupancies = np.reshape(occupancies, (self.n_tweezers, self.n_loops, -1))
        first_diff = np.diff(occupancies, axis=2)
        n_dark_to_bright = np.sum(first_diff == -1, axis=(1, 2))
        n_dark = np.sum(occupancies[:,:,:-1] == 0, axis=(1, 2))
        n_bright_to_dark = np.sum(first_diff == 1, axis=(1, 2))
        n_bright = np.sum(occupancies[:,:,:-1] == 1, axis=(1, 2))
        prob_db, prob_bd = n_dark_to_bright / n_dark, n_bright_to_dark / n_bright
        fig = plt.figure(figsize=(12.8, 4.8))
        plt.bar(np.arange(self.n_tweezers), prob_bd, label=f'Bright to Dark Probability', color='orange')
        plt.bar(np.arange(self.n_tweezers), prob_db, label=f'Dark to Bright Probability', color='steelblue')
        plt.axhline(prob_bd.mean(), label=f"Bright to Dark Average={prob_bd.mean():.3}", color='darkorange', linestyle='--')
        plt.axhline(prob_db.mean(), label=f"Dark to Bright Average={prob_db.mean():.3}", color='dodgerblue', linestyle='--')
        plt.xlabel('Tweezer Number')
        plt.ylabel('Probability')
        plt.legend(loc='upper left')
        plt.title('Fidelity')
        plt.show()
        return prob_db, prob_bd, fig

    def load_training_data(self, path):
        """
        Load .mat files from self.data_path if that directory exists.

        Returns:
        """
        pv_stack = []
        nuvu_stack = []
        for file in os.listdir(os.path.join(path, 'nuvu')):
            if file.endswith('.mat'):
                name = file.split(',', 1)
                if os.path.exists(os.path.join(path, 'pvcam', f"camera,{name[1]}")):
                    nuvu_stack.extend(loadmat(os.path.join(path, 'nuvu', file))['stack'])
                    pv_stack.extend(loadmat(os.path.join(path, path, 'pvcam', f"camera,{name[1]}")['stack']))
        nuvu_stack = np.array(nuvu_stack)
        pv_stack = np.array(pv_stack)
        gcd = np.gcd(pv_stack.shape[0], nuvu_stack.shape[0])
        pv_per_nuvu = pv_stack.shape[0] // gcd
        nuvu_per_pv = nuvu_stack.shape[0] // gcd
        pv_start = np.ceil(pv_per_nuvu / 2) - 1
        nuvu_start = np.ceil(nuvu_per_pv / 2) - 1
        return pv_stack[pv_start::pv_per_nuvu], nuvu_stack[nuvu_start::nuvu_per_pv]
    
    def load_data(self, path):
        return

    def make_labels(self, pv_stack, green_model):
        self.img_processor.find_tweezer_positions(pv_stack)
        pv_crops = self.img_processor.crop_tweezers(pv_stack, 3)
        labels = green_model.predict(np.reshape(pv_crops, (-1, *pv_crops.shape[-2:])))
        labels = np.argmax(labels, axis=1)
        return np.transpose(np.array([np.absolute(labels - 1), labels]))
    
    def make_crops(self, nuvu_stack):
        crops_3x3 = self.img_processor.crop_tweezers(nuvu_stack, 3)
        return np.reshape(crops_3x3, (-1, *crops_3x3.shape[-2:]))
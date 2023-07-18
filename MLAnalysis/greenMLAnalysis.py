from PipelineStages import *
import matplotlib.pyplot as plt
from tensorflow.keras import models
import logging
import numpy as np
import os
from scipy.io import loadmat

class greenMLAnalysis():

    DEFAULT = {
        'validation_split': 0.1,
        'epochs': 8
    }

    def __init__(self, n_tweezers, n_loops, tweezer_positions=None, make_plots=True):
        self.n_tweezers = n_tweezers
        self.n_loops = n_loops
        self.img_processor = ImageProcessing.ImageProcessor(n_tweezers, n_loops,
                                                             tweezer_positions=tweezer_positions, make_plots=make_plots)
        self.make_plots = make_plots
        self.model = None

    def find_tweezers(self, stack):
        stack = np.reshape(stack, (-1, *stack.shape[-2:]))
        positions = self.img_processor.find_tweezer_positions(stack)
        self.img_processor.find_nn_dist()
        return positions

    def train_model(self, data_path, model_path, training_kwargs=DEFAULT):
        stack = self.load_data(data_path)
        self.model = models.load_model(model_path)
        self.find_tweezers(stack)
        labels = self.make_labels(stack)
        crops = self.make_crops(stack)
        crops, labels = self.filter_unlabeled(crops, labels)
        history = self.model.fit(crops, labels, **training_kwargs)
        self.model.save(model_path)
        return history
    
    def get_occupancies(self, data_path):
        stack = self.load_data(data_path)
        stack = np.concatenate(stack)
        crops = self.make_crops(np.reshape(stack, (1, -1, *stack.shape[-2:])))
        occupancies = self.model.predict(crops)
        return np.argmax(occupancies, axis=1)
    
    def fidelity_analysis(self, data_path):
        occupancies = self.get_occupancies(data_path)
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

    def load_data(self, path):
        """
        Load .mat files from self.data_path if that directory exists.

        Returns:
        - stack: brightness values for each image stored in a numpy array. The first axis
                corresponds ot file number and the second image number within the file.
                The last two axes of this array hold the row and column pixel values for 
                each image.
        - n_files: the number of .mat files found at self.data_path
        """
        if os.path.isdir(path):
            stack = []
            for file in os.listdir(path):
                if file.endswith('.mat'):
                    data = loadmat(path + '/' + file)
                    stack.append(data['stack'])     
            stack = np.array(stack)
        else:
            stack = np.array(loadmat(path)['stack'])
        return stack

    def make_labels(self, stack):
        per_tweezer_file = stack.shape[1]
        n_files = stack.shape[0]
        labels = np.empty((self.n_tweezers * n_files * per_tweezer_file))
        for i, file_stack in enumerate(stack):
            crops_1x1 = self.img_processor.crop_tweezers(file_stack, 1)
            labeler = Labeler.Labeler(crops_1x1, self.n_tweezers, self.n_loops, make_plots=self.make_plots)
            bright_dark_fits, r_sq = labeler.bright_dark_fit()
            thresholds, plots = labeler.find_thresholds(bright_dark_fits)
            all_below_upper, all_above_lower = labeler.threshold_misfits(thresholds)
            labels[i * self.n_tweezers * per_tweezer_file:
                   (i  + 1) * self.n_tweezers * per_tweezer_file] = labeler.make_labels(thresholds)
        return np.transpose(np.array([np.absolute(labels - 1), labels]))
    
    def make_crops(self, stack):
        crop_size = 2 * np.rint(1.5 * self.img_processor.nn_dist).astype(int) + 1
        per_tweezer_file = stack.shape[1]
        n_files = stack.shape[0]
        crops_3x3 = np.empty((self.n_tweezers * n_files * per_tweezer_file, crop_size, crop_size))
        for i, file_stack in enumerate(stack):
            crops = self.img_processor.crop_tweezers(file_stack, 3)
            crops_3x3[i * self.n_tweezers * per_tweezer_file:
                      (i + 1) * self.n_tweezers * per_tweezer_file] = np.reshape(crops, (-1, crop_size, crop_size))
        return np.reshape(crops_3x3, (-1, *crops_3x3.shape[-2:]))
    
    def filter_unlabeled(self, crops, labels):
        unlabeled_images = np.isnan(labels[:, 1])
        mask = ~ unlabeled_images
        return crops[mask], labels[mask]
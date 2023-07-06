import numpy as np
from tensorflow.keras import layers, models, optimizers

import logging
log = logging.getLogger(__name__)

class DatasetBuilder():

    def __init__(self, crops, labels, testing_fraction=0.1):
        self.crops = crops.reshape((-1, *crops.shape[2:]))
        self.labels = labels
        self.testing_fraction = testing_fraction

    def run(self):
        crops, labels = self.filter_unlabeled()
        testing_indices = np.random.choice(crops.shape[0], int(crops.shape[0] * self.testing_fraction), replace=False)
        labels = np.transpose(np.array([np.absolute(labels - 1), labels]))
        training_crops = np.delete(crops, testing_indices, axis=0)
        training_labels = np.delete(labels, testing_indices, axis=0)
        testing_crops = crops[testing_indices]
        testing_labels = labels[testing_indices]
        info = self.dataset_statistics(training_labels, testing_labels)
        for key, val in info.items():
            log.info(f"{key}: {val}")
        return (training_crops, training_labels), (testing_crops, testing_labels), self.crops, info

    def dataset_statistics(self, training_labels, testing_labels):
        info = {"Number of Training Images": training_labels.shape[0],
                "Number of Testing Images": testing_labels.shape[0],
                "Training Bright Fraction": self.bright_fraction(training_labels),
                "Testing Bright Fraction": self.bright_fraction(testing_labels),
                "Labeled Dataset Fraction": 1 - np.sum(np.isnan(self.labels)) /self.crops.shape[0]
                }
        return info
    
    def filter_unlabeled(self):
        unlabeled_images = np.isnan(self.labels)
        mask = ~ unlabeled_images
        return self.crops[mask], self.labels[mask]
    
    def bright_fraction(self, labels):
        """ Training first, then testing"""
        n_bright = np.sum(labels[:, 1])
        return n_bright / labels.shape[0]
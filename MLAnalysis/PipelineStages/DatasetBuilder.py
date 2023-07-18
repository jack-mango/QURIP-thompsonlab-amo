import numpy as np
from tensorflow.keras import layers, models, optimizers

import logging
log = logging.getLogger(__name__)

class DatasetBuilder():

    """
        A class containing the methods related to the final formatting needed to prepare a dataset
        to be trained and tested with a machien learning model.

    Attributes:
    - crops: a three dimensional array, where the first axis corresponds to the image number and
        the two remaining correspond to the x and y axes for the pixel values in an image.
    - labels: an array containing either zero for dark images, one for bright, and NaNs for unlabeled.
    - testing_fraction: A float between zero and one used to determine what fraction of data will be returned
                        by run as testing data.
    """

    def __init__(self, crops, labels, testing_fraction=0.2):
        self.crops = crops.reshape((-1, *crops.shape[-2:]))
        self.labels = labels
        self.testing_fraction = testing_fraction

    def dataset_info(self, training, testing):
        """
        Calculate some useful information about the contents of the training and testing datasets.

        Parameters:
        - training: a tuple with the first entry as an array of crops and the second as an array of labels
        - testing: a tuple with the first entry as an array of crops and the second as an array of labels

        Returns:
        - info: a dictionary containing useful information about data collected from this pipeline stage
        """
        info = {"Number of Training Images": training[0].shape[0],
                "Number of Testing Images": testing[0].shape[0],
                "Training Bright Fraction": self.bright_fraction(training[1]),
                "Testing Bright Fraction": self.bright_fraction(testing[1]),
                "Labeled Dataset Fraction": 1 - np.sum(np.isnan(self.labels)) /self.crops.shape[0]
                }
        return info
    
    def filter_unlabeled(self, index_data=None):
        """
        Return a copy of the dataset with all unlabeled data removed. A crop is considered unlabeled if its 
        corresponding label is NaN.

        Returns:
        - labeled_crops: a three dimensional array, where the first axis corresponds to the image number and
        the two remaining correspond to the x and y axes for the pixel values in an image. 
        - labels: an array containing only those labels that are bright (1) or dark (0).
        """
        unlabeled_images = np.isnan(self.labels)
        mask = ~ unlabeled_images
        if index_data is None:
            return self.crops[mask], self.labels[mask]
        else:
            return self.crops[mask], self.labels[mask], index_data[mask]
    
    def bright_fraction(self, labels):
        """
        Calculate what fraction of the labels have a value of one, corresponding to a bright crop classification. 

        Parameters:
        - labels: a one dimensional array containing only ones and zeros, corresponding to whether or not a tweezer
                  is labeled as dark or bright.

        Returns:
        - bright_frac: the fraction of labels that are equal to one divided by the total number of labels
        """
        return np.mean(labels)
    
    def testing_training_split(self, crops, labels):
        """
        Split the crops and labels arrays into two separate arrays for testing and training, with the size of each
        determined by the specified testing_fraction. Additionally turns the one dimensional label array into a two
        dimensional array: the first axis still corresponds to crop number but the second axis now corresponds to bright/dark
        label. A bright crop will be given the label [1, 0] while a dark one will be given [0, 1]. 

        Parameters:
        - crops: a three dimensional array where the first axis corresponds to the image number and the remaining two
                correspond to the x and y axes for pixel values in the image.
        - labels: a one dimensional array of zeros or ones; should have the same length as crops in the first axis

        Returns:
        - training: a tuple where the first entry is an array of all crops to be used for testing and the second contains
                    a one dimensional array of the labels for training. 
        - testing: a tuple with the same ordering as training, but to be used for testing instead.
        """
        labels = np.transpose(np.array([np.absolute(labels - 1), labels]))
        testing_indices = np.random.choice(crops.shape[0], int(crops.shape[0] * self.testing_fraction), replace=False)
        training_crops = np.delete(crops, testing_indices, axis=0)
        training_labels = np.delete(labels, testing_indices, axis=0)
        testing_crops = crops[testing_indices]
        testing_labels = labels[testing_indices]
        return (training_crops, training_labels), (testing_crops, testing_labels)
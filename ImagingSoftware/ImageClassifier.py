import tensorflow as tf
from tensorflow.keras import datasets, layers, models, optimizers

class ImageClassifier():

    def __init__(self, model):
        self.model = model

    def train(self, dataset, labels=None):
        """ Train the neural network for this dataset processing using the provided labels. """
        return

    def classify(self, dataset):
        """ Use the Image classifier's model to evaluate the occupancy of the images in dataset. """
        return
    
    def save_model(self, filename):
        """ Save the model for later use at filename. """
        return
    
    def load_model(self, filename):
        """ Load a model from filename. """
        return
    
    
    
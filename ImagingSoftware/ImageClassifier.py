import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow.keras import datasets, layers, models, optimizers

class ImageClassifier():

    def __init__(self, model=None, filename=None):
        """
            model : an instance of an already compiled tensorflow neural network
        """
        if filename: 
            self.load_model(filename)
        else:
            self.model = model

    def train(self, training_images, training_labels, epochs=10):
        """ Train the neural network for this dataset processing using the provided labels. """
        self.model.fit(training_images, training_labels, epochs=epochs)
        return self.model.fit(training_images, training_labels, epochs=epochs)
    
    def evaluate(self, testing_images, testing_labels):
        self.model.evaluate(testing_images, testing_labels)

    def classify(self, images):
        """ Use the Image classifier's model to evaluate the occupancy of the given images. """
        return self.model.predict(images)
    
    def save_model(self, filename):
        """ Save the model for later use at filename. """
        self.model.save(filename)
        return
    
    def load_model(self, filename):
        """ Load a model from filename. """
        self.model = models.load_model(filename)
        return
    
    def fidelity_analysis(self, images, n_tweezers, n_loops, plot=False):
        """ Return probability of bright to dark and dark to bright of each tweezer. """
        labels = np.reshape(np.argmax(self.classify(images), axis=1), (n_tweezers, n_loops, -1))
        first_diff = np.diff(labels, axis=2)
        n_dark_to_bright = np.sum(first_diff == -1, axis=(1, 2))
        n_dark = np.sum(labels[:,:,:-1] == 0, axis=(1, 2))
        n_bright_to_dark = np.sum(first_diff == 1, axis=(1, 2))
        n_bright = np.sum(labels[:,:,:-1] == 1, axis=(1, 2))
        p_db, p_bd = n_dark_to_bright / n_dark, n_bright_to_dark / n_bright
        if plot:
            plt.figure(figsize=(12.8, 4.8))
            plt.bar(np.arange(n_tweezers), p_bd, label=f'Bright to Dark Probability', color='orange')
            plt.bar(np.arange(n_tweezers), p_db, label=f'Dark to Bright Probability', color='steelblue')
            plt.axhline(p_bd.mean(), label=f"Bright to Dark Average={p_bd.mean():.3}", color='darkorange', linestyle='--')
            plt.axhline(p_db.mean(), label=f"Dark to Bright Average={p_db.mean():.3}", color='dodgerblue', linestyle='--')
            plt.xlabel('Tweezer Number')
            plt.ylabel('Probability')
            plt.legend(loc='upper left')
            plt.title('Fidelity')
            plt.show()
        return n_dark_to_bright / n_dark, n_bright_to_dark / n_bright

import numpy as np
import matplotlib as plt
import tensorflow as tf
from tensorflow.keras import datasets, layers, models, optimizers

class ImageClassifier():

    def __init__(self, model=None):
        """
            model : an instance of an already compiled tensorflow neural network
        """
        self.model = model

    def train(self, training_images, training_labels, testing_images, testing_labels, epochs=10):
        """ Train the neural network for this dataset processing using the provided labels. """
        self.model.fit(training_images, training_labels, epochs=epochs)
        return self.model.evaluate(testing_images, testing_labels)

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
    
    def get_fidelity(self, images, n_tweezers, n_loops, plot=False):
        """ Return probability of bright to dark and dark to bright of each tweezer. """
        labels = np.reshape(self.classify(images)[:, 1], (n_tweezers, n_loops, -1))
        first_diff = np.diff(labels, axis=2)
        n_dark_to_bright = np.sum(first_diff == -1, axis=(1, 2))
        n_dark = np.sum(labels[:,:,:-1] == 0, axis=(1, 2))
        n_bright_to_dark = np.sum(first_diff == 1, axis=(1, 2))
        n_bright = np.sum(labels[:,:,:-1] == 1, axis=(1, 2))
        p_db, p_bd = n_dark_to_bright / n_dark, n_bright_to_dark / n_bright
        if plot:
            plt.bar(p_bd, label=f'Bright to Dark Probability (average={p_bd.mean()})')
            plt.bar(p_db, label=f'Dark to Bright Probability (average={p_db.mean()})')
            plt.x_label('Tweezer Number')
            plt.y_label('Probability')
            plt.legend()
            plt.title('Fidelity')
            plt.show()
        return n_dark_to_bright / n_dark, n_bright_to_dark / n_bright

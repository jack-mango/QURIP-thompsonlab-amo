import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow.keras import datasets, layers, models, optimizers

import logging
log = logging.getLogger(__name__)

class ModelTrainer():

    def __init__(self, model, training, testing, fidelity, n_tweezers, n_loops, learning_rate=1e-4, epochs=8):
        """
            model : an instance of an already compiled tensorflow neural network
        """
        self.model = model
        self.training_crops, self.training_labels = training
        self.testing_crops, self.testing_labels = testing
        self.fidelity_crops, self.fidelity_labels = fidelity
        self.n_tweezers = n_tweezers
        self.n_loops = n_loops
        self.learning_rate = learning_rate
        self.epochs = epochs

    def run(self):
        self.train()
        metrics = self.evaluate()
        fidelity = self.fidelity_analysis
        info = self.information(metrics, fidelity)
        return self.model, info

    def information(self, metrics, fidelity):
        info = {

        }
        return info

    def train(self):
        """ Train the neural network for this dataset processing using the provided labels. """
        return self.model.fit(self.training_crops, self.training_labels, epochs=self.epochs)
    
    def evaluate(self):
        return self.model.evaluate(self.testing_crops, self.testing_labels)
    
    def fidelity_analysis(self, n_tweezers, n_loops, plot=False):
        """ Return probability of bright to dark and dark to bright of each tweezer. """
        labels = np.reshape(np.argmax(self.model.predict(self.fidelity_crops), axis=1), (self.n_tweezers, self.n_loops, -1))
        first_diff = np.diff(self.fidelity_labels, axis=2)
        n_dark_to_bright = np.sum(first_diff == -1, axis=(1, 2))
        n_dark = np.sum(labels[:,:,:-1] == 0, axis=(1, 2))
        n_bright_to_dark = np.sum(first_diff == 1, axis=(1, 2))
        n_bright = np.sum(labels[:,:,:-1] == 1, axis=(1, 2))
        prob_db, prob_bd = n_dark_to_bright / n_dark, n_bright_to_dark / n_bright
       #if plot:
       #    plt.figure(figsize=(12.8, 4.8))
       #    plt.bar(np.arange(n_tweezers), p_bd, label=f'Bright to Dark Probability', color='orange')
       #    plt.bar(np.arange(n_tweezers), p_db, label=f'Dark to Bright Probability', color='steelblue')
       #    plt.axhline(p_bd.mean(), label=f"Bright to Dark Average={p_bd.mean():.3}", color='darkorange', linestyle='--')
       #    plt.axhline(p_db.mean(), label=f"Dark to Bright Average={p_db.mean():.3}", color='dodgerblue', linestyle='--')
       #    plt.xlabel('Tweezer Number')
       #    plt.ylabel('Probability')
       #    plt.legend(loc='upper left')
       #    plt.title('Fidelity')
       #    plt.show()
        return prob_db, prob_bd

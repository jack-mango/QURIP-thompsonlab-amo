import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow.keras import datasets, layers, models, optimizers

class ImageClassifier():
    
    def __init__(self, model):
        self.model = model

    def get_occupancies(self, crops):
        return np.argmax(self.model.predict(crops), axis=1)
    
    def fidelity_analysis(self, crops, n_tweezers, n_loops):
        """
        Determine the probability of a crop transitioning from bright to dark or dark to bright in a single loop
        for each tweezer. In addition make a bar graph for each tweezer containing this information.

        Returns:
        - prob_db: an n_tweezers long array containing the dark to bright probability for each tweezer.
        - prob_bd: an n_tweezers long array containing the bright to dark probability for each tweezer.
        - fig: a bar plot containing each tweezer's bright to dark probability and dark to bright probability.
        """
        occ = self.get_occupancies(crops)
        occ = np.reshape(occ, (n_tweezers, n_loops, -1))
        first_diff = np.diff(occ, axis=2)
        n_dark_to_bright = np.sum(first_diff == -1, axis=(1, 2))
        n_dark = np.sum(occ[:,:,:-1] == 0, axis=(1, 2))
        n_bright_to_dark = np.sum(first_diff == 1, axis=(1, 2))
        n_bright = np.sum(occ[:,:,:-1] == 1, axis=(1, 2))
        prob_db, prob_bd = n_dark_to_bright / n_dark, n_bright_to_dark / n_bright
        fig = plt.figure(figsize=(12.8, 4.8))
        plt.bar(np.arange(n_tweezers), prob_bd, label=f'Bright to Dark Probability', color='orange')
        plt.bar(np.arange(n_tweezers), prob_db, label=f'Dark to Bright Probability', color='steelblue')
        plt.axhline(prob_bd.mean(), label=f"Bright to Dark Average={prob_bd.mean():.3}", color='darkorange', linestyle='--')
        plt.axhline(prob_db.mean(), label=f"Dark to Bright Average={prob_db.mean():.3}", color='dodgerblue', linestyle='--')
        plt.xlabel('Tweezer Number')
        plt.ylabel('Probability')
        plt.legend(loc='upper left')
        plt.title('Fidelity')
        plt.show()
        return prob_db, prob_bd, fig


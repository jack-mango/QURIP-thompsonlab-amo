import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow.keras import datasets, layers, models, optimizers

class ImageClassifier():
    
    def __init__(self, n_tweezers, n_loops, model, crops):
        self.n_tweezers = n_tweezers
        self.n_loops = n_loops
        self.model = model
        self.crops = crops

    def run(self):
        occ = self.get_occupancies()
        prob_db, prob_bd, fig = self.fidelity_analysis(occ)
        info = {
            "Average Dark to Bright Probability": prob_db.mean(),
            "Average Bright to Dark Probability": prob_bd.mean(),
            "Plot": fig
        }
        occ = np.transpose(np.reshape(occ, (self.n_tweezers, -1)))
        return occ, info

    def get_occupancies(self):
        return np.argmax(self.model.predict(self.crops), axis=1)
    
    def fidelity_analysis(self, occ):
        """ Return probability of bright to dark and dark to bright of each tweezer. """
        occ = np.reshape(occ, (self.n_tweezers, self.n_loops, -1))
        first_diff = np.diff(occ, axis=2)
        n_dark_to_bright = np.sum(first_diff == -1, axis=(1, 2))
        n_dark = np.sum(occ[:,:,:-1] == 0, axis=(1, 2))
        n_bright_to_dark = np.sum(first_diff == 1, axis=(1, 2))
        n_bright = np.sum(occ[:,:,:-1] == 1, axis=(1, 2))
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

import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow.keras import datasets, layers, models, optimizers

import logging
log = logging.getLogger(__name__)

class ModelTrainer():

    """
    Contains the methods used in the model training stage of the pipeline. 

    Attributes:
    - model: an instance of a tensorflow.keras model. The model should be configured to have the same input dimension as
            training/testing crops and output dimension as training/testing labels
    - training_crops/testing_crops: a three dimensional array with the first axis corresponding to image number and the remaining
                                    two corresponding to the x and y axes of images. Used for testing/training
    - training_labels/testing_labels: a two dimensional array where the first axis corresponds to the image number
                                      and the second corresponds to bright/dark designation: [1, 0] -> dark; [0, 1] -> bright.
    - fidelity_crops: a three dimensional array with the first axis corresponding to image number and the remaining two
                     corresponding to the x and y axes of images. Used for fidelity analysis
    - n_tweezers: integer number of tweezers  
    - n_loops: integer number of loops
    - learning_rate: float to be used as learning rate when calling keras Model.fit method.
    - epochs: integer number of times the whole training dataset is used for training. 
    - validation_split: float between zero and one that determines how much of the training is used for validation
                        when training. 
    """

    def __init__(self, model, training, testing, fidelity_crops, n_tweezers, n_loops,
                  learning_rate=1e-4, epochs=8, validation_split=0.25):
        self.model = model
        self.training_crops, self.training_labels = training[:2]
        self.testing_crops, self.testing_labels = testing[:2]
        self.fidelity_crops = fidelity_crops
        self.n_tweezers = n_tweezers
        self.n_loops = n_loops
        self.learning_rate = learning_rate
        self.epochs = epochs
        self.validation_split = validation_split

    def run(self):
        """
        Execute the necessary methods of this class to obtain information for the next pipeline stage.

        Returns:
        - model: an instance of a keras Model that has been trained with self.training_crops
        - info: a dictionary including a plot with the results of fidelity analysis, the binary accuracy of the model during
                testing, and the average dark to bright and bright to dark probabilities. 
        """
        self.train()
        metrics = self.evaluate()
        prob_db, prob_bd, plot = self.fidelity_analysis()
        info = {
            "Plot": plot,
            "Binary Accuracy": metrics[0],
            "Average Tweezer Dark to Bright Probability": prob_db,
            "Average Tweezer Bright to Dark Probabiblity": prob_bd
        }
        return self.model, info

    def train(self):
        """
        Trains the instance's model with the training dataset and gives history of the training. 

        Returns:
        - history: a list containing the metrics of the model for each epoch during training. 
        """
        return self.model.fit(self.training_crops, self.training_labels,
                               validation_split=self.validation_split, epochs=self.epochs,
                               batch_size=32)
    
    def evaluate(self):
        """
        Use the labeled testing dataset to determine the binary accuracy of the model. 

        Returns:
        - metrics: an array containing the metrics of this network for the testing dataset. Metrics are determined
                   at model compilation. 
        """
        return self.model.evaluate(self.testing_crops, self.testing_labels)
    
    def fidelity_analysis(self):
        """
        Determine the probability of a crop transitioning from bright to dark or dark to bright in a single loop
        for each tweezer. In addition make a bar graph for each tweezer containing this information.

        Returns:
        - prob_db: an n_tweezers long array containing the dark to bright probability for each tweezer.
        - prob_bd: an n_tweezers long array containing the bright to dark probability for each tweezer.
        - fig: a bar plot containing each tweezer's bright to dark probability and dark to bright probability.
        """        
        labels = np.reshape(np.argmax(self.model.predict(self.fidelity_crops), axis=1), (self.n_tweezers, self.n_loops, -1))
        first_diff = np.diff(labels, axis=2)
        n_dark_to_bright = np.sum(first_diff == -1, axis=(1, 2))
        n_dark = np.sum(labels[:,:,:-1] == 0, axis=(1, 2))
        n_bright_to_dark = np.sum(first_diff == 1, axis=(1, 2))
        n_bright = np.sum(labels[:,:,:-1] == 1, axis=(1, 2))
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

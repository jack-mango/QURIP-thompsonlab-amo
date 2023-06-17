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
    
    
    
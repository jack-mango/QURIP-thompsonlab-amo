import numpy as np
from tensorflow.keras import layers, models, optimizers

class DatasetBuilder():

    def __init__(self, crops, labels, model, testing_fraction=0.1):
        self.model = model
        self.crops = crops
        self.labels = labels
        self.testing_fraction = testing_fraction

    def run(self):
        crops, labels = self.filter_unlabeled()
        testing_indices = np.random.choice(len(crops), int(len(crops) * self.testing_fraction), replace=False)
        labels = np.transpose(np.array([np.absolute(labels - 1), labels]))
        training_crops = np.delete(crops, testing_indices, axis=0)
        training_labels = np.delete(labels, testing_indices, axis=0)
        testing_crops = crops[testing_indices]
        testing_labels = crops[testing_indices]
        info = self.dataset_statistics(training_labels, testing_labels)
        return (training_crops, training_labels), (testing_crops, testing_labels), (crops, labels), info

    def dataset_statistics(self, training_labels, testing_labels):
        info = {"Number of Training Images": training_labels.shape[0],
                "Number of Testing Images": testing_labels.shape[0],
                "Training Bright to Dark Ratio": self.bright_dark_ratio(training_labels),
                "Testing Bright to Dark Ratio": self.bright_dark_ratio(testing_labels),
                "Number of Unlabeled Images": np.sum(np.isnan(self.labels))
                }
        return info
    
    def filter_unlabeled(self, crops, labels):
        unlabeled_images = np.isnan(labels)
        mask = ~ unlabeled_images
        return crops[mask], labels[mask]
    
    def bright_dark_ratio(self, labels):
        """ Training first, then testing"""
        training_numbers = np.sum(self.training_labels, axis=0)
        testing_numbers = np.sum(self.testing_labels, axis=0)
        return training_numbers[1] / training_numbers[0], testing_numbers[1], training_numbers[0]





    def train(self, learning_rate, epochs):
        self.model.compile(optimizer=optimizers.Adam(learning_rate=learning_rate),
                            loss='binary_crossentropy', metrics=['binary_accuracy'])
        return self.model.fit(self.training_crops, self.training_labels, epochs=epochs)
    
    def evaluate(self):
        return self.model.evaluate(self.testing_crops, self.testing_labels)
    

    

from PipelineStages import *
import argparse
import numpy as np
from tensorflow.keras import models

parser = argparse.ArgumentParser(
    prog="GreenModelTraining",
    description="This program trains a neural network to classify fluoresence images"
)

parser.add_argument('n_loops', type=int,
                    help="Enter the number of loops in a dataset")
parser.add_argument('n_tweezers', type=int,
                    help="Enter the number of tweezers in an image")
parser.add_argument('data_dir', type=str,
                    help="Specify the relative path of the data location")
parser.add_argument('model_dir', type=str,
                    help="Specify the relative path of where the model is located")
                    
if __name__ == "__main__":
    info = {}
    args = parser.parse_args()

    # Load data and stack
    loader = Loader.Loader(args.n_loops, args.n_tweezers, args.data_dir, args.model_dir)
    stack, tot_loops, model = loader.run()

    # Find positions and create crops
    processor = ImageProcessing.ImageProcessor(stack, args.n_tweezers, tot_loops)
    crops3x3, crops1x1, positions = processor.run()

    # Traing the model and report its performance
    classifier = ImageClassifier.ImageClassifier(args.n_tweezers, tot_loops, model, crops3x3)
    occ, info = classifier.run()

    # Save positions and occupancies
    np.savez_compressed(args.data_dir, occupancies = occ)
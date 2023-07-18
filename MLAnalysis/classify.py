from PipelineStages import *
import argparse
import numpy as np
from tensorflow.keras import models
                    
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

def load(n_loops, n_tweezers, data_dir, model_dir):
    return

def crop_tweezers(positions, separation):
    processor
    return

def classify():
    return
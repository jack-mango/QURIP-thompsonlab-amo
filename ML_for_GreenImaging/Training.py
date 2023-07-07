from PipelineStages import *
import argparse
from tensorflow.keras import models
import logging


logging.basicConfig(level=logging.INFO)

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
parser.add_argument('-b', '--build', action='store_true', default=False,
                    help="If used, a new model will be built and trained from scratch")

if __name__ == "__main__":
    info = {}
    args = parser.parse_args()

    # Load data and stack
    loader = Loader.Loader(args.n_loops, args.n_tweezers, args.data_dir, args.model_dir)
    stack, tot_loops, model = loader.run()

    # Find positions and create crops
    processor = ImageProcessing.ImageProcessor(stack, args.n_tweezers, tot_loops)
    crops3x3, crops1x1, positions, stage_info = processor.run()
    info.update(stage_info)

    # Create labels
    labeler = Labeler.Labeler(crops1x1, args.n_tweezers, tot_loops)
    labels, stage_info = labeler.run()
    info.update(stage_info)

    # Make training/testing/fidelity analysis datasets
    builder = DatasetBuilder.DatasetBuilder(crops3x3, labels)
    training, testing, fidelity, stage_info = builder.run()
    info.update(stage_info)

    # Training the model and report its performance
    trainer = ModelTrainer.ModelTrainer(model, training, testing, fidelity, args.n_tweezers, tot_loops)
    model, stage_info = trainer.run()
    info.update(stage_info)

    # Save the model and generate a report of the training process
    model.save(args.model_dir)



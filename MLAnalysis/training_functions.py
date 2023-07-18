from PipelineStages import *
import argparse
from tensorflow.keras import models
import logging
import numpy as np


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

def load(n_loops, n_tweezers, data_dir, model_dir):
    loader = Loader.Loader(n_loops, n_tweezers, data_dir, model_dir)
    stack, n_files = loader.load_data()
    model = loader.load_model()
    return stack, model, n_files

def process_images(stack, n_tweezers, n_loops):
    processor = ImageProcessing.ImageProcessor(stack, n_tweezers, n_loops)
    processor.find_tweezer_positions()
    processor.plot()
    nn_dist = processor.find_nn_dist()
    crops_3x3 = processor.crop_tweezers(3, nn_dist)
    crops_1x1 = processor.crop_tweezers(1, nn_dist)
    info = {"Positions plot": processor.plot(), "Positions": processor.positions}
    return crops_3x3, crops_1x1, info

def split_training_fidelity(crops_1x1, crops_3x3, n_loops, n_files):
    if n_files == 1:
        random_loop_num = np.random.randint(n_loops)
        per_loop = crops_1x1.shape[1] // n_loops
        fidelity = crops_3x3[:, random_loop_num * per_loop: (random_loop_num + 1) * per_loop]
        labeling = np.delete(crops_1x1, np.s_[random_loop_num * per_loop: (random_loop_num + 1) * per_loop], axis=1)
        training = np.delete(crops_3x3, np.s_[random_loop_num * per_loop: (random_loop_num + 1) * per_loop], axis=1)
    else:
        random_file_num = np.random.randint(n_files)
        per_file = crops_1x1.shape[1] // n_files
        fidelity = crops_3x3[:, random_file_num * per_file: (random_file_num + 1) * per_file]
        labeling = np.delete(crops_1x1, np.s_[random_file_num * per_file: (random_file_num + 1) * per_file], axis=1)
        training = np.delete(crops_3x3, np.s_[random_file_num * per_file: (random_file_num + 1) * per_file], axis=1)
    return training, labeling, fidelity

def label_images(crops, n_files, n_loops, n_tweezers):
    if n_files == 1:
        labels, info = label_images_helper(crops, n_loops, n_tweezers)
    else:
        per_file = crops.shape[1] // n_files # number of images per tweezer per file
        labels = np.empty((n_files, n_tweezers, per_file))
        info = {}
        for i in range(n_files):
            file_labels, file_info = label_images_helper(crops[:, i * per_file : (i + 1) * per_file], n_loops, n_tweezers)
            labels[i] = np.reshape(file_labels, (n_tweezers, -1))
            info.update(file_info)
        labels = np.swapaxes(labels, 0, 1)
    return labels.flatten(), info

def label_images_helper(crops, n_loops, n_tweezers):
    labeler = Labeler.Labeler(crops, n_tweezers, n_loops)
    fits, r_sq = labeler.bright_dark_fit()
    thresholds, plots = labeler.find_thresholds(fits)
    all_below_upper, all_above_lower = labeler.threshold_misfits(thresholds)
    labels = labeler.make_labels(thresholds)
    info = {
            "Histogram fits plot": plots,
            "Thresholds": thresholds,
            "Tweezers missing dark labels": all_above_lower,
            "Tweezers missing bright labels": all_below_upper,
            "R^2 Values": r_sq,
            "Image Value Distribution Fits": fits
            }
    return labels, info

def build_dataset(crops, labels):
    builder = DatasetBuilder.DatasetBuilder(crops, labels)
    crops, labels = builder.filter_unlabeled()
    training, testing = builder.testing_training_split(crops, labels)
    info = builder.dataset_info(training, testing)
    return training, testing, info

def train_model(model, training, testing):
    n_training = training[1].shape[0]
    n_testing = testing[1].shape[0]
    history = model.fit(training[0], training[1], epochs=8, validation_split = n_testing / (n_training + n_testing))
    testing_metrics = model.evaluate(testing[0], testing[1])
    return history, testing_metrics

def analyze_fidelity(model, crops, n_loops, n_tweezers):
    classifier = ImageClassifier.ImageClassifier(model)
    prob_db, prob_bd, fig = classifier.fidelity_analysis(np.reshape(crops, (-1, *crops.shape[-2:])), n_tweezers, n_loops)
    return prob_db, prob_bd

if __name__ == "__main__":
    info = {}
    args = parser.parse_args()
    
    stack, model, n_files = load(args.n_loops, args.n_tweezers, args.data_dir, args.model_dir)

    crops_3x3, crops_1x1, info = process_images(stack, args.n_tweezers, args.n_loops * n_files)

    training, labeling, fidelity_crops = split_training_fidelity(crops_1x1, crops_3x3, args.n_loops, n_files)

    labels, info = label_images(labeling, n_files - 1, args.n_loops, args.n_tweezers)

    training, testing, info = build_dataset(crops_3x3, labels)

    history, testing_metrics = train_model(model, training, testing)

    fidelity = analyze_fidelity(fidelity_crops, args.n_loops, args.n_tweezers)

    model.save(args.model_dir)
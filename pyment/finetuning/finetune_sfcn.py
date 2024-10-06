import os
import argparse
import pandas as pd
import numpy as np
import sys

from pyment.models import get as get_model, ModelType
from pyment.data import AsyncNiftiGenerator, NiftiDataset  # NiftiDatasetCSV  # type: ignore

import tensorflow as tf
import keras
from keras.callbacks import CSVLogger
from keras.callbacks import EarlyStopping  # noqa F401
from keras.callbacks import ModelCheckpoint
from tensorflow.keras.optimizers.schedules import CosineDecay  # type: ignore

import wandb
import wandb.integration.keras as WK

# continuously print for HPC
import datetime
ts = str(datetime.datetime.now())

# wandb.init(project="SFCN", config={"batch_size": 4, "dropout_rate": 0.3})

# tf.config.run_functions_eagerly(True)

# Place where the output file is sent to
def printf(a, ts=ts):
    print(a)
    path = os.path.join(
        '/home/internvas/vas_tmp/Yichi_BrainAge/Results/finetuning_pyment',
        'result', ts)
    os.makedirs(path, exist_ok=True)
    with open(os.path.join(path, 'output.txt'), mode='a+') as f:
        print(a, file=f)


def load_datasets_csv(args):

    print(f"Loading train dataset from {os.path.join(args.labels_path, f'train_fold_{args.fold}.csv')}")

    args.train_dataset = NiftiDataset.from_csv(
        os.path.join(args.labels_path, f'train_fold_{args.fold}.csv'), labels_cols=['age'], target='age'
    )
    print(f"Loaded train dataset: {args.train_dataset}")

    print(f"Loading validation dataset from {os.path.join(args.labels_path, f'val_fold_{args.fold}.csv')}")

    args.val_dataset = NiftiDataset.from_csv(
        os.path.join(args.labels_path, f'val_fold_{args.fold}.csv'), labels_cols=['age'], target='age'
    )
    print(f"Loaded validation dataset: {args.val_dataset}")

    print(f"Loading test dataset from {os.path.join(args.labels_path, f'test_fold_{args.fold}.csv')}")

    args.test_dataset = NiftiDataset.from_csv(
       os.path.join(args.labels_path, f'test_fold_{args.fold}.csv'), labels_cols=['age'], target='age'
    )
    print(f"Loaded test dataset: {args.test_dataset}")

    return args


# load generators from datasets
def load_generators(args):
    def preprocessor(x):
        return (x / 255.0) if args.normalize else x

    if args.threads is None or args.threads == 1:
        raise NotImplementedError(('Predicting from synchronous generator '
                                   'is not implemented'))

    args.train_gen = AsyncNiftiGenerator(args.train_dataset,
                                         preprocessor=preprocessor,
                                         batch_size=args.batch_size,
                                         threads=args.threads,
                                         infinite=True, shuffle=True)
    args.val_gen = AsyncNiftiGenerator(args.val_dataset,
                                       preprocessor=preprocessor,
                                       batch_size=args.batch_size,
                                       threads=args.threads,
                                       infinite=True, shuffle=True)
    args.test_gen = AsyncNiftiGenerator(args.test_dataset,
                                        preprocessor=preprocessor,
                                        batch_size=args.batch_size,
                                        threads=args.threads)
    return args


# set steps by reducing if only doing a quick test
def set_steps(args):
    if args.quick_test:
        args.steps_tr = 2
        args.steps_val = 2
        args.steps_te = None
    else:
        args.steps_tr = args.train_gen.batches
        args.steps_val = args.val_gen.batches
        args.steps_te = None
    return args


# callbacks
def set_callbacks(args):
    args.monitor = 'val_mae'
    args.cb = None

    # early stopping
    # args.cb = [EarlyStopping(monitor=args.monitor, min_delta=1e-5, patience=5, restore_best_weights=True)]

    if args.cp_path is not None:
        args.cb = []

        # stream epoch results
        args.csv_filename = os.path.join(args.cp_path, 'log.csv')
        args.cb.append(CSVLogger(args.csv_filename))

        # save best model
        args.bw_filepath = os.path.join(args.cp_path, 'best_model.h5')
        args.cb.append(ModelCheckpoint(filepath=args.bw_filepath,
                                       monitor=args.monitor, verbose=1,
                                       save_best_only=True, mode='min'))

        # save models after every epoch
        if not args.save_best_only:
            args.cp_filepath = os.path.join(args.cp_path, 'epochs',
                                            '{epoch:04d}.h5')
            args.cb.append(ModelCheckpoint(filepath=args.cp_filepath,
                                           verbose=1))

    # Add Wandb Metrics logger to track metrics
    # args.cb.append(WK.WandbMetricsLogger())

    # Add Wandb Model Checkpoint to save the best model to wandb
    # args.cb.append(WandbModelCheckpoint(filepath="wandb://model-best.h5", monitor=args.monitor, save_best_only=True))

    return args


# set model with hyperparameters
def set_model(args):
    # load
    args.model = get_model(args.model_name, weights=args.weights,
                           dropout=args.dropout_rate,
                           weight_decay=args.weight_decay,
                           prediction_range=None)

    # layers to train
    if args.last_layer_only:
        for layer in args.model.layers:
            if layer.name == 'Regression3DSFCN/predictions':
                layer.trainable = True
            else:
                layer.trainable = False
    else:
        args.model.trainable = True

    # learning rate decay
    args.decay_steps = args.train_gen.batches*args.lr_decay_epochs
    args.lr = CosineDecay(initial_learning_rate=args.initial_learning_rate,
                          decay_steps=args.decay_steps)
    args.model.compile(optimizer=keras.optimizers.Adam(learning_rate=args.lr),
                       loss=keras.losses.MeanSquaredError(),
                       metrics=['mae', 'mse'])
    args.model.summary()

    # TEMP
    printf('Last layer only')
    printf(args.last_layer_only)
    printf('Dropout rate')
    printf(args.dropout_rate)
    printf('Weight decay')
    printf(args.weight_decay)
    printf('Initial learning rate')
    printf(args.initial_learning_rate)
    printf('Learning rate decay epochs')
    printf(args.lr_decay_epochs)
    printf('Initial training metrics:')
    printf(args.model.evaluate(args.train_gen, verbose=2, steps=args.steps_tr))
    printf(' ')
    printf('Initial validation metrics:')
    printf(args.model.evaluate(args.val_gen, verbose=2, steps=args.steps_val))

    return args


# configure training parameters, callbacks, etc.
def config_training(args):
    args = set_steps(args)
    args = set_callbacks(args)
    args = set_model(args)
    return args


# run training
def run_training(args):
    print("Finetuning Started")
    args.history = args.model.fit(args.train_gen, epochs=args.max_epochs,
                                  verbose=2, steps_per_epoch=args.steps_tr,
                                  validation_data=args.val_gen,
                                  validation_steps=args.steps_val,
                                  callbacks=args.cb)

    # wandb.finish()
    return args


# generate predictions on test set
def predict_brain_age(args):
    args.ids = args.test_gen.dataset.ids
    args.labels = args.test_gen.dataset.y

    if args.cp_path is not None:
        args.model.load_weights(args.bw_filepath)

    args.predictions = args.model.predict(args.test_gen, steps=args.steps_te)
    if args.model.type == ModelType.REGRESSION:
        args.predictions = args.predictions.squeeze()

    # save
    args.df = pd.DataFrame(
        {'age': args.labels, 'prediction': args.predictions}, index=args.ids
    )
    args.df.to_csv(args.destination)

    return args

# Test function
def test_generator(generator, num_batches):
    for batch_index, (inputs, labels) in enumerate(generator):
        if batch_index >= num_batches:
            break

        # Access the MRI images and modalities from the inputs dictionary
        mri_images = inputs['Regression3DSFCN/inputs']
        input_types = inputs['input_type']

        print(f"Batch {batch_index}:")
        print(f" - MRI Image Shape: {mri_images.shape}")
        print(f" - Input Types: {input_types}")
        print(f" - Labels: {labels}")

# Finetune brain age model then generate predictions on test set
def finetune_and_predict_brain_age(args):
    keras.utils.set_random_seed(0)

    if args.model_name == "sfcn-regwe":
        args = load_datasets_csv_modality(args)
        args = load_generators_modality(args)
        test_generator(args.test_gen, 10)

    else:
        args = load_datasets_csv(args)
        args = load_generators(args)

    args = config_training(args)
    args = run_training(args)
    args = predict_brain_age(args)


# pass in arguments
if __name__ == '__main__':
    parser = argparse.ArgumentParser(('Performs finetuning'))

    parser.add_argument('-f', '--folder', required=False,
                        help=('Folder containing images'))
    parser.add_argument('--labels_path', required=True,
                        help='Path to directory containing labels')
    parser.add_argument('-m', '--model_name', required=True,
                        help='Name of the model to use (e.g. sfcn-reg)')
    parser.add_argument('-w', '--weights', required=False, default=None,
                        help='Weights to load in the model')
    parser.add_argument('-b', '--batch_size', required=True, type=int,
                        help='Batch size to use while training and predicting')
    parser.add_argument('-t', '--threads', required=False, default=None,
                        type=int, help=('Number of threads to use for reading '
                                        'data. If not set, a synchronous '
                                        'generator will be used'))
    parser.add_argument('-n', '--normalize', action='store_true',
                        help=('If set, images will be normalized to range '
                              '(0, 1) before prediction'))
    parser.add_argument('-d', '--destination', required=True,
                        help=('Path where CSV containing ids, labels '
                              'and predictions are stored'))
    parser.add_argument('-o', '--fold', required=True,
                        help=('Indicator of train/test fold (e.g. 0)'))
    parser.add_argument('-e', '--max_epochs', required=False, default=2,
                        type=int, help=('Maximum number of epochs for '
                                        'finetuning'))
    parser.add_argument('-a', '--last_layer_only', action='store_true',
                        help=('if set, only finetune last layer'))
    parser.add_argument('-r', '--dropout_rate', required=False, default=0.3,
                        type=float, help=('p of dropout for dropout layers'))
    parser.add_argument('-g', '--weight_decay', required=False, default=1e-3,
                        type=float, help=('weight decay'))
    parser.add_argument('-i', '--initial_learning_rate', required=False,
                        default=1e-3, type=float, help=('initial l.r.'))
    parser.add_argument('-l', '--lr_decay_epochs', required=False,
                        default=25, type=int, help=('# epochs to decay over'))
    parser.add_argument('-c', '--cp_path', required=False, default=None,
                        help=('Path to directory where intermediate results '
                              'are saved. If None, these are not saved, but '
                              'best weights would not be restored.'))
    parser.add_argument('-v', '--save_best_only', action='store_true')
    parser.add_argument('-q', '--quick_test', action='store_true')

    args = parser.parse_args()
    # print(f"Parsed threads: {args.threads}")  # Debugging line

    finetune_and_predict_brain_age(args)

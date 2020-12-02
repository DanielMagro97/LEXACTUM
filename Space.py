# Copyright (C) 2020  Daniel Magro
# Full License at: https://github.com/DanielMagro97/LEXACTUM/blob/main/LICENSE

from typing import List             # for type annotation

import argparse                     # for passing hyperparameters from the CLI
import os                           # for working with files and directories
import re                           # for regex (splitting strings by delimiters)
import sys                          # for exiting the program in case of invalid value
import timeit                       # for calculating execution time of models
import pandas as pd                 # for Dataframes (importing the labels .csv file with ease)
from sklearn.utils import shuffle   # for randomising the order of data
import numpy as np                  # for numpy arrays

from utils.load_fits import load_fits_image                 # for loading fits images with optional normalisation
from utils.image_augmentation import initialise_augmenter   # for defining the augmenter which will be used during training
from neural_networks.cas_swinburne import cas_swinburne     # for using the defined CNN model
from neural_networks.lastro_epfl import lastro_epfl         # for using the defined CNN model
from neural_networks.cmu_deeplens import cmu_deeplens       # for using the defined resnet model
from neural_networks.wsi_net import wsi_net
from neural_networks.lens_flow import lens_flow
from neural_networks.lens_finder import lens_finder

from tensorflow import keras

from utils.results import show_results  # for plotting ROC curves, calculating AUC, TPR0 and TPR10


# Hyper Parameters
image_size: int = 101
no_of_channels: int = 1
# whether to train a model from scratch, or to load a model from disk
# accepted values: 'train' and 'load'
model_train_load: str = 'train'
# if model_train_load is set to 'train', which model to train
# # accepted values: cas_swinburne, lastro_epfl, cmu_deeplens, wsi_net, lens_flow, lens_finder
# if model_train_load is set to 'load', which model to load from the models folder
# # accepted values: any model in the models folder, including the .h5 extension
model_name: str = 'cmu_deeplens' #'space_cmu_deeplens_10epochs.h5'
# number of epochs to train for
no_of_epochs: int = 1
# number of images for the image generator to load in one batch
batch_size: int = 8
# whether to augment images during training
augment_images: bool = True


def load_data(data_path: str):
    data_subdirectories: List[str] = next(os.walk(data_path))[1]

    # store a list of paths to each .fits file
    image_paths: List[str] = []
    # store a list of labels corresponding to each image
    image_labels: List[int] = []

    # load csv of labels (column 1 index number - column 22 0 if there is a source, 1 if there isn't)
    labels: pd.DataFrame = pd.read_csv(os.path.join(data_path, 'euclidB_image_catalog.csv'),
                                       comment='#', header='infer', index_col='ID')

    # loop over every subdirectory in the Data folder
    for data_subdirectory in data_subdirectories:
        # save the current subdirectory's path
        current_path: str = os.path.join(data_path, data_subdirectory, 'Public', 'Band1')

        # loop over every .fits file in the current subdirectory
        for fits_file in os.listdir(current_path):
            # check that the file is a .fits file
            if fits_file.endswith('.fits'):
                # append the current_path+fits_file to the list of image_paths
                image_paths.append(os.path.join(current_path, fits_file))

                # Find the corresponding label for the current .fits file
                # find the name of the current .fits file
                fits_file_id: str = re.split('[-.]', fits_file)[1]
                # find the label corresponding to the current .fits file from the labels DataFrame
                image_label: int = labels.at[int(fits_file_id), 'no_source']

                # "0 if there is a source, 1 if there isn't"
                # cases without sources are not lenses => label 1 = not a lens
                image_labels.append(image_label)

    # based on the above, the labels are flipped, so that 1 now represents a case with a lens
    image_labels = [1 - x for x in image_labels]

    return image_paths, image_labels


# could use: sklearn.model_selection.train_test_split
def split_data(image_paths: List[str], image_labels: List[int],
               train_size: float, val_size: float, test_size: float):
    # if the ratio of training to validation to test data does not add up to 1
    if (train_size + val_size + test_size) != 1:
        # throw an error and do not continue
        sys.exit('Invalid Data Split Ratio')

    # calculate how many input,target pairs each set should have
    train_set_size: int = int(len(image_paths) * train_size)
    val_set_size: int = int(len(image_paths) * val_size)
    test_set_size: int = int(len(image_paths) * test_size)

    # Splitting the data into a training set and a validation set
    train_images = image_paths[:train_set_size]
    train_labels = image_labels[:train_set_size]

    val_images = image_paths[train_set_size:train_set_size+val_set_size]
    val_labels = image_labels[train_set_size:train_set_size+val_set_size]

    test_images = image_paths[-test_set_size:]
    test_labels = image_labels[-test_set_size:]

    return train_images, train_labels, val_images, val_labels, test_images, test_labels


class ImageLabelGenerator(keras.utils.Sequence):
    def __init__(self, image_paths: List[str], image_labels: List[int], batch_size: int = 8,
                 image_size: int = 101, no_of_channels: int = 1,
                 augmenter=None, shuffle: bool = True):
        self.image_paths: List[str] = image_paths
        self.image_labels: List[int] = image_labels
        self.batch_size: int = batch_size
        self.image_size: int = image_size
        self.no_of_channels: int = no_of_channels
        self.augmenter = augmenter
        self.shuffle: bool = shuffle
        self.on_epoch_end()

    def __getitem__(self, index):
        if (index + 1) * self.batch_size > len(self.image_paths):
            batch_size = len(self.image_paths) - index * self.batch_size
        else:
            batch_size = self.batch_size

        image_paths_batch = self.image_paths[index * batch_size: (index + 1) * batch_size]
        image_labels_batch = self.image_labels[index * batch_size: (index + 1) * batch_size]

        # create the numpy arrays which will store the batch of images and the batch of corresponding labels
        image_batch: np.ndarray = np.empty((self.batch_size, self.image_size, self.image_size, self.no_of_channels))
        label_batch: np.ndarray = np.empty((self.batch_size), dtype=int)

        for i, (image_path, image_label) in enumerate(zip(image_paths_batch, image_labels_batch)):
            image_batch[i] = load_fits_image(image_path, 'ZScale')#, fits_file_normalisation) #TODO
            label_batch[i] = image_label

        # to display a 'before augmentation' image
        # import imgaug as ia
        # ia.imshow(np.hstack(np.reshape(image_batch, (batch_size, image_size, image_size))))

        # if an augmenter has been passed (i.e. this is a training data generator)
        if self.augmenter is not None:
            # augment the batch of images
            image_batch = self.augmenter(images=image_batch)

        # to display an 'after augmentation' image
        # ia.imshow(np.hstack(np.reshape(image_batch, (batch_size, image_size, image_size))))

        return image_batch, label_batch

    def on_epoch_end(self):
        # shuffle the data at the end of each epoch
        if self.shuffle:
            self.image_paths, self.image_labels = shuffle(self.image_paths, self.image_labels)

    def __len__(self):
        return int(np.ceil(len(self.image_paths) / float(self.batch_size)))


if __name__ == '__main__':
    # Parse command line arguments
    parser = argparse.ArgumentParser(description='Train Models to classify Gravitational Lenses.')

    parser.add_argument('--dataset', required=False, type=str,
                        metavar='Dataset Path', help='Directory of the dataset')
    parser.add_argument('--train_or_load', required=False, default='train', type=str, choices=['train', 'load'],
                        help='Whether to train a model, or to load one from disk')
    parser.add_argument('--model_name', required=False, default='cmu_deeplens', type=str,
                        help='The name of the model to train. Accepted values are: '
                             'cas_swinburne, lastro_epfl, cmu_deeplens, wsi_net, lens_flow or lens_finder '
                             'Or the name of the file in the models folder to load from disk (including .h5)')
    parser.add_argument('--no_of_epochs', required=False, default=10, type=int,
                        metavar='Number of training epochs', help='Number of training epochs')
    parser.add_argument('--batch_size', required=False, default=8, type=int,
                        metavar='Number of images per batch', help='Number of images per batch')
    # set to str instead of bool (False would be considered True). Should be --aug vs --no_aug
    parser.add_argument('--augment_images', required=False, default='True', type=str, choices=['True', 'False'],
                        help='Whether to perform Image Augmentation on the training data while training')

    args = parser.parse_args()
    if args.dataset is None:
        args.dataset = os.path.join('D:', os.sep, 'datasets', 'GravitationalLensFindingChallenge',
                                    'Challenge1.0', 'Space')
    print(args.dataset)
    data_path: str = args.dataset
    print(args.train_or_load)
    model_train_load: str = args.train_or_load
    print(args.model_name)
    model_name: str = args.model_name
    print(args.no_of_epochs)
    no_of_epochs: int = args.no_of_epochs
    print(args.batch_size)
    batch_size: int = args.batch_size
    if args.augment_images == 'True':
        augment_images: bool = True
    else:
        augment_images: bool = False
    print(augment_images)

    # data_path: str = os.path.join('D:', os.sep, 'datasets', 'GravitationalLensFindingChallenge',
    #                               'Challenge1.0', 'Space')

    # Loading the data
    print('Locating Data')
    # call the load_data function to initialise two lists image_paths and image_labels with the
    # paths of the images and their corresponding label
    image_paths, image_labels = load_data(data_path)

    # Randomising the order of the data
    print('Randomising the order of the data')
    image_paths, image_labels = shuffle(image_paths, image_labels)

    # Splitting the data into Training, Validation and Test sets
    print('Splitting Data into Training, Validation and Test Sets')
    train_images, train_labels, val_images, val_labels, test_images, test_labels = \
        split_data(image_paths, image_labels, 0.19, 0.01, 0.8)

    # Training or Loading the Model
    # if a new model will be trained
    if model_train_load == 'train':
        print('Training Model')

        if model_name == 'cas_swinburne':
            model = cas_swinburne(image_size, no_of_channels)
        elif model_name == 'lastro_epfl':
            model = lastro_epfl(image_size, no_of_channels)
        elif model_name == 'cmu_deeplens':
            model = cmu_deeplens(image_size, no_of_channels)
        elif model_name == 'wsi_net':
            model = wsi_net(image_size, no_of_channels)
        elif model_name == 'lens_flow':
            model = lens_flow(image_size, no_of_channels)
        elif model_name == 'lens_finder':
            model = lens_finder(image_size, no_of_channels)
        else:
            sys.exit('Invalid model_name chosen, value must be one of:\n'
                     'cas_swinburne, lastro_epfl, cmu_deeplens, wsi_net, '
                     'lens_flow or lens_finder')

        model.compile(optimizer='adam',
                      loss='binary_crossentropy',
                      metrics=['acc'])
        # model.compile(optimizer='adam',
        #               loss='binary_crossentropy',
        #               metrics=[keras.metrics.AUC()])

        # if the augment_images hyper parameter has been set to true
        if augment_images:
            # initialise the augmenter
            augmenter = initialise_augmenter()
        else:
            augmenter = None

        train_data_generator = ImageLabelGenerator(train_images, train_labels, batch_size, image_size, no_of_channels, augmenter)
        val_data_generator = ImageLabelGenerator(val_images, val_labels, batch_size, image_size, no_of_channels)

        steps_per_epoch: int = int(len(train_images) / batch_size)
        validation_steps: int = int(len(val_images) / batch_size)

        print('Training:')
        model.fit(train_data_generator,
                  steps_per_epoch=steps_per_epoch,
                  validation_data=val_data_generator,
                  validation_steps=validation_steps,
                  epochs=no_of_epochs,
                  verbose=2)

        # loss = model.history['loss']
        # val_loss = model.history['val_loss']
        # print(loss)
        # print(val_loss)
    # if a model will be loaded from disk
    elif model_train_load == 'load':
        print('Loading Model from Disk')

        model_path = os.path.join('models', model_name)
        if os.path.isfile(model_path):
            model = keras.models.load_model(model_path)
        else:
            sys.exit('Model at ' + model_path + ' was not found!')
    else:
        sys.exit('Invalid value for model_train_load chosen,\n'
                 'value must either be train or load')

    print('Evaluating:')
    # saving the start time to evaluate how long the model takes to execute
    start_time = timeit.default_timer()
    # model_eval = model.evaluate(ImageLabelGenerator(test_images, test_labels, batch_size, image_size, no_of_channels, shuffle=False),
    #                             verbose=2)
    stop_time = timeit.default_timer()
    # print(model_eval)

    # calculating execution time
    execution_time = stop_time - start_time
    print('Model took ' + str(execution_time) + ' seconds to execute on ' + str(len(test_images)) + ' images')

    print('Predicting:')
    model_pred = model.predict(ImageLabelGenerator(test_images, test_labels, batch_size, image_size, no_of_channels, shuffle=False))
    model_pred = model_pred.ravel()
    # since the data generator may load more images than there are due to the batch size
    model_pred = model_pred[:len(test_labels)]

    # TODO try model.eval_ROC(xval,yval)
    # Compute the metrics for the trained model
    # pos_label has been set to 1 since the labels were flipped
    # (by default this dataset uses 0 as the positive label)
    auc, tpr_0, tpr_10 = show_results(test_labels, model_pred, pos_label=1)

    if model_train_load == 'train':
        # Saving model weights to disk
        print('Saving Model:')
        # make sure the directory where the model will be saved exists
        if not os.path.exists('models'):
            # if not, create it
            os.makedirs('models')
        model_name: str = 'space_' + model_name + '_' + str(no_of_epochs)+'epochs'
        model.save('models/' + model_name + '.h5')
        print('Model saved to disk.')

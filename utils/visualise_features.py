# Adapted from:
# https://machinelearningmastery.com/how-to-visualize-filters-and-feature-maps-in-convolutional-neural-networks/

import os                           # for working with files and directories
import sys                          # for exiting the program in case of invalid value
import numpy as np                  # for numpy arrays

from utils.load_fits import load_fits_image                 # for loading fits images with optional normalisation

from tensorflow import keras

import matplotlib.pyplot as plt         # for plotting


def visualize_features(model, image):
    for i, layer in enumerate(model.layers):
        # check for convolutional layer
        if 'conv' not in layer.name:
            continue
        # summarize output shape
        print(i, layer.name, layer.output.shape)

    # choosing the last convolutional layer of cmu_deeplens
    # model = keras.Model(inputs=model.inputs, outputs=model.layers[155].output)
    model = keras.Model(inputs=model.inputs, outputs=model.layers[1].output)

    image = np.expand_dims(image, axis=0)
    feature_maps = model.predict(image)

    # TODO test for early convolutional layers
    # plot all 64 maps in an 8x8 squares
    square = 4
    ix = 1
    for _ in range(square):
        for _ in range(square):
            # specify subplot and turn of axis
            ax = plt.subplot(square, square, ix)
            ax.set_xticks([])
            ax.set_yticks([])
            # plot filter channel in grayscale
            plt.imshow(feature_maps[0, :, :, ix - 1], cmap='gray')
            ix += 1
    # show the figure
    plt.show()
    ###
    # # plot all 64 maps in an 8x8 squares
    # square = 8
    # ix = 1
    # for _ in range(square):
    #     for _ in range(square):
    #         # specify subplot and turn of axis
    #         ax = plt.subplot(square, square, ix)
    #         ax.set_xticks([])
    #         ax.set_yticks([])
    #         # plot filter channel in grayscale
    #         plt.imshow(feature_maps[0, :, :, ix - 1], cmap='gray')
    #         ix += 1
    # # show the figure
    # plt.show()


def visualize_features_all_conv(model, image):
    conv_layers = []
    for i, layer in enumerate(model.layers):
        # check for convolutional layer
        if 'conv' not in layer.name:
            continue
        # summarize output shape
        print(i, layer.name, layer.output.shape)
        conv_layers.append(i)
    print(conv_layers)

    # choosing all the convolutional layers of the model
    model = keras.Model(inputs=model.inputs,
                        outputs=[model.layers[i].output for i in conv_layers])

    image = np.expand_dims(image, axis=0)
    feature_maps = model.predict(image)

    square = 4
    for i, feature_map in enumerate(feature_maps):
        layer_name: str = model.layers[conv_layers[i]].name
        print(layer_name)
        # plot all 16 maps in an 4x4 square
        index = 1
        for _ in range(square):
            for _ in range(square):
                # specify subplot and turn of axis
                ax = plt.subplot(square, square, index)
                ax.set_xticks([])
                ax.set_yticks([])
                # plot filter channel in grayscale
                plt.imshow(feature_map[0, :, :, index - 1], cmap='gray')
                index += 1
        # plt.subplots_adjust(top=0.85)
        plt.suptitle(layer_name, size=16)
        # show the figure
        plt.show()


if __name__ == '__main__':
    print('Loading Model from Disk')
    # model_name = 'space_cas_swinburne_50epochs.h5'
    # model_name = 'space_lastro_epfl_250epochs.h5'
    model_name = 'space_cmu_deeplens_500epochs.h5'
    # model_name = 'space_wsi_net_250epochs.h5'
    # model_name = 'space_lens_flow_250epochs.h5'
    # model_name = 'space_lens_finder_250epochs.h5'
    model_path = os.path.join('..', 'models', model_name)
    if os.path.isfile(model_path):
        model = keras.models.load_model(model_path)
    else:
        sys.exit('Model at ' + model_path + ' was not found!')
    # model.summary()

    print('Loading Image from Disk')
    sample_image_path = os.path.join('D:', os.sep, 'datasets', 'GravitationalLensFindingChallenge',
                                     'Challenge1.0', 'Space',
                                     'Data_EuclidBig.0', 'Public', 'Band1', 'imageEUC_VIS-100003.fits')
    image = load_fits_image(sample_image_path)

    # display the input image
    plt.imshow(image, cmap='gray')
    plt.axis('off')
    plt.suptitle('input image: ' + sample_image_path.split(os.sep)[-1], size=16)
    plt.show()

    # visualize_features(model, image)

    visualize_features_all_conv(model, image)

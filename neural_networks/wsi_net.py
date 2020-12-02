from tensorflow import keras


# From: WSI-Net: Branch-Based and Hierarchy-Aware Network for Segmentation and Classification of Breast Histopathological Whole-Slide Images
def wsi_net(image_size: int, no_of_channels: int) -> keras.Model:
    input = keras.layers.Input((image_size, image_size, no_of_channels))

    x = input

    # conv0
    x = keras.layers.Conv2D(32, (7, 7), padding='same', activation='elu')(x)
    # x = keras.layers.BatchNormalization()(x)

    # res1
    x = resnet_block(x, 16, 32, False)

    # res2
    x = resnet_block(x, 32, 64, True)

    # Classification 'Branch'
    # 1x1 Conv
    x = keras.layers.Conv2D(32, (1, 1), padding='same', activation='elu')(x)
    # Group Normalisation
    # x = tfa.layers.GroupNormalization()(x)
    x = keras.layers.BatchNormalization()(x)
    # ReLU
    x = keras.layers.ReLU()(x)
    # 5x5 Conv
    x = keras.layers.Conv2D(32, (5, 5), padding='same', activation='elu')(x)
    # Group Normalisation
    # x = tfa.layers.GroupNormalization()(x)
    x = keras.layers.BatchNormalization()(x)
    # ReLU
    x = keras.layers.ReLU()(x)

    # x = SpatialPyramidPooling([1, 2, 4, 8])(x)
    x = keras.layers.MaxPool2D()(x)

    x = keras.layers.Flatten()(x)

    x = keras.layers.Dense(128)(x)

    output = keras.layers.Dense(1, activation='sigmoid')(x)

    model = keras.models.Model(input, output)

    return model


# import tensorflow_addons as tfa     # for Group Normalization layer
# from utils.keras_spp.spp.SpatialPyramidPooling import SpatialPyramidPooling
#
#
# def wsi_net(image_size: int, no_of_channels: int) -> keras.Model:
#     input = keras.layers.Input((image_size, image_size, no_of_channels))
#
#     x = input
#
#     # conv0
#     x = keras.layers.Conv2D(32, (7, 7), padding='same', activation='elu')(x)
#     # x = keras.layers.BatchNormalization()(x)
#
#     # res1
#     x = resnet_block(x, 16, 32, False)
#
#     # res2
#     x = resnet_block(x, 32, 64, True)
#
#     # Classification 'Branch'
#     # 1x1 Conv
#     x = keras.layers.Conv2D(32, (1, 1), padding='same', activation='elu')(x)
#     # Group Normalisation
#     x = tfa.layers.GroupNormalization()(x)
#     # ReLU
#     x = keras.layers.ReLU()(x)
#     # 5x5 Conv
#     x = keras.layers.Conv2D(32, (5, 5), padding='same', activation='elu')(x)
#     # Group Normalisation
#     x = tfa.layers.GroupNormalization()(x)
#     # ReLU
#     x = keras.layers.ReLU()(x)
#
#     print(x)
#
#     spp = SpatialPyramidPooling([1, 2, 4, 8])
#     output_shape = spp.compute_output_shape()
#     # x = SpatialPyramidPooling([1, 2, 4, 8])(x)
#     x = SpatialPyramidPooling([1])(x)
#     # x = keras.layers.MaxPool2D()(x)
#
#     print(x)
#
#     # x = keras.layers.Flatten()(x)   # Flattens seems to be done by SPP
#
#     x = keras.layers.Dense(256, input_shape=SpatialPyramidPooling.compute_output_shape())(x)
#
#     output = keras.layers.Dense(1, activation='sigmoid')(x)
#
#     model = keras.models.Model(input, output)
#
#     return model


def resnet_block(x, no_filters_in: int, no_filters_out: int, downsampling: bool):
    if downsampling:
        x = keras.layers.BatchNormalization()(x)
        x = keras.layers.ELU()(x)

        # shortcut branch
        x_shortcut = keras.layers.Conv2D(no_filters_out, (1, 1), strides=(2, 2), padding='same', activation='elu')(x)

        # convolution branch
        x = keras.layers.Conv2D(no_filters_in, (1, 1), strides=(2, 2), padding='same', activation='elu')(x)

        x = keras.layers.BatchNormalization()(x)
        x = keras.layers.ELU()(x)
        x = keras.layers.Conv2D(no_filters_in, (3, 3), padding='same', activation='elu')(x)

        x = keras.layers.BatchNormalization()(x)
        x = keras.layers.ELU()(x)
        x = keras.layers.Conv2D(no_filters_out, (1, 1), padding='same', activation='elu')(x)

        x = keras.layers.Add()([x, x_shortcut])
    else:
        x_shortcut = x

        x = keras.layers.BatchNormalization()(x)
        x = keras.layers.ELU()(x)
        x = keras.layers.Conv2D(no_filters_in, (1, 1), padding='same', activation='elu')(x)

        x = keras.layers.BatchNormalization()(x)
        x = keras.layers.ELU()(x)
        x = keras.layers.Conv2D(no_filters_in, (3, 3), padding='same', activation='elu')(x)

        x = keras.layers.BatchNormalization()(x)
        x = keras.layers.ELU()(x)
        x = keras.layers.Conv2D(no_filters_out, (1, 1), padding='same', activation='elu')(x)

        x = keras.layers.Add()([x, x_shortcut])

    return x


if __name__ == '__main__':
    model = wsi_net(101, 1)
    model.summary()

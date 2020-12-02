from tensorflow import keras


def cmu_deeplens_opt(image_size: int, no_of_channels: int, no_of_resnets_per_block: int,
                     no_of_resnet_triples: int, dropout: float) -> keras.Model:
    input = keras.layers.Input((image_size, image_size, no_of_channels))

    x = input

    x = keras.layers.Conv2D(32, (7, 7), padding='same', activation='elu')(x)
    x = keras.layers.BatchNormalization()(x)

    for i in range(no_of_resnet_triples):
        # set the number of filters to 2^i+5, so that the first block has 16 in_filters
        no_of_in_filters: int = 2 ** (i + 4)
        no_of_out_filters: int = 2 ** (i + 4 + 1)
        if i == 0:
            x = resnet_block(x, no_of_in_filters, no_of_out_filters, False)
        else:
            x = resnet_block(x, no_of_in_filters, no_of_out_filters, True)

        for j in range(no_of_resnets_per_block - 1):
            x = resnet_block(x, no_of_in_filters, no_of_out_filters, False)

        x = keras.layers.Dropout(dropout)(x)   # TODO does this work?

    x = keras.layers.AvgPool2D()(x)

    x = keras.layers.Flatten()(x)

    output = keras.layers.Dense(1, activation='sigmoid')(x)

    model = keras.models.Model(input, output)

    return model


# From: https://ui.adsabs.harvard.edu/abs/2018MNRAS.473.3895L/abstract
def cmu_deeplens(image_size: int, no_of_channels: int) -> keras.Model:
    input = keras.layers.Input((image_size, image_size, no_of_channels))

    x = input

    x = keras.layers.Conv2D(32, (7, 7), padding='same', activation='elu')(x)
    x = keras.layers.BatchNormalization()(x)

    x = resnet_block(x, 16, 32, False)
    x = resnet_block(x, 16, 32, False)
    x = resnet_block(x, 16, 32, False)

    x = resnet_block(x, 32, 64, True)
    x = resnet_block(x, 32, 64, False)
    x = resnet_block(x, 32, 64, False)

    x = resnet_block(x, 64, 128, True)
    x = resnet_block(x, 64, 128, False)
    x = resnet_block(x, 64, 128, False)

    x = resnet_block(x, 128, 256, True)
    x = resnet_block(x, 128, 256, False)
    x = resnet_block(x, 128, 256, False)

    x = resnet_block(x, 256, 512, True)
    x = resnet_block(x, 256, 512, False)
    x = resnet_block(x, 256, 512, False)

    x = keras.layers.AvgPool2D()(x)

    x = keras.layers.Flatten()(x)

    output = keras.layers.Dense(1, activation='sigmoid')(x)

    model = keras.models.Model(input, output)

    return model


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
    model = cmu_deeplens(101, 1)
    model.summary()

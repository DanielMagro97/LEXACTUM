from tensorflow import keras


# From: https://ui.adsabs.harvard.edu/abs/2017MNRAS.471..167J/abstract
def cas_swinburne(image_size: int, no_of_channels: int) -> keras.Model:
    model = keras.Sequential()

    model.add(keras.layers.Conv2D(96, (11, 11), padding='same', strides=1,
                                  activation='relu',
                                  input_shape=(image_size, image_size, no_of_channels)))
    # model.add(keras.layers.MaxPool2D(pool_size=(2, 2)))
    model.add(keras.layers.MaxPool2D(pool_size=(3, 3)))

    model.add(keras.layers.Conv2D(128, (5, 5), padding='same', strides=1,
                                  activation='relu'))
    model.add(keras.layers.MaxPool2D(pool_size=(3, 3)))

    model.add(keras.layers.Conv2D(256, (3, 3), padding='same', strides=1,
                                  activation='relu'))
    model.add(keras.layers.MaxPool2D(pool_size=(3, 3)))

    model.add(keras.layers.Flatten())

    model.add(keras.layers.Dense(1024, activation='relu'))
    model.add(keras.layers.Dropout(0.5))

    model.add(keras.layers.Dense(1024, activation='relu'))
    model.add(keras.layers.Dropout(0.5))

    model.add(keras.layers.Dense(1, activation='sigmoid'))

    return model


if __name__ == '__main__':
    model = cas_swinburne(101, 1)
    model.summary()

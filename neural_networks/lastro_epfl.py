from tensorflow import keras


# From: https://ui.adsabs.harvard.edu/abs/2018A%26A...611A...2S/abstract
def lastro_epfl(image_size: int, no_of_channels: int) -> keras.Model:
    model = keras.Sequential()

    model.add(keras.layers.Conv2D(16, (4, 4), padding='valid', strides=1, activation='relu',
                                  input_shape=(image_size, image_size, no_of_channels)))
    model.add(keras.layers.Conv2D(16, (3, 3), padding='valid', strides=1, activation='relu'))
    model.add(keras.layers.MaxPool2D(pool_size=(2, 2)))
    model.add(keras.layers.BatchNormalization())

    model.add(keras.layers.Conv2D(32, (3, 3), padding='valid', strides=1, activation='relu'))
    model.add(keras.layers.Conv2D(32, (3, 3), padding='valid', strides=1, activation='relu'))
    model.add(keras.layers.MaxPool2D(pool_size=(2, 2)))
    model.add(keras.layers.BatchNormalization())

    model.add(keras.layers.Conv2D(64, (3, 3), padding='valid', strides=1, activation='relu'))
    model.add(keras.layers.Conv2D(64, (3, 3), padding='valid', strides=1, activation='relu'))
    model.add(keras.layers.MaxPool2D(pool_size=(2, 2)))
    model.add(keras.layers.BatchNormalization())

    model.add(keras.layers.Dropout(0.1))

    model.add(keras.layers.Conv2D(128, (3, 3), padding='valid', strides=1, activation='relu'))
    model.add(keras.layers.Dropout(0.1))

    model.add(keras.layers.Conv2D(128, (3, 3), padding='valid', strides=1, activation='relu'))
    model.add(keras.layers.BatchNormalization())
    model.add(keras.layers.Dropout(0.1))

    model.add(keras.layers.Flatten())

    model.add(keras.layers.Dense(1024, activation='relu'))
    model.add(keras.layers.Dropout(0.1))
    model.add(keras.layers.Dense(1024, activation='relu'))
    model.add(keras.layers.Dropout(0.1))
    model.add(keras.layers.Dense(1024, activation='relu'))
    model.add(keras.layers.BatchNormalization())

    model.add(keras.layers.Dense(1, activation='sigmoid'))

    return model


if __name__ == '__main__':
    model = lastro_epfl(101, 1)
    model.summary()

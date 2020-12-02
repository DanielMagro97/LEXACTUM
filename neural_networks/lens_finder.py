from tensorflow import keras


# From: https://www.researchgate.net/publication/323359284_Auto-detection_of_strong_gravitational_lenses_using_convolutional_neural_networks
def lens_finder(image_size: int, no_of_channels: int) -> keras.Model:
    model = keras.Sequential()

    model.add(keras.layers.Conv2D(64, (5, 5), padding='same', strides=1, activation='relu',
                                  input_shape=(image_size, image_size, no_of_channels)))
    model.add(keras.layers.MaxPool2D())

    model.add(keras.layers.Conv2D(128, (3, 3), padding='same', strides=1, activation='relu'))
    model.add(keras.layers.MaxPool2D())

    model.add(keras.layers.Flatten())

    model.add(keras.layers.Dense(128, activation='relu'))

    model.add(keras.layers.Dense(1, activation='sigmoid'))

    return model


if __name__ == '__main__':
    model = lens_finder(101, 1)
    model.summary()

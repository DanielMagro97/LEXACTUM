from tensorflow import keras


# From: https://arxiv.org/pdf/1705.05857.pdf
def lens_flow(image_size: int, no_of_channels: int) -> keras.Model:
    model = keras.Sequential()

    model.add(keras.layers.AvgPool2D((3, 3), strides=(3, 3),
                                     input_shape=(image_size, image_size, no_of_channels)))

    model.add(keras.layers.Conv2D(16, (5, 5), padding='same', strides=1, activation='relu'))
    model.add(keras.layers.MaxPool2D((2, 2), strides=(2, 2)))

    model.add(keras.layers.Conv2D(25, (5, 5), padding='same', strides=1, activation='relu'))
    model.add(keras.layers.MaxPool2D((2, 2), strides=(2, 2)))

    model.add(keras.layers.Conv2D(36, (4, 4), padding='same', strides=1, activation='relu'))
    model.add(keras.layers.MaxPool2D((2, 2), strides=(2, 2)))

    model.add(keras.layers.Flatten())

    model.add(keras.layers.Dense(128, activation='relu'))
    model.add(keras.layers.Dropout(0.5))

    model.add(keras.layers.Dense(1, activation='sigmoid'))

    return model


if __name__ == '__main__':
    model = lens_flow(101, 1)
    model.summary()

# Copyright (C) 2020  Daniel Magro
# Full License at: https://github.com/DanielMagro97/LEXACTUM/blob/main/LICENSE

from typing import List             # for type annotation
import numpy as np                  # for numpy arrays
from astropy.io import fits         # for opening .fits files
import astropy.visualization        # for ZScaleInterval (data normalisation)


# Function which loads a single .fits file as an image_sizeximage_sizex1 np.ndarray, given its path
def load_fits_image(fits_file_path: str, fits_file_normalisation: str = 'ZScale') -> np.ndarray:
    image_data: np.ndarray = fits.getdata(fits_file_path)

    # image normalisation, according to Hyper Parameters (ZScale etc)
    if fits_file_normalisation == 'none':
        pass
    elif fits_file_normalisation == 'ZScale':
        # normalise the data using astropy's ZScaleInterval
        interval = astropy.visualization.ZScaleInterval()
        image_data = interval(image_data)

    # add another dimension so that the image becomes 128 x 128 x 1 (for tensorflow)
    image_data = np.expand_dims(image_data, axis=2)

    return image_data


# Function which loads a single .fits file as an image_sizeximage_sizex1 np.ndarray, given its path
def load_fits_four_channel(fits_file_paths: List[str], fits_file_normalisation: str = 'ZScale') -> np.ndarray:
    # create a list of numpy arrays which will store each channel as a numpy array
    channels: List[np.ndarray] = []
    # iterate over every channel's file path
    for channel in fits_file_paths:
        # load the current channel's data (fits file) from disk
        image_data: np.ndarray = fits.getdata(channel)

        # image normalisation, according to Hyper Parameters (ZScale etc)
        if fits_file_normalisation == 'none':
            pass
        elif fits_file_normalisation == 'ZScale':
            # normalise the data using astropy's ZScaleInterval
            interval = astropy.visualization.ZScaleInterval()
            image_data = interval(image_data)

        # add the loaded and normalised channel to the list of numpy arrays
        channels.append(image_data)

    # stack the channels such that they become an image_size x image_size x no_of_channels array (for tensorflow)
    image_data: np.ndarray = np.stack(channels, 2)

    return image_data

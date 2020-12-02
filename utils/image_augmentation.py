# Copyright (C) 2020  Daniel Magro
# Full License at: https://github.com/DanielMagro97/LEXACTUM/blob/main/LICENSE

import numpy as np                  # for numpy arrays
import imgaug.augmenters as iaa     # for augmenting images


def initialise_augmenter():
    # Horizontal and Vertical Flips (set to 1 as the SomeOf function will choose when to apply these itself)
    horizontal_flip = iaa.Fliplr(1)#0.5)
    vertical_flip = iaa.Flipud(1)#0.5)

    # 90, 180 and 270 degree rotations
    rotate_90 = iaa.Affine(rotate=90)
    rotate_180 = iaa.Affine(rotate=180)
    rotate_270 = iaa.Affine(rotate=270)

    # Translations of -10% to 10% of the image's pixels
    translate_x = iaa.TranslateX(percent=(-0.1, 0.1))
    translate_y = iaa.TranslateY(percent=(-0.1, 0.1))

    # Scale the image between 0.75 and 1.1 of the original size
    scale_x = iaa.ScaleX((0.75, 1.1))
    scale_y = iaa.ScaleY((0.75, 1.1))

    # Shear the image between -20 and 20 degrees
    shear_x = iaa.ShearX((-20, 20))
    shear_y = iaa.ShearY((-20, 20))

    augmentation = iaa.SomeOf((0, None), [horizontal_flip, vertical_flip,
                                          iaa.OneOf([rotate_90, rotate_180, rotate_270]),
                                          translate_x, translate_y,
                                          scale_x, scale_y,
                                          shear_x, shear_y], random_order=True)

    return augmentation


# def augment_image(image: np.ndarray, augmenter: iaa) -> np.ndarray:
#     augmented_image: np.ndarray = augmenter(image=image)
#     # augmented_image = augmenter.augment_image(image)
#
#     return augmented_image
#
#
# def augment_images(images: np.ndarray, augmenter: iaa) -> np.ndarray:
#     augmented_images: np.ndarray = augmenter(images=images)
#     # augmented_image = augmenter.augment_images(image)
#
#     return augmented_images


if __name__ == '__main__':
    augmenter = initialise_augmenter()

    import imgaug as ia
    from utils.load_fits import load_fits_image  # for loading fits images with optional normalisation
    from utils.load_fits import load_fits_four_channel

    # one image one channel
    image = load_fits_image(r'D:\datasets\GravitationalLensFindingChallenge\Challenge1.0\Space\Data_EuclidBig.0\Public\Band1\imageEUC_VIS-100088.fits')
    ia.imshow(image[:, :, 0])
    # augmented_image = augment_image(image, augmenter)
    augmented_image = augmenter(image=image)
    ia.imshow(augmented_image[:, :, 0])
    print(augmented_image.shape)

    # 4 images one channel each
    image_1 = load_fits_image(r'D:\datasets\GravitationalLensFindingChallenge\Challenge1.0\Space\Data_EuclidBig.0\Public\Band1\imageEUC_VIS-100088.fits')
    image_2 = load_fits_image(r'D:\datasets\GravitationalLensFindingChallenge\Challenge1.0\Space\Data_EuclidBig.0\Public\Band1\imageEUC_VIS-100089.fits')
    image_3 = load_fits_image(r'D:\datasets\GravitationalLensFindingChallenge\Challenge1.0\Space\Data_EuclidBig.0\Public\Band1\imageEUC_VIS-100101.fits')
    image_4 = load_fits_image(r'D:\datasets\GravitationalLensFindingChallenge\Challenge1.0\Space\Data_EuclidBig.0\Public\Band1\imageEUC_VIS-100164.fits')
    images = np.empty((4, image_1.shape[0], image_1.shape[1], image_1.shape[2]))
    images[0] = image_1
    images[1] = image_2
    images[2] = image_3
    images[3] = image_4
    print(images.shape)
    ia.imshow(np.hstack(np.reshape(images, (images.shape[0], images.shape[1], images.shape[2]))))
    # augmented_images = augment_images(images, augmenter)
    augmented_images = augmenter(images=images)
    ia.imshow(np.hstack(np.reshape(augmented_images, (images.shape[0], images.shape[1], images.shape[2]))))
    print(augmented_images.shape)

    # one image 4 channels
    paths = [r'D:\datasets\GravitationalLensFindingChallenge\Challenge1.0\Ground\Data_KiDS_Big.0\Public\Band1\imageSDSS_R-100006.fits',
             r'D:\datasets\GravitationalLensFindingChallenge\Challenge1.0\Ground\Data_KiDS_Big.0\Public\Band2\imageSDSS_I-100006.fits',
             r'D:\datasets\GravitationalLensFindingChallenge\Challenge1.0\Ground\Data_KiDS_Big.0\Public\Band3\imageSDSS_G-100006.fits',
             r'D:\datasets\GravitationalLensFindingChallenge\Challenge1.0\Ground\Data_KiDS_Big.0\Public\Band4\imageSDSS_U-100006.fits']
    img_4_channel = load_fits_four_channel(paths)

    ia.imshow(img_4_channel)
    # img_4_aug = augment_image(img_4_channel, augmenter)
    img_4_aug = augmenter(image=img_4_channel)
    ia.imshow(img_4_aug)
    print(img_4_aug.shape)

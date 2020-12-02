# LEXACTUM

LEXACTUM (Lens EXtrActor CaTania University of Malta) was developed to explore new solutions for the
[Gravitational Lens Finding Challenge 1.0](http://metcalf1.difa.unibo.it/blf-portal/gg_challenge.html), primarily those
based on Convolutional Neural Networks (CNNs).

## Main Features

* The first of its main features are the Image Augmentation techniques implemented in `image_augmentation.py`, which can be toggled on or off to train for longer without overfitting.
* Another feature is the modularity of the code, allowing for the rather easy development of new models, with very easy integration of new models into the pipeline.
* Other features include the ability to set parameters from the command line. Examples of such parameters are the dataset path, whether to train a model or load one from disk, the name of the model (to train or load), the number of epochs to train for, the batch size, and whether to use image augmentation during training or not.
* LEXACTUM also uses a custom `Data Generator`, which loads and preprocesses images in batches with the CPU, while the GPU can train on the last batch of images. Apart from image augmentation during training, all images are normalised using ZScale. Like other components, the normalisation method can be easily swapped out for other techniques.
* Furthermore, LEXACTUM comes with a `results` package, which scores the trained models. This script plots the Area Under the Curve (AUC) curve, and also finds other metrics such as the TPR<sub>0</sub> and TPR<sub>10</sub> defined for this challenge.
* Moreover, LEXACTUM saves trained models to disk, and also provides functionality for loading trained models.
* Finally, `visualise_features.py` was created, which allows for the viewing of the feature maps at every convolutional layer that a trained model is 'looking at' during execution.

## Installing Dependencies

All the required Python packages can be installed by running this command:
```bash
pip install astropy tensorflow-gpu numpy pandas scikit-learn imgaug matplotlib
```

## Training a New Model

To train a model on the Space set, run the following command, substituting `<>`s with your own values.
```bash
python Space.py --dataset <path_to_dataset> --train_or_load train --model_name <name-of-model> --no_of_epochs <number-of-epochs> 
```

To train a model on the Ground set, run the following command instead:
```bash
python Ground.py --dataset <path_to_dataset> --train_or_load train --model_name <name-of-model> --no_of_epochs <number-of-epochs> 
```

Accepted values for `<name-of-model>` are:\
`cas_swinburne`, `lastro_epfl`, `cmu_deeplens`, `wsi_net`, `lens_flow` or `lens_finder`

The trained model will then be saved in the `models` directory as `models/<space/ground>_<model-name>_<number-of-epochs>epochs.h5`.

## Loading a Pre-Trained Model

To run a trained model on the Space set, run the following command, substituting `<>`s with your own values.
```bash
python Space.py --dataset <path_to_dataset> --train_or_load load --model_name <name-of-model> 
```

To run a trained model on the Ground set, run the following command instead.
```bash
python Ground.py --dataset <path_to_dataset> --train_or_load load --model_name <name-of-model> 
```

For `<name-of-model>`, use the name of the `.h5` file stored inside the `models/` folder, excluding `models/`,
but including the extension `.h5`.

## Visualising the Feature Maps of a Trained Model

To visualise the feature maps of a trained model at every convolutional layer, the `visualise_features.py` script can be used.
At around `line 105` of the code, set the variable `model_name` to the name of the trained model inside the `models`
directory that you would like to visualise the convolutional layers of. Do not include `models/`, but do include the
`.h5` extension.
On `line 117`, `sample_image_path` can be set to the path of any image that will be passed through the model, for
which the outputs of the convolutional layers will be visualised. 

Once these variables have been set, the `visualise_features.py` script can be run, and any convolutional layers will be
detected automatically, and displayed at the end.

## Command-Line Arguments

A description of accepted command-line arguments can be found by running:
```bash
python Space.py --help
# or
python Ground.py --help
```

Here is a short description of each recognised parameter
* `--dataset` - The path to the directory containing the dataset
* `--train_or_load` - Whether to train a model, or to load one from disk - Accepted values: `train` or `load`
* `--model_name` - The name of the model to train - Accepted values: `cas_swinburne`, `lastro_epfl`, `cmu_deeplens`,
`wsi_net`, `lens_flow` or `lens_finder`\
Or the name of the file in the models folder to load from disk (including .h5)
* `--no_of_epochs` - The number of epochs to train the model for
* `--batch_size` - Number of images per batch
* `--augment_images` - Whether to perform Image Augmentation on the training data while training - Accepted values: `True` or `False`

# Results

These are the results obtained by the specified models when trained for the specified number of epochs, using image
augmentation. The results presented are the TPR, FPR, AUC, TPR<sub>0</sub>, TPR<sub>10</sub> and the average time
taken to execute the model on a given image.

## Space Results

| Model Name    | No. of Training Epochs | TPR    | FPR    | AUC    | TPR<sub>0</sub> | TPR<sub>10</sub> | Avg. Execution Time per Image (seconds) |
|---------------|------------------------|--------|--------|--------|-----------------|------------------|-----------------------------------------|
| CAS Swinburne | 5                      | 0.5250 | 0.0603 | 0.8489 | 0.1531          | 0.1861           | 0.0124                                  |
|               | 10                     | 0.5517 | 0.1077 | 0.8171 | 0.1054          | 0.1509           |                                         |
|               | 25                     | 0.7221 | 0.1178 | 0.8870 | 0.0000          | 0.2705           |                                         |
|               | 50                     | 0.6252 | 0.0461 | 0.8894 | 0.2411          | 0.3000           |                                         |
|               | 75                     | 0.6503 | 0.0474 | 0.8963 | 0.0000          | 0.3221           |                                         |
|               | 100                    | 0.6604 | 0.0591 | 0.8915 | 0.0000          | 0.3016           |                                         |
|               | 500                    | 0.6551 | 0.0295 | 0.9086 | 0.0000          | 0.3602           |                                         |
| Lastro EPFL   | 5                      | 0.3507 | 0.0042 | 0.8641 | 0.1539          | 0.2112           | 0.0061                                  |
|               | 10                     | 0.7302 | 0.3543 | 0.7825 | 0.1894          | 0.2455           |                                         |
|               | 50                     | 0.6650 | 0.0287 | 0.9132 | 0.2107          | 0.3823           |                                         |
|               | 250                    | 0.7937 | 0.0687 | 0.9322 | 0.0000          | 0.2268           |                                         |
| CMU Deeplens  | 5                      | 0.6056 | 0.1539 | 0.7984 | 0.0000          | 0.1206           | 0.0061                                  |
|               | 10                     | 0.8268 | 0.2880 | 0.8710 | 0.0000          | 0.2309           |                                         |
|               | 25                     | 0.8738 | 0.2726 | 0.9113 | 0.0000          | 0.0000           |                                         |
|               | 50                     | 0.7570 | 0.0628 | 0.9243 | 0.0000          | 0.4073           |                                         |
|               | 100                    | 0.8170 | 0.1321 | 0.9226 | 0.0000          | 0.0000           |                                         |
|               | 250                    | 0.7592 | 0.0436 | 0.9291 | 0.0000          | 0.0000           |                                         |
|               | 500                    | 0.7952 | 0.0626 | 0.9343 | 0.0000          | 0.0000           |                                         |
|               | 1000                   | 0.8611 | 0.1634 | 0.9303 | 0.0000          | 0.0000           |                                         |
| WSI Net       | 5                      | 0.7132 | 0.2955 | 0.7935 | 0.0000          | 0.0000           | 0.0055                                  |
|               | 10                     | 0.5437 | 0.0187 | 0.8867 | 0.1799          | 0.2934           |                                         |
|               | 50                     | 0.7888 | 0.1194 | 0.9115 | 0.0000          | 0.0000           |                                         |
|               | 100                    | 0.7348 | 0.0624 | 0.9069 | 0.0000          | 0.3976           |                                         |
|               | 250                    | 0.7255 | 0.0531 | 0.9083 | 0.0000          | 0.4211           |                                         |
| Lens Flow     | 5                      | 0.6508 | 0.1520 | 0.8389 | 0.0728          | 0.1260           | 0.0054                                  |
|               | 25                     | 0.6431 | 0.0726 | 0.8799 | 0.1903          | 0.2704           |                                         |
|               | 100                    | 0.6780 | 0.0636 | 0.8963 | 0.0000          | 0.3379           |                                         |
|               | 250                    | 0.7384 | 0.0889 | 0.9046 | 0.0000          | 0.3632           |                                         |
| Lens Finder   | 5                      | 0.4915 | 0.1001 | 0.8038 | 0.0885          | 0.1056           | 0.0197                                  |
|               | 25                     | 0.6203 | 0.0663 | 0.8739 | 0.2103          | 0.2395           |                                         |
|               | 100                    | 0.6912 | 0.0855 | 0.8857 | 0.0000          | 0.2721           |                                         |
|               | 250                    | 0.7651 | 0.1062 | 0.9056 | 0.0000          | 0.3739           |                                         |

## Ground Results

| Model Name          | No. of Training Epochs | TPR    | FPR    | AUC    | TPR<sub>0</sub> | TPR<sub>10</sub> | Avg. Execution Time per Image (seconds) |
|---------------------|------------------------|--------|--------|--------|-----------------|------------------|-----------------------------------------|
| **CAS   Swinburne** | 10                     | 0.8779 | 0.1077 | 0.9608 | 0.0000          | 0.0000           | 0.0469                                  |
|                     | 50                     | 0.8995 | 0.0944 | 0.9720 | 0.0000          | 0.0000           |                                         |
|                     | 100                    | 0.8565 | 0.0406 | 0.9742 | 0.0000          | 0.0000           |                                         |
|                     | 250                    | 0.8726 | 0.0429 | 0.9758 | 0.0000          | 0.0000           |                                         |
| **Lastro EPFL**     | 50                     | 0.9073 | 0.0536 | 0.9824 | 0.0000          | 0.5133           | 0.0429                                  |
|                     | 100                    | 0.9110 | 0.0482 | 0.9844 | 0.0000          | 0.5504           |                                         |
|                     | 250                    | 0.9197 | 0.0489 | 0.9862 | 0.0000          | 0.0000           |                                         |
| **CMU Deeplens**    | 25                     | 0.7733 | 0.0232 | 0.9588 | 0.0000          | 0.3840           | 0.0594                                  |
|                     | 50                     | 0.9138 | 0.0568 | 0.9825 | 0.6046          | 0.6827           |                                         |
|                     | 75                     | 0.9026 | 0.0550 | 0.9804 | 0.0000          | 0.6536           |                                         |
|                     | 100                    | 0.9333 | 0.0660 | 0.9851 | 0.0000          | 0.6673           |                                         |
|                     | 150                    | 0.9205 | 0.0445 | 0.9870 | 0.0000          | 0.7042           |                                         |
|                     | 250                    | 0.8593 | 0.0858 | 0.9570 | 0.0000          | 0.0000           |                                         |
| **WSI Net**         | 50                     | 0.8560 | 0.0589 | 0.9620 | 0.0000          | 0.0000           | 0.0231                                  |
|                     | 100                    | 0.8218 | 0.0301 | 0.9710 | 0.0000          | 0.5347           |                                         |
|                     | 250                    | 0.9127 | 0.0864 | 0.9742 | 0.0000          | 0.0000           |                                         |
| **Lens Flow**       | 50                     | 0.8784 | 0.0744 | 0.9708 | 0.0000          | 0.5101           | 0.0349                                  |
|                     | 100                    | 0.8831 | 0.0738 | 0.9726 | 0.0000          | 0.5648           |                                         |
|                     | 250                    | 0.9006 | 0.0733 | 0.9758 | 0.0000          | 0.0000           |                                         |
| **Lens Finder**     | 50                     | 0.8556 | 0.0648 | 0.9665 | 0.0000          | 0.4442           | 0.0293                                  |
|                     | 100                    | 0.8938 | 0.0805 | 0.9718 | 0.0000          | 0.5664           |                                         |
|                     | 250                    | 0.8997 | 0.0880 | 0.9671 | 0.0000          | 0.0000           |                                         |
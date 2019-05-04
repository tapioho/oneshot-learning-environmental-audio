# Oneshot Learning with Siamese Networks for Environmental Audio
The purpose of this reposirory is to provide a working example of a oneshot learning implementation utilizing siamese networks for environmental audio classification.

## Table of contents
1. [Dataset](#dataset)
2. [Model](#model)
3. [How to use](#how-to-use)


## Dataset <a name="dataset"></a>
For this example script, the ESC-50 dataset is used. The dataset is available here: [ESC-50](https://github.com/karoldvl/ESC-50)
Clone the repository to the root of this repository, or alternative change the data path variable in main.py

## Model <a name="model"></a>
The model consists of two convolutional input networks, followed by
a merging layer and a final output layer. The input networks share the same architecture
and weights in order to act as identical encoding layers for both inputs. This also
means that the weights are updated simultaneously for both networks during training.
The proposed architecture for a single input network consists of several convolutional
blocks followed by a fully connected layer.

## How to use <a name="how-to-use"></a>
1. Download [ESC-50](https://github.com/karoldvl/ESC-50) and move it to the root of this repository (or change the data path variable in main.py)
2. Run main.py. The script should start by calculating a mel-log scaled spectrogram for each audio sample from the dataset. The spectra are then saved for future, i.e. the next time the script is run the spectra are not required to be calculated.
3. Next, the script will start the training procedure.
4. Finally, the results for the evaluation are calculated, visualized and saved.

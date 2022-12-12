Project 1: Armageddon - The hazard of small asteroids
=====================================================

Synopsis:
---------

This project was pursued for an open competition Federal Emergency Managment Agency (FEMA) aimed to improve emergency orotocols under hurricane threats. Here below you can find our implementation for the problem released.

 Hurricanes can cause upwards of 1,000 deaths and $50 billion in damages in a single event, and have been responsible for well over 160,000 deaths globally in recent history. During a tropical cyclone, humanitarian response efforts hinge on accurate risk approximation models that can help predict optimal emergency strategic decisions.
 
 This achine Learning model predicts how a hurracane Evolves in the near future.

Problem definition
------------------
The Competition Provided a dataset of 494 NASA Satellite images of tropical storms. They are of a varied length (4 - 648, avg 142) and labelled by id, ocean (1 or 2) and wind speed. The dataset can be reached here: https://mlhub.earth/data/nasa_tropical_storm_competition .

Our Objective was, given one active hurricane where some satellite images have already been made available, to generate a ML/DL-based solution able to generate as many future image predictions as possible based on these existing images for that given storm. We have used only storms with more than 100 samples and design, train, and present your results in the video found in the repository.

Additional sections
~~~~~~~~~~~~~~~~~~~

We have two main features in our package:

* Preprocessing: This is the script that contains functions to upload and manipulate the data and prepare it for learning.

* Network: Here instead we build our network to learn and predict a selected storm from the dataset.

Please note we have left the package flaxible to personalisations, thus performance is mainly achieved through hyper-parameter tuning and optimisation.

Out software is deployed as a package with two scripts containing most of the functions. We have implemeted a Colvolutional Long Short Term Memory network.

In script preprocessing.py we have two classes:

* StormTensorDataset: class to format the stor dataset and has two main objects
    - len to find out lenght pf dataset
    - getitm to an image and its target.

* Preprocessor: this class is to prepare and format data to go through the trainig network.
    - data_download: download data to a path in directory
    - select_storm: slect a storm to predict through its id
    - get_mean_std: get mean a standard deviation of storm data
    - create_datasets_dataloaders: create training and validation sets and from those create dataloaders

In the Network.py instead we have:

- an initial function to switch to check for GPu when in collab

* ConvLSTMCell: This creates the convolution operator whithin our LSTM network.
    - Forward: function that defines a forward pass theough the convolutional cell

* ConvLSTM: This class creates an LSTM network from our prvious convolution cell
    - forward: Thus is an object that crated a forward pass through our ConvLSTM architecture.

* Seq2Seq: this class let us create a network of arbitrary ConvLSTM layers to give us the flexibility to try different set ups
    - forward: creates forward pass through all the layers in our network.

In addition, we have further 3 functions that help us manipulate the network:

- train_conv_lstm: function to train our network.

- validate_conv_lstm: function to validate your network:

- eval_images: function generating the predicted images and comparing to the test set.


Function API
============

Python 3.8 is used in this implementation used

containing the following packages in requiroments.txt
```
numpy >= 1.13.0
ipython
scipy
sympy
pandas
matplotlib
mpltools
pytest
pytest-timeout
sphinx
seaborn
pillow
torch
sklearn
pycm
livelossplot
Torchvision
radiant_mlhub
torchsummary
ipython
wand
```

* Preprocessing:

```
import tarfile
from pathlib import Path
from glob import glob
import numpy as np
from radiant_mlhub import Dataset
import matplotlib.image as mpimg
import torch
from torch.utils.data import TensorDataset, DataLoader
import torchvision.transforms as transforms
```

* Network:

```
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torchvision.transforms as transforms
```

All the rest are used in the notebooks for analysis/visualisation of results.

.. automodule:: preprocessing
  :members: StormTensorDataset

.. automodule:: preprocessing
  :members: Preprocessor
  :noindex: preprocessing


.. automodule:: Network
  :members: ConvLSTMCell

.. automodule:: Network
  :members: ConvLSTM
  :noindex: Network

.. automodule:: Network
  :members: Seq2Seq
  :noindex: Network

.. automodule:: Network
  :members: train_conv_lstm
  :noindex: Network

.. automodule:: Network
  :members: validate_conv_lstm
  :noindex: Network

.. automodule:: Network
  :members: eval_images
  :noindex: Network
  
  

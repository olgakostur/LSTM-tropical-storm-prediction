import tarfile
from pathlib import Path
from glob import glob
import numpy as np
from radiant_mlhub import Dataset
import matplotlib.image as mpimg
import torch
from torch.utils.data import TensorDataset, DataLoader
import torchvision.transforms as transforms


class StormTensorDataset(TensorDataset):
    """
    Custom class pf dataset designed specifically for storm images.
    Has a parameter sequence_lenghth = 5 corresponding to 5 images.
    """
    def __init__(self, data, transform=None, sequence_len=5):
        """
        Set up the fileter-parameters and constants for the convolution

        Parameters
        ----------
        data : pytorch tensor
            A tensor containing the data e.g. images

        transform: callable, optional
            Optional transform to be applied on a sample.

        sequence_len: int
            number of images in each sequence

        Examples
        --------
        >>> prepro = Preprocessor()
        >>> data = torch.Tensor([1,1,1,1,1])
        >>> Tensor_Dataset = StormTensorDataset(data)
        >>> len(Tensor_Dataset)
        0
        """
        self.data = data
        self.transform = transform
        self.sequence_len = sequence_len

    def __len__(self):
        """
        return differnce between darta length and sequence lenght
        """
        return len(self.data) - self.sequence_len

    def __getitem__(self, idx):
        """
        get single item in data

        Parameters
        ----------

        idx : int
            index of data sample

        returns
        ______

        sample: pytorch.tensor
            Tensor rapresenting an image

        targer: pytorch.tensor
            Tensor rapresentin class of image

        """
        # sample = mpimg.imread(self.data[idx])
        if self.transform:
            sample = torch.stack([self.transform(self.data[idx+i])
                                 for i in range(self.sequence_len)])
            sample = torch.transpose(sample, 0, 1)
            target = self.transform(self.data[idx+self.sequence_len])
        return sample, target


class Preprocessor():
    """
    Loads the data, filters images to select storm by id,
    creates datasets and loaders.
    """
    def data_download(self, path):
        """
        Download data to a path. API key needs to be obtained
        and pasted after executing:
        ! mlhub configure.
        Then Google Drive must be mounted:
        from google.colab import drive
        drive.mount('/content/drive')

        Parameters
        ----------
        path : string
            path to folder where all the folders with indicidual images are
            stored The structure is specific to the projcet's dataset.

            Hidden state
        """
        download_dir = Path(path).expanduser().resolve()
        data_set = Dataset.fetch('nasa_tropical_storm_competition')
        archive_paths = data_set.download(download_dir)
        for archive_path in archive_paths:
            print(f'Extracting {archive_path}...')
            with tarfile.open(archive_path) as tfile:
                tfile.extractall(path=download_dir)
        print('Done')

    def select_storm(self, path, storm_id):
        """
        Function to select storm images by the storm id.

        Parameters
        ----------
        path : string
            path to folder where all the folders with indicidual images are
            stored The structure is specific to the projcet's dataset.

        storm_id: string
            list of arrays (filetered images corresponding to storm
            id in sequential order)

        Returns
        -------
        storn_1: list of arrays
            filetered images corresponding to storm id in sequential order

        """
        jpg = sorted(glob(path))
        jpg_id = []
        # storm id is stored between index -17 and -14
        jpg_id = [jpg[i]
                  for i, j in enumerate(jpg) if jpg[i][-17:-14] == storm_id]
        storm_1 = [mpimg.imread(jpg_id[i]) for i in range(len(jpg_id))]
        return storm_1

    def get_mean_std(self, storm):
        """
        Calculates mean and standard deviation of storm images

        Parameters
        ----------
        storm : list of arrays
            the filtered storm

        storm_id: string
            list of arrays (filetered images corresponding to storm
            id in sequential order)

        Returns
        -------
        mean: float
            mean of images

        std: float
            standard deviattion of images

        """
        mean = 0.
        std = 0.
        for image in storm:
            mean += np.mean(image)
            std += np.std(image)

        mean /= len(storm)
        std /= len(storm)
        return mean, std

    def create_datasets_dataloaders(self, storm, train_franction,
                                    sequence_len):
        """
        Takes in filtered list of storms and parameters for dataset creation
        and returns custom datasets and loaders.

        Parameters
        ----------
        storm : list of arrays
            subset of dataset filtered by particular storm

        train_franction: float
            how much data will be in training subset

        sequence_len: int
            length of sequence

        Returns
        -------
        train_data: pytorch dataset
            dataset set for training

        val_data: pytorch dataset
            dataset set for validation

        train_loader: pytorch dataloader
            dataloader for triaing

        val_loader: pytorch dataloader
            dataloader for validation

        """
        train_img = storm[:int(len(storm)*train_franction)]
        val_img = storm[int(len(storm)*train_franction):]
        transform = transforms.Compose([transforms.ToTensor()])
        train_data = StormTensorDataset(train_img, transform, sequence_len)
        val_data = StormTensorDataset(val_img, transform, sequence_len)

        train_loader = DataLoader(train_data, batch_size=6, shuffle=False)
        val_loader = DataLoader(val_data, batch_size=6, shuffle=False)
        return train_data, val_data, train_loader, val_loader

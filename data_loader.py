import numpy as np
from utils import plot_images

import torch
from torchvision import datasets
from torchvision import transforms
from torch.utils.data.sampler import SubsetRandomSampler

import os
from PIL import Image
from torch.utils.data import Dataset
import random

from datasets import CIFAR_RAM, CIFAR_RAM_TEST, PlacesDataSet, PlacesDataSet2, HDF5Dataset, PlacesGlimpseDataSet



def get_train_valid_loader(data_dir,
                           batch_size,
                           random_seed,
                           valid_size=0.1,
                           shuffle=True,
                           show_sample=False,
                           num_workers=4,
                           pin_memory=False,
                           train_file=None,
                           val_file=None):
    """
    Utility function for loading and returning train and valid
    multi-process iterators over the specified dataset. A sample
    9x9 grid of the images can be optionally displayed.

    If using CUDA, num_workers should be set to 1 and pin_memory to True.

    Args
    ----
    - data_dir: path directory to the dataset.
    - batch_size: how many samples per batch to load.
    - random_seed: fix seed for reproducibility.
    - valid_size: percentage split of the training set used for
      the validation set. Should be a float in the range [0, 1].
      In the paper, this number is set to 0.1.
    - shuffle: whether to shuffle the train/validation indices.
    - show_sample: plot 9x9 sample grid of the dataset.
    - num_workers: number of subprocesses to use when loading the dataset.
    - pin_memory: whether to copy tensors into CUDA pinned memory. Set it to
      True if using GPU.

    Returns
    -------
    - train_loader: training set iterator.
    - valid_loader: validation set iterator.
    """
    error_msg = "[!] valid_size should be in the range [0, 1]."
    assert ((valid_size >= 0) and (valid_size <= 1)), error_msg

    # define transforms
    normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                     std=[0.229, 0.224, 0.225])
    trans = transforms.Compose([
        transforms.ColorJitter(0.2, 0.2, 0.2, 0.2),
        transforms.Resize((256,256)),
        transforms.ToTensor(), normalize,
    ])

    # load dataset

    #dataset = datasets.MNIST(
    #   data_dir, train=True, download=True, transform=trans
    # )
    dataset = datasets.CIFAR10(
        data_dir, train=True, download=True, transform=trans
    )
    # dataset = CIFAR_RAM(
    #     data_dir, train=True, download=True, transform=trans, random_bg=False
    # )
    #dataset = PlacesGlimpseDataSet(data_dir, train_file, transform=trans)
    #dataset = HDF5Dataset(os.path.join(data_dir, "placesh5"), transform=trans)
    
    num_train = len(dataset)
    indices = list(range(num_train))
    split = int(np.floor(valid_size * num_train))

    if shuffle:
        np.random.seed(random_seed)
        np.random.shuffle(indices)

    train_idx, valid_idx = indices[split:], indices[:split]

    train_sampler = SubsetRandomSampler(train_idx)
    valid_sampler = SubsetRandomSampler(valid_idx)

    train_loader = torch.utils.data.DataLoader(
        dataset, batch_size=batch_size, sampler=train_sampler,
        num_workers=num_workers, pin_memory=pin_memory,
    )

    valid_loader = torch.utils.data.DataLoader(
        dataset, batch_size=batch_size, sampler=valid_sampler,
        num_workers=num_workers, pin_memory=pin_memory,
    )

    # visualize some images
    if show_sample:
        sample_loader = torch.utils.data.DataLoader(
            dataset, batch_size=9, shuffle=shuffle,
            num_workers=num_workers, pin_memory=pin_memory
        )
        data_iter = iter(sample_loader)
        images, labels, fix = data_iter.next()
        print(fix)
        X = images.numpy() 
        X = (X / 4)
        X = np.transpose(X, [0, 2, 3, 1])
        plot_images(X, labels)

    return (train_loader, valid_loader)


def get_test_loader(data_dir,
                    batch_size,
                    num_workers=4,
                    pin_memory=False,
                    test_file=None):
    """
    Utility function for loading and returning a multi-process
    test iterator over the specified dataset.

    If using CUDA, num_workers should be set to 1 and pin_memory to True.

    Args
    ----
    - data_dir: path directory to the dataset.
    - batch_size: how many samples per batch to load.
    - num_workers: number of subprocesses to use when loading the dataset.
    - pin_memory: whether to copy tensors into CUDA pinned memory. Set it to
      True if using GPU.

    Returns
    -------
    - data_loader: test set iterator.
    """
    # define transforms
    # define transforms
    normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                     std=[0.229, 0.224, 0.225])
    trans = transforms.Compose([
        transforms.ColorJitter(0.2, 0.2, 0.2, 0.2),
        transforms.Resize((256,256)),
        transforms.ToTensor(), normalize,
    ])

    # load dataset
    #dataset = datasets.MNIST(
    #     data_dir, train=False, download=True, transform=trans
    # )
    dataset = datasets.CIFAR10(
        data_dir, train=False, download=True, transform=trans
    )
    # dataset = CIFAR_RAM(
    #     data_dir, train=False, download=True, transform=trans, random_bg=False
    # )
    #dataset = PlacesGlimpseDataSet(data_dir, test_file, transform=trans)
    #dataset = HDF5Dataset(os.path.join(data_dir, "placesh5"), transform=trans)

    data_loader = torch.utils.data.DataLoader(
        dataset, batch_size=batch_size, shuffle=False,
        num_workers=num_workers, pin_memory=pin_memory,
    )

    return data_loader

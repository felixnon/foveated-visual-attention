"""
Definitions of custom datasets that can be used from the data loaders
"""

import numpy as np

import torch
from torchvision import datasets
from torchvision import transforms
from torch.utils.data.sampler import SubsetRandomSampler

import os
from PIL import Image
from torch.utils.data import Dataset
import random

import h5py

from pathlib import Path


class CIFAR_RAM(datasets.CIFAR10):
    """
    CIFAR 10 Dataset variation.
    Overrides original CIFAR10 dataset so that images are placed randomly inside of an larger image.
    """

    def __init__(self, root, train=True, transform=None, target_transform=None,
                 download=False, random_bg=False, size=64, patch_size=16):
        super().__init__(root, train, transform, target_transform, download)
        self.random_bg = random_bg
        self.size = size
        self.patch_size = patch_size

    def __getitem__(self, index):
        """
        Args:
            index (int): Index

        Returns:
            tuple: (image, target) where target is index of the target class.
        """
        img, target = self.data[index], self.targets[index]

        # doing this so that it is consistent with all other datasets
        # to return a PIL Image
        img = Image.fromarray(img)
        img = img.resize((self.patch_size, self.patch_size))
        if self.random_bg:
            # Create a noisy background consisting of the colors from the original image
            background = np.asarray(img.resize((self.size, self.size)))
            background = np.reshape(background, (-1,3))
            background = np.random.permutation(background)
            background = np.reshape(background, (self.size, self.size, 3))
            background = Image.fromarray(background)
        else:
            # create a completely black background
            background = Image.new('RGB', (self.size, self.size), (0, 0, 0))
        # Paste the CIFAR image into the background with a random offset
        offset = (np.random.randint(0, self.size-img.size[0]),
                  np.random.randint(0, self.size-img.size[1]))
        background.paste(img, offset)
        img = background

        # apply transformations
        if self.transform is not None:
            img = self.transform(img)

        if self.target_transform is not None:
            target = self.target_transform(target)

        return img, target

class CIFAR_RAM_TEST(datasets.CIFAR10):
    """
    Version of the CIFAR_RAM dataset for testing. In addition to the image and target
    it also returns the center coordinates of the embedded image. 
    """

    def __init__(self, root, train=True, transform=None, target_transform=None,
                 download=False, random_bg=False, size=64, patch_size=16):
        super().__init__(root, train, transform, target_transform, download)
        self.random_bg = random_bg
        self.size = size
        self.patch_size = patch_size

    def __getitem__(self, index):
        """
        Args:
            index (int): Index

        Returns:
            tuple: (image, target, offset) where target is index of the target class.
        """
        img, target = self.data[index], self.targets[index]

        # doing this so that it is consistent with all other datasets
        # to return a PIL Image
        img = Image.fromarray(img)
        img = img.resize((self.patch_size, self.patch_size))
        if self.random_bg:
            # Create a noisy background consisting of the colors from the original image
            background = np.asarray(img.resize((self.size, self.size)))
            background = np.reshape(background, (-1,3))
            background = np.random.permutation(background)
            background = np.reshape(background, (self.size, self.size, 3))
            background = Image.fromarray(background)
        else:
            # create a completely black background
            background = Image.new('RGB', (self.size, self.size), (0, 0, 0))
        # Paste the CIFAR image into the background with a random offset
        offset = (random.randint(0, self.size-img.size[0]),
                  random.randint(0, self.size-img.size[1]))
        background.paste(img, offset)
        img = background
        
        # apply transformations
        if self.transform is not None:
            img = self.transform(img)

        if self.target_transform is not None:
            target = self.target_transform(target)

        return img, target, (offset[0]/self.size + 0.125, offset[1]/self.size + 0.125)


class PlacesDataSet(Dataset):
    """
    Custom Places Dataset.
    Loads the places dataset from a directory with sub-directories for each category
    """
    def __init__(self, root, transform = None):
        """
        Args:
            root_dir (string): Directory with all the images.
        """

        self.transform = transform
        print("Initializing Places Dataset...")
        places_dir = os.path.join(root, "places")
        print("Data located in:", places_dir)
        class_directories = [dI for dI in os.listdir(places_dir) if os.path.isdir(os.path.join(places_dir,dI))]
        print(class_directories)

        self.data = []
        for i, directory in enumerate(class_directories):
            print("Loading", directory, "...")
            print(os.path.join(places_dir, directory))
            files = os.listdir(os.path.join(places_dir, directory))
            for img_file in files:
                self.data.append(
                    {"path": os.path.join(places_dir, directory, img_file), 
                     "class_id": i, 
                     "class_name": directory})

        random.shuffle(self.data)
    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        """
        Args:
            index (int): Index

        Returns:
            tuple: (image, target) where target is index of the target class.
        """
        img = Image.open(self.data[index]["path"]).convert("RGB")
        target = self.data[index]["class_id"]

        if self.transform is not None:
            img = self.transform(img)
        return img, target


class PlacesDataSet2(Dataset):
    """
    Custom Places Dataset.
    Loads the places dataset from a text file listing all training/validation images.
    Format of the file: train/<label>/<path_to_img>
    """
    def __init__(self, root, transform = None, train=True):
        """
        Args:
            root_dir (string): Directory with all the images.
        """

        self.transform = transform
        self.train = train
        print("Initializing Places Dataset...")
        
        places_dir = os.path.join(root, "places365")
        if train:
            path_file = os.path.join(places_dir, "train.txt")
        else:
            path_file = os.path.join(places_dir, "val.txt")

        with open(path_file) as f:
            if train:
                paths = f.read().splitlines()[:196157]
            else:
                paths = f.read().splitlines()[:3999]
        self.classes = list(dict.fromkeys([s.split("/")[1] for s in paths]))
        self.data = [{"path": os.path.join(places_dir,s), "class_name": s.split("/")[1], "class_id": self.classes.index(s.split("/")[1])} for s in paths]

        print("Loaded {} classes and {} items.".format(len(self.classes), len(self.data)))
        print([(i, c) for i,c in enumerate(self.classes)])
        
        random.shuffle(self.data)

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        """
        Args:
            index (int): Index

        Returns:
            tuple: (image, target) where target is index of the target class.
        """
        img = Image.open(self.data[index]["path"]).convert("RGB")
        target = self.data[index]["class_id"]

        if self.transform is not None:
            img = self.transform(img)
        return img, target

    def get_class_name(self, c):
        return self.classes[c]


class PlacesGlimpseDataSet(Dataset):
    """
    Custom Places Dataset augmented with glimpse positions.
    Loads the places dataset from a text file listing all training/validation images along with the proposed fixation points.
    Format of the file: train/<label>/<path_to_img>;[(x1,y1),(x2,y2),...,(xn,yn)]
    The file is generated by the Glimpse generation algorithm.
    """
    def __init__(self, train_file, val_file=None, transform = None, train=True):
        """
        Args:
            train_file (string): path to text file with all the images.
        """

        self.transform = transform
        self.train = train
        print("Initializing Places Dataset...")
        
        places_dir = os.path.dirname(train_file)
        if train:
            path_file = train_file
        else:
            path_file = val_file

        with open(path_file) as f:
            if train:
                lines = f.read().splitlines()
            else:
                lines = f.read().splitlines()
        paths = [line.split(";", maxsplit=1)[0] for line in lines]
        fixations = [np.array(eval(line.split(";", maxsplit=1)[1]))*2 -1 for line in lines]
        self.classes = list(dict.fromkeys([s.split("/")[1] for s in paths]))
        self.data = [{"path": os.path.join(places_dir,s), "class_name": s.split("/")[1], "class_id": self.classes.index(s.split("/")[1]), "fixations": fixations[paths.index(s)]} for s in paths]

        print("Loaded {} classes and {} items.".format(len(self.classes), len(self.data)))
        print([(i, c) for i,c in enumerate(self.classes)])
        
        random.shuffle(self.data)


    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        """
        Args:
            index (int): Index

        Returns:
            tuple: (image, target) where target is index of the target class.
        """
        img = Image.open(self.data[index]["path"]).convert("RGB")
        target = self.data[index]["class_id"]
        fixations = self.data[index]["fixations"]
        # apply randomly a horizontal flip to the image. also flip the fixation points then
        b = random.randint(0,1)
        if b:
            flip = transforms.RandomHorizontalFlip(p=1)
            img = flip(img)
            fixations[:,0] = fixations[:,0] * (-1)

        # apply random center crop to the image. adjust fixation points. resize back to orig size
        size = img.size[0]
        size_factor = random.uniform(0.8, 1.2)
        crop = transforms.CenterCrop(int(size*size_factor))
        img = crop(img)
        img = transforms.Resize(size)(img)
        fixations = fixations / size_factor

        # slightly move fixation points for more augmentation
        offsets = np.random.uniform(-0.04, 0.04, fixations.shape)
        fixations = fixations + offsets
        
        # clamp fixations so they are in range of the image
        fixations = np.clip(a=fixations, a_min=-1, a_max=1)

        # shuffle the order of the fixations
        np.random.shuffle(fixations)

        if self.transform is not None:
            img = self.transform(img)
        return img, target, fixations

    def get_class_name(self, c):
        return self.classes[c]


class HDF5Dataset(Dataset):
    """Represents an abstract HDF5 dataset.
    
    Input params:
        file_path: Path to the folder containing the dataset (one or multiple HDF5 files).
        recursive: If True, searches for h5 files in subdirectories.
        load_data: If True, loads all the data immediately into RAM. Use this if
            the dataset is fits into memory. Otherwise, leave this at false and 
            the data will load lazily.
        data_cache_size: Number of HDF5 files that can be cached in the cache (default=3).
        transform: PyTorch transform to apply to every data instance (default=None).
    
    Source: https://towardsdatascience.com/hdf5-datasets-for-pytorch-631ff1d750f5
    """
    def __init__(self, file_path, recursive=False, load_data=False, data_cache_size=3, transform=None):
        super().__init__()
        self.data_info = []
        self.data_cache = {}
        self.data_cache_size = data_cache_size
        self.transform = transform

        # Search for all h5 files
        p = Path(file_path)
        assert(p.is_dir())
        if recursive:
            files = sorted(p.glob('**/*.h5'))
        else:
            files = sorted(p.glob('*.h5'))
        if len(files) < 1:
            raise RuntimeError('No hdf5 datasets found')

        for h5dataset_fp in files:
            self._add_data_infos(str(h5dataset_fp.resolve()), load_data)
            
    def __getitem__(self, index):
        # get data
        x = self.get_data("data", index)
        if self.transform:
            x = self.transform(Image.fromarray(x))
        else:
            x = torch.from_numpy(x)

        # get label
        y = self.get_data("label", index)
        y = torch.from_numpy(y)
        return (x, int(y[0]))

    def __len__(self):
        return len(self.get_data_infos('data'))
    
    def _add_data_infos(self, file_path, load_data):
        with h5py.File(file_path, "r") as h5_file:
            # Walk through all groups, extracting datasets
            for gname, group in h5_file.items():
                for dname, ds in group.items():
                    # if data is not loaded its cache index is -1
                    idx = -1
                    if load_data:
                        # add data to the data cache
                        idx = self._add_to_cache(ds[()], file_path)
                    
                    # type is derived from the name of the dataset; we expect the dataset
                    # name to have a name such as 'data' or 'label' to identify its type
                    # we also store the shape of the data in case we need it
                    self.data_info.append({'file_path': file_path, 'type': dname, 'shape': ds[()].shape, 'cache_idx': idx})

    def _load_data(self, file_path):
        """Load data to the cache given the file
        path and update the cache index in the
        data_info structure.
        """
        with h5py.File(file_path, "r") as h5_file:
            for gname, group in h5_file.items():
                for dname, ds in group.items():
                    # add data to the data cache and retrieve
                    # the cache index
                    idx = self._add_to_cache(ds[()], file_path)

                    # find the beginning index of the hdf5 file we are looking for
                    file_idx = next(i for i,v in enumerate(self.data_info) if v['file_path'] == file_path)

                    # the data info should have the same index since we loaded it in the same way
                    self.data_info[file_idx + idx]['cache_idx'] = idx

        # remove an element from data cache if size was exceeded
        if len(self.data_cache) > self.data_cache_size:
            # remove one item from the cache at random
            removal_keys = list(self.data_cache)
            removal_keys.remove(file_path)
            self.data_cache.pop(removal_keys[0])
            # remove invalid cache_idx
            self.data_info = [{'file_path': di['file_path'], 'type': di['type'], 'shape': di['shape'], 'cache_idx': -1} if di['file_path'] == removal_keys[0] else di for di in self.data_info]

    def _add_to_cache(self, data, file_path):
        """Adds data to the cache and returns its index. There is one cache
        list for every file_path, containing all datasets in that file.
        """
        if file_path not in self.data_cache:
            self.data_cache[file_path] = [data]
        else:
            self.data_cache[file_path].append(data)
        return len(self.data_cache[file_path]) - 1

    def get_data_infos(self, type):
        """Get data infos belonging to a certain type of data.
        """
        data_info_type = [di for di in self.data_info if di['type'] == type]
        return data_info_type

    def get_data(self, type, i):
        """Call this function anytime you want to access a chunk of data from the
            dataset. This will make sure that the data is loaded in case it is
            not part of the data cache.
        """
        fp = self.get_data_infos(type)[i]['file_path']
        if fp not in self.data_cache:
            self._load_data(fp)
        
        # get new cache_idx assigned by _load_data_info
        cache_idx = self.get_data_infos(type)[i]['cache_idx']
        return self.data_cache[fp][cache_idx]

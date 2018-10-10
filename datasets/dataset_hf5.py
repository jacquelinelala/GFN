import torch.utils.data as data
import torch
from skimage.io import imread, imsave
import numpy as np
import random
from os.path import join
import glob
import h5py
import sys
import os
from os.path import join

#=================== Utils ===================#
def is_image_file(filename):
    if filename.startswith('._'):
        return
    return any(filename.endswith(extension) for extension in [".png", ".jpg", ".jpeg"])

#=================== Testing ===================#

class DataValSet(data.Dataset):
    def __init__(self, root_dir):
        #one input & ground truth
        self.input_dir = join(root_dir, 'LR_Blur')
        self.sr_dir = join(root_dir, 'HR')

        #Online Loading
        self.input_names = [x for x in sorted(os.listdir(self.input_dir)) if is_image_file(x)]
        self.hr_names = [x for x in sorted(os.listdir(self.sr_dir)) if is_image_file(x)]

    def __len__(self):
        return len(self.hr_names)

    def __getitem__(self, index):
        input     = np.asarray(imread(join(self.input_dir, self.input_names[index])).transpose((2, 0, 1)), np.float32).copy() / 255
        target    = np.asarray(imread(join(self.sr_dir, self.hr_names[index])).transpose((2, 0, 1)), np.float32).copy() / 255
        return input, target

#=================== Training ===================#

class DataSet(data.Dataset):
    def __init__(self, h5py_file_path):
        super(DataSet, self).__init__()
        self.hdf5_file  = h5py_file_path

        self.file    = h5py.File(self.hdf5_file, 'r')
        #self.file.keys()
        self.inputs  = self.file.get("data")
        self.deblurs = self.file.get("label_db")
        self.hrs     = self.file.get("label")

    def __len__(self):
        return self.inputs.shape[0]

    def __getitem__(self, index):
        #print(index)
        # numpy
        input_patch  = np.asarray(self.inputs[index, :, :, :], np.float32)
        deblur_patch = np.asarray(self.deblurs[index, :, :, :], np.float32)
        hr_patch     = np.asarray(self.hrs[index, :, :, :], np.float32)
        # randomly flip
        if random.randint(0, 1) == 0:
            input_patch  = np.flip(input_patch, 2)
            deblur_patch = np.flip(deblur_patch, 2)
            hr_patch     = np.flip(hr_patch, 2)
        # randomly rotation
        rotation_times = random.randint(0, 3)
        input_patch    = np.rot90(input_patch, rotation_times, (1, 2))
        deblur_patch   = np.rot90(deblur_patch, rotation_times, (1, 2))
        hr_patch       = np.rot90(hr_patch, rotation_times, (1, 2))

        return input_patch.copy(),\
               deblur_patch.copy(),\
               hr_patch.copy()




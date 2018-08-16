import torch.utils.data as data
import torch
from PIL import Image
from skimage.io import imread, imsave
import numpy as np
import random
from os.path import join
import glob
import h5py
import sys
import os
sys.path.append('..')
from os.path import join
from skimage.transform import rotate
from skimage import img_as_float

#=================== Utils ===================#
def is_image_file(filename):
    if filename.startswith('._'):
        return
    return any(filename.endswith(extension) for extension in [".png", ".jpg", ".jpeg"])

#=================== Testing ===================#

class DataValSet(data.Dataset):
    def __init__(self, root_dir):
        # input_dir = join(root_val_dir, 'LR_Blur')  #---------------Input path
        #one input & two ground truths
        self.input_dir = join(root_dir, 'LR_Blur')
        self.lr_deblur_dir = join(root_dir, 'LR')
        self.sr_dir = join(root_dir, 'HR')

        #Online Loading
        self.input_names = [x for x in sorted(os.listdir(self.input_dir)) if is_image_file(x)]
        self.lr_deblur_names = [x for x in sorted(os.listdir(self.lr_deblur_dir)) if is_image_file(x)]
        self.hr_names = [x for x in sorted(os.listdir(self.sr_dir)) if is_image_file(x)]

    def __len__(self):
        return len(self.hr_names)

    def __getitem__(self, index):
        input = np.asarray(imread(join(self.input_dir, self.input_names[index])).transpose((2, 0, 1)), np.float32).copy() / 255
        target_lr = np.asarray(imread(join(self.lr_deblur_dir, self.lr_deblur_names[index])).transpose((2, 0, 1)), np.float32).copy() /255
        target = np.asarray(imread(join(self.sr_dir, self.hr_names[index])).transpose((2, 0, 1)), np.float32).copy() / 255

        return input, target_lr, target, self.hr_names[index]

#=================== Training ===================#

class DataSet(data.Dataset):
    def __init__(self, h5py_file_path):
        super(DataSet, self).__init__()
        self.hdf5_file  = h5py_file_path

        self.file   = h5py.File(self.hdf5_file, 'r')
        self.file.keys()
        self.inputs = np.array(self.file.get("data"))
        self.deblurs = np.array(self.file.get("label_db"))
        self.hrs = np.array(self.file.get("label"))

    def __len__(self):
        return self.inputs.shape[0]

    def __getitem__(self, index):
        input_patch  = self.inputs[index, :, :, :]
        deblur_patch = self.deblurs[index, :, :, :]
        hr_patch     = self.hrs[index, :, :, :]
        # randomly flip
        if random.randint(0, 1) == 0:
            input_patch  = np.flip(input_patch, 1)
            deblur_patch = np.flip(deblur_patch, 1)
            hr_patch     = np.flip(hr_patch, 1)
        else:
            input_patch  = np.flip(input_patch, 2)
            deblur_patch = np.flip(deblur_patch, 2)
            hr_patch     = np.flip(hr_patch, 2)
        # randomly rotation
        rotation_times = random.randint(0, 3)
        input_patch  = np.rot90(input_patch, rotation_times, (1, 2))
        deblur_patch = np.rot90(deblur_patch, rotation_times, (1, 2))
        hr_patch     = np.rot90(hr_patch, rotation_times, (1, 2))

        return torch.from_numpy(np.asarray(input_patch, np.float32).copy()),\
               torch.from_numpy(np.asarray(deblur_patch, np.float32).copy()),\
               torch.from_numpy(np.asarray(hr_patch, np.float32).copy())




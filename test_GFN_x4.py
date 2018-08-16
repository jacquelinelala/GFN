from __future__ import print_function

import argparse
import os
import time
from math import log10
from os.path import join
import numpy as np
from torchvision import transforms
import torch
from torch.utils.data import DataLoader
from GFN.datasets.dataset_hf5 import DataValSet
import statistics

# Training settings
model_sets = {'GFN_4x':'models/GFN_4x.pth'}
modelName = 'GFN_4x'

# Training settings
parser = argparse.ArgumentParser(description="PyTorch LapSRN Test")
parser.add_argument("--cuda", action="store_true", help="use cuda?")
parser.add_argument("--model", default=model_sets[modelName], type=str, help="model path")
parser.add_argument("--scale", default=4, type=int, help="scale factor, Default: 4")
os.environ['CUDA_VISIBLE_DEVICES'] = '0'
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def test(test_gen, model, criterion, SR_dir):
    avg_psnr = 0
    med_time = []

    with torch.no_grad():
        for iteration, batch in enumerate(test_gen, 1):
            LR_Blur = batch[0]
            LR_Deblur = batch[1]
            HR = batch[2]
            names = batch[3]

            LR_Blur = LR_Blur.to(device)
            LR_Deblur = LR_Deblur.to(device)
            HR = HR.to(device)

            start_time = time.perf_counter()#-------------------------begin to deal with an image's time
            [lr_deblur, scoremap, sr] = model(LR_Blur)
            #modify
            im_l = lr_deblur.cpu().numpy()
            im_l[im_l < 0] = 0
            im_l[im_l > 1] = 1
            im_h = sr.cpu().numpy()
            im_h[im_h < 0] = 0
            im_h[im_h > 1] = 1
            torch.cuda.synchronize()#wait for CPU & GPU time syn
            evalation_time = time.perf_counter() - start_time#---------finish an image
            med_time.append(evalation_time)


            lr_tensor = torch.from_numpy(np.asarray(im_l, np.float32))
            sr_tensor = torch.from_numpy(np.asarray(im_h, np.float32))

            resultSRDeblur = transforms.ToPILImage()(sr_tensor[0])
            resultSRDeblur.save(join(SR_dir, '{0:04d}_GFN_4x.png'.format(iteration)))
            LR = lr_tensor.to(device)
            SR = sr_tensor.to(device)

            mse = criterion(SR, HR)
            psnr = 10 * log10(1 / mse)
            avg_psnr += psnr
            print(iteration)

        print("===>Avg. SR PSNR:{:4f} dB".format(avg_psnr / iteration))
        median_time = statistics.median(med_time)
        print(median_time)

opt = parser.parse_args()
print(opt)

print("===> Loading datasets")
root_val_dir = '/4TB/datasets/LR-GOPRO/Validation_4x'   #----------Validation path
SR_dir = join(root_val_dir, 'Results')  #--------------------------SR results save path
print(SR_dir)
testloader = DataLoader(DataValSet(root_val_dir), batch_size=1, shuffle=False, pin_memory=False)
print("===> Loading model and criterion")
print(opt.model)
model = torch.load(opt.model, map_location=lambda storage, loc: storage)["model"]
model = model.to(device)
criterion = torch.nn.MSELoss(size_average=True)
criterion = criterion.to(device)

test(testloader, model, criterion, SR_dir)



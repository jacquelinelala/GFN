# GFN

**"Gated Fusion Network for Joint Image Deblurring and Super-Resolution"** by [Xinyi Zhang](http://xinyizhang.tech), Hang Dong, [Zhe Hu](http://eng.ucmerced.edu/people/zhu), [Wei-Sheng Lai](http://graduatestudents.ucmerced.edu/wlai24/), Fei Wang, [Ming-Hsuan Yang](http://faculty.ucmerced.edu/mhyang/)**(oral presentation on BMVC2018)**.

[[arXiv](https://arxiv.org/abs/1807.10806)][[Slide](http://xinyizhang.tech/files/BMVC_slides.ppt)]

There are more details you can find on [Project Website : http://xinyizhang.tech/bmvc2018](http://xinyizhang.tech/bmvc2018/).

![Archi](http://xinyizhang.tech/content/images/2018/09/gated-fusion-network.png)
![heatmap](http://xinyizhang.tech/content/images/2018/07/2-1.png)
## Inproved the training process
In order to obtain a more stable training process, now we adopt a three-step training strategy, which differs from our paper and improves PSNR from 27.74dB to 27.81dB on LR-GOPRO 4x dataset.

| Model | LR-GOPRO 4x PSNR(dB) | Time(s) |
|  :-----  |  :-----:  | :-----:  |
|  [SCGAN](https://sites.google.com/view/xiangyuxu/deblursr_iccv17)  |  22.74  | 0.66  |
|  [SRResNet](https://arxiv.org/abs/1609.04802)  |  24.40  | 0.07  |
|  ED-DSRN  |  26.44  | 0.10  |
|  [DeepDeblur](https://github.com/SeungjunNah/DeepDeblur_release) + [EDSR](https://github.com/thstkdgus35/EDSR-PyTorch)  |  25.09  | 2.70  |
|  [EDSR](https://github.com/thstkdgus35/EDSR-PyTorch) + [DeepDeblur](https://github.com/SeungjunNah/DeepDeblur_release)  |  26.35  | 8.10  |
|  GFN(BMVC paper)  |  27.74  | 0.07  |
|  GFN(Now)  |  27.81  | 0.07  |

## Dependencies
* Python 3.6
* PyTorch >= 0.4.0
* torchvision
* numpy
* skimage
* h5py
* MATLAB

## How to test:
### Test on LR-GOPRO Validation
#### Test on the latest trained model
This model is the result of the third step with 55 epoch.
1. Git clone this repository.
```bash
$git clone https://github.com/jacquelinelala/GFN.git
$cd GFN
```
2. Download the original GOPRO_Large dataset from [Google Drive](https://drive.google.com/file/d/1H0PIXvJH4c40pk7ou6nAwoxuR4Qh_Sa2/view?usp=sharing).
3. Generate the validation images of LR-GOPRO dataset: Run matlab function ``GFN/h5_generator/gopro_val_generator.m``. The generated test images will be stored in your_downloads_directory/GOPRO_Large/Validation_4x.

(If you don't have access to MATLAB, we offer a validation dataset for testing. You can download it from [GoogleDrive](https://drive.google.com/file/d/1H0PIXvJH4c40pk7ou6nAwoxuR4Qh_Sa2/view?usp=sharing) or [Pan Baidu](https://pan.baidu.com/s/1vsVTLoBA8pmOz_omNLUQTw).)
```bash
>> folder = 'your_downloads_directory/GOPRO_Large'; # You should replace the your_downloads_directory by your GOPRO_Large's directory.
>> gopro_val_generator(folder)
```
4. Download the trained model ``GFN_epoch_55.pkl`` from [here](http://xinyizhang.tech/files/GFN_epoch_55.pkl.zip), then unzip and move the ``GFN_epoch_55.pkl`` to ``GFN/models`` folder.

5. Run the ``GFN/test_GFN_x4.py`` with cuda on command line: 
```bash
GFN/$python test_GFN_x4.py --dataset your_downloads_directory/GOPRO_Large/Validation_4x
```
Then the deblurring and super-solving images ending with GFN_4x.png are in the directory of GOPRO_Large/Validation/Results.

6. Calculate the PSNR using Matlab function ``GFN/evaluation/test_RGB.m``. The output of the average PSNR is 27.810232 dB. You can also use the ``GFN/evaluation/test_bicubic.m`` to calculate the bicubic method.  
```bash
>> folder = 'your_downloads_directory/GOPRO_Large';
>> test_RGB(folder)
```

## How to train
### Train on LR-GOPRO dataset
You should accomplish the first two steps in **Test on LR-GOPRO Validation** before the following steps.
#### Train from scratch
1. Generate the train hdf5 files of LR-GOPRO dataset: Run the matlab function ``gopro_hdf5_generator.m`` which is in the directory of GFN/h5_generator. The generated hdf5 files are stored in the your_downloads_directory/GOPRO_Large/GOPRO_train256_4x_HDF5.
```bash
>> folder = 'your_downloads_directory/GOPRO_Large';
>> gopro_hdf5_generator(folder)
```
2. Run the ``GFN/train_GFN_4x.py`` with cuda on command line:
```bash
GFN/$python train_GFN_4x.py --dataset your_downloads_directory/GOPRO_Large/GOPRO_train256_4x_HDF5
```
3. The three step intermediate models will be respectively saved in models/1/ models/2 and models/3. You can also use the following command to test the intermediate results during the training process.
Run the ``GFN/test_GFN_x4.py`` with cuda on command line: 
```bash
GFN/$python test_GFN_x4.py --dataset your_downloads_directory/GOPRO_Large/Validation_4x --intermediate_process models/1/GFN_epoch_30.pkl # We give an example of step1 epoch30. You can replace another pkl file in models/.
```
#### Resume training from breakpoints
Since the training process will take 3 or 4 days, you can use the following command to resume the training process from any breakpoints.
Run the ``GFN/train_GFN_4x.py`` with cuda on command line:
```bash
GFN/$python train_GFN_4x.py --dataset your_downloads_directory/GOPRO_Large/GOPRO_train256_4x_HDF5 --resume models/1/GFN_epoch_30.pkl # Just an example of step1 epoch30.
```
## Citation

If you use these models in your research, please cite:

	@conference{Zhang2018,
		author = {Xinyi Zhang and Hang Dong and Zhe Hu and Wei-Sheng Lai and Fei Wang and Ming-Hsuan Yang},
		title = {Gated Fusion Network for Joint Image Deblurring and Super-Resolution},
		booktitle = {BMVC},
		year = {2018}
	}


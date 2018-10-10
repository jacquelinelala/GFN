# GFN

**"Gated Fusion Network for Joint Image Deblurring and Super-Resolution"** by [Xinyi Zhang](http://xinyizhang.tech), Hang Dong, [Zhe Hu](http://eng.ucmerced.edu/people/zhu), [Wei-Sheng Lai](http://graduatestudents.ucmerced.edu/wlai24/), Fei Wang, [Ming-Hsuan Yang](http://faculty.ucmerced.edu/mhyang/)**(oral presentation on BMVC2018)**.

[[arXiv](https://arxiv.org/abs/1807.10806)][[Slide](http://xinyizhang.tech/files/BMVC_slides.ppt)]

There are more details you can find on [Project Website : http://xinyizhang.tech/bmvc2018](http://xinyizhang.tech/bmvc2018/).

![Archi](http://xinyizhang.tech/content/images/2018/09/gated-fusion-network.png)
![heatmap](http://xinyizhang.tech/content/images/2018/07/2-1.png)

In order to obtain a more stable training process, now we adopt a three-step training strategy, which differs from our paper and gains a PSNR growth from previous 27.74dB to 27.81dB.

| Model | GOPRO 4x PSNR(dB) | Time(s) |
|  :-----  |  :-----:  | :-----:  |
|  SCGAN  |  22.74  | 0.66  |
|  SRResNet[18]  |  24.40  | 0.07  |
|  ED-DSRN[20]  |  26.44  | 0.10  |
|  DB[21] + SR[20]  |  25.09  | 2.70  |
|  SR[20] + DB[21]  |  26.35  | 8.10  |
|  GFN(Ours)  |  27.81  | 0.65  |

## Dependencies
* Python 3.6
* PyTorch >= 0.4.0
* torchvision
* numpy
* skimage
* h5py
* matlab R2017a

## How to test:
### Test on GOPRO Validation
#### Test on given trained model(This model is the result of the third step with 55 epoch.)
1. Git clone this repository.
```bash
$git clone https://github.com/jacquelinelala/GFN.git
$cd GFN
```
2. Download GOPRO_Large dataset from [Google Drive](https://drive.google.com/file/d/1H0PIXvJH4c40pk7ou6nAwoxuR4Qh_Sa2/view?usp=sharing).
3. Generate the validation images: Run matlab function ``gopro_val_generator.m`` which is in the directory of GFN/h5_generator. The generated test images will be stored in your_downloads_directory/GOPRO_Large/Validation_4x.
```bash
>> folder = 'your_downloads_directory/GOPRO_Large'; # You should replace the your_downloads_directory by your GOPRO_Large's directory.
>> gopro_val_generator(folder)
```
4. Download the trained model ``GFN_epoch_55.pkl`` from [here](http://xinyizhang.tech/files/GFN_epoch_55.pkl.zip), then unzip and move the ``GFN_epoch_55.pkl`` to ``GFN/models`` folder.

5. Run the ``GFN/test_GFN_x4.py`` with cuda on command line: 
```bash
GFN/$python test_GFN_x4.py --dataset your_downloads_directory/GOPRO_Large/Validation_4x
```
Then the deblurring and super-resolution images ending with GFN_4x.png are in the directory of GOPRO_Large/Validation/Results.

6. Calculate the PSNR using Matlab function ``test_RGB.m`` on directory of GFN/evaluation/. The output of the average PSNR is 27.810232db. You can also use the ``test_bicubic.m`` to calculate the bicubic method.  
```bash
>> folder = 'your_downloads_directory/GOPRO_Large';
>> test_RGB(folder)
```
#### Test on intermediate traing process
You should complete the first 3 steps in **Test on given trained model**. Meanwhile, you should have done some training in **Train on GOPRO dataset** and get some parts of traring pkl files. This is no need to accomplish the whole 3-step training, because it focuses on testing the intermediate training result.

Run the ``GFN/test_GFN_x4.py`` with cuda on command line: 
```bash
GFN/$python test_GFN_x4.py --dataset your_downloads_directory/GOPRO_Large/Validation_4x --intermediate_process models/1/GFN_epoch_30.pkl # We give an example of step1 epoch30. You can replace the pkl file on your models/'s.
```
## How to train
### Train on GOPRO dataset
You should accomplish the first two steps in **Test on GOPRO Validation** before the following steps.
#### Train from scratch
1. Generate the train hdf5 files: Run matlab function ``gopro_hdf5_generator.m`` which is in the directory of GFN/h5_generator. The generated hdf5 files are stored in the your_downloads_directory/GOPRO_Large/GOPRO_train256_4x_HDF5.
```bash
>> folder = 'your_downloads_directory/GOPRO_Large';
>> gopro_hdf5_generator(folder)
```
2. Run the ``GFN/train_GFN_4x.py`` with cuda on command line:
```bash
GFN/$python train_GFN_4x.py --dataset your_downloads_directory/GOPRO_Large/GOPRO_train256_4x_HDF5
```
#### Train from resume
You should have done some training in **Train from scratch** and get some parts of pkl files. We use the trained model resume training.
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


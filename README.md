# GFN

**"Gated Fusion Network for Joint Image Deblurring and Super-Resolution"** by [Xinyi Zhang](http://xinyizhang.tech), Hang Dong, [Zhe Hu](http://eng.ucmerced.edu/people/zhu), [Wei-Sheng Lai](http://graduatestudents.ucmerced.edu/wlai24/), Fei Wang, [Ming-Hsuan Yang](http://faculty.ucmerced.edu/mhyang/)**(oral presentation on BMVC2018)**.

[[arXiv](https://arxiv.org/abs/1807.10806)][[Slide](http://xinyizhang.tech/files/BMVC_slides.ppt)]

There are more details you can find on [Project Website : http://xinyizhang.tech/bmvc2018](http://xinyizhang.tech/bmvc2018/).

![Archi](http://xinyizhang.tech/content/images/2018/09/gated-fusion-network.png)
![heatmap](http://xinyizhang.tech/content/images/2018/07/2-1.png)

## Dependencies
* Python 3.6
* PyTorch >= 0.4.0
* torchvision
* numpy
* skimage
* h5py
* matlab R2017a

## How to test:
### Test on LR-GOPRO Validation
1. Git clone this repository.
```bash
$git clone https://github.com/jacquelinelala/GFN.git
$cd GFN
```
2. Download the trained model GFN_4x.pth from [http://xinyizhang.tech/files/](http://xinyizhang.tech/files/), then unzip and move the GFN_4x.pth to "models" directory.
3. Download the test LR-GOPRO dataset from [Google Drive](https://drive.google.com/file/d/11TD3gVRtjlOobT8k9x2oXjEOx-dLtoDt/view?usp=sharing)(Recommended) or [Pan Baidu](https://pan.baidu.com/s/17oo5rDk4v2RUD3wzte_1aw?errno=0&errmsg=Auth%20Login%20Sucess&&bduss=&ssnerror=0&traceid=)(Optional).

3. Run the test_GFN_x4.py with cuda on command line: 
```bash
GFN/$python test_GFN_x4.py --dataset your_downloads_directory/LR-GOPRO/Validation_4x
```
Then the deblurring and super-resolution images ending with GFN_4x.png are in the directory of Validation/Results.
4. Calculate the PSNR & SSIM using Matlab on directory of evaluation/.

### Test on your own dataset
## How to train
### Train on LR-GOPRO dataset
Run the train_GFN_4x.py with cuda on command line:
```bash
GFN/$python train_GFN_4x.py --dataset your_downloads_directory/LR-GOPRO/GOPRO_train256_4x_HDF5
```
## Citation

If you use these models in your research, please cite:

	@conference{Zhang2018,
		author = {Xinyi Zhang and Hang Dong and Zhe Hu and Wei-Sheng Lai and Fei Wang and Ming-Hsuan Yang},
		title = {Gated Fusion Network for Joint Image Deblurring and Super-Resolution},
		booktitle = {BMVC},
		year = {2018}
	}


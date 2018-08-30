# GFN

"Gated Fusion Network for Joint Image Deblurring and Super-Resolution"([https://arxiv.org/abs/1807.10806](https://arxiv.org/abs/1807.10806))
by [Xinyi Zhang](http://xinyizhang.tech), Hang Dong, [Zhe Hu](http://eng.ucmerced.edu/people/zhu), [Wei-Sheng Lai](http://graduatestudents.ucmerced.edu/wlai24/), Fei Wang, [Ming-Hsuan Yang](http://faculty.ucmerced.edu/mhyang/) on BMVC2018.

There are more details you can find on [Project Website : http://xinyizhang.tech/bmvc2018](http://xinyizhang.tech/bmvc2018/).

![Archi](http://xinyizhang.tech/content/images/2018/07/A1.png)

## Version1:
1. Git clone this repository.
```bash
$git clone https://github.com/jacquelinelala/GFN.git
```
2. Download the trained model GFN_4x.pth from [http://xinyizhang.tech/files/](http://xinyizhang.tech/files/), then unzip and move the GFN_4x.pth to "models" directory.
3. Download the test LR-GOPRO dataset from [Google Drive](https://drive.google.com/file/d/11TD3gVRtjlOobT8k9x2oXjEOx-dLtoDt/view?usp=sharing)(Recommended) or [Baidu Pan](https://pan.baidu.com/s/17oo5rDk4v2RUD3wzte_1aw?errno=0&errmsg=Auth%20Login%20Sucess&&bduss=&ssnerror=0&traceid=)(Optional).
3. You need to modify some datasets directories in the codes depend on your downloading directory.
4. Run the test_GFN_x4.py with cuda, then the deblurring and super-resolution images ending with GFN_4x.png are in the directory of Validation/Results.
5. Calculate the PSNR & SSIM using Matlab on directory of evaluation/.

### Citation

If you use these models in your research, please cite:

	@conference{Zhang2018,
		author = {Xinyi Zhang and Hang Dong and Zhe Hu and Wei-Sheng Lai and Fei Wang and Ming-Hsuan Yang},
		title = {Gated Fusion Network for Joint Image Deblurring and Super-Resolution},
                booktitle = {BMVC},
		year = {2018}
	}


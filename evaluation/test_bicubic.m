% -------------------------------------------------------------------------
%   Description:
%       Use bicubic to upscale the input, and evaluate the results in terms
%       of PSNR.
%
%   Input:
%       - Validation/LR_Blur : Blurry low-resolution input
%       - Validation/HR : HR
%
%   Output:
%       - Validation/Results: images using bicubic upscaling and formatting
%       with 'Bic****.png' in Results directory
%       - PSNR: Average PSNR
%       
%   Citation: 
%       Gated Fusion Network for Joint Image Deblurring and Super-Resolution
%       The British Machine Vision Conference(BMVC2018 oral)
%       Xinyi Zhang, Hang Dong, Zhe Hu, Wei-Sheng Lai, Fei Wang and Ming-Hsuan Yang
%   Thanks:
%       Many thanks to Wei-Sheng Lai al. for LapSRN. In this project we use
%       some codes from their project.
%   Contact:
%       cvxinyizhang@gmail.com
%   Project Website:
%       http://xinyizhang.tech/bmvc2018
%       https://github.com/jacquelinelala/GFN

%% testing
function test_bicubic(folder)
addpath(genpath('utils'));
lr_blur_path = fullfile(folder, 'Validation_4x/LR_Blur');
hr_path = fullfile(folder, 'Validation_4x/HR');
results_path = fullfile(folder, 'Validation_4x/Results');
lr_blur_dir = dir(fullfile(lr_blur_path, '*.png'));
hr_dir = dir(fullfile(hr_path, '*.png'));
count = length(lr_blur_dir);
fprintf("%d\n",count);
PSNR = zeros(count, 1);
SSIM = zeros(count, 1);
IFC = zeros(count, 1);
ifc = 0;
scale = 4;

for i = 1 : count
	HR = im2double(imread(fullfile(hr_dir(i).folder, hr_dir(i).name)));
	LR_Blur = im2double(imread(fullfile(lr_blur_dir(i).folder, lr_blur_dir(i).name)));
	LR_Blur_Bic = imresize(LR_Blur, 4, 'bicubic');
	imwrite(LR_Blur_Bic, fullfile(results_path, strcat(hr_dir(i).name(1:4), 'Bic.png')),'png');
	[PSNR(i), IFC(i)] = evaluate_SR(HR, LR_Blur_Bic, scale, ifc);
end

PSNR(count + 1) = mean(PSNR(:));

fprintf("Average PSNR is %f\n", PSNR(count + 1));

PSNR_path = fullfile(folder, 'Validation_4x', 'PSNR-LRBlurBic_HR.txt');
save_matrix(PSNR, PSNR_path);


%   Description:
%   -testting RGB channels PSNR & SSIM
%
%   Input:    
%   - Validation/Results: some specific group images with limited endings.
%   - Validation/HR
%
%   Output:
%       - PSNR  Average PSNR
%       - SSIM  Average SSIM
%   Citation: 
%       Gated Fusion Network for Joint Image Deblurring and Super-Resolution
%       The British Machine Vision Conference(BMVC2018 oral)
%       Xinyi Zhang, Hang Dong, Zhe Hu, Wei-Sheng Lai,?Fei Wang and?Ming-Hsuan Yang
%   Thanks:
%       Many thanks to Wei-Sheng Lai al. for LapSRN. In this project we use
%       some codes from their project.
%   Contact:
%       cvxinyizhang@gmail.com
%   Project Website:
%       http://xinyizhang.tech/bmvc2018
%       https://github.com/jacquelinelala/GFN

%% testing
addpath(genpath('utils'));
results_path = '/4TB/datasets/LR-GOPRO/Validation_4x/Results';
hr_path = '/4TB/datasets/LR-GOPRO/Validation_4x/HR';

result_dir = dir(fullfile(results_path, '*GFN_4x.png'));
hr_dir = dir(fullfile(hr_path, '*.png'));
count = length(hr_dir);
PSNR = zeros(count, 1);
SSIM = zeros(count, 1);
IFC = zeros(count, 1);
ifc = 0;
scale = 4;


for i = 1 : count
	HR = imread(fullfile(hr_dir(i).folder, hr_dir(i).name));
	Results = imread(fullfile(result_dir(i).folder, result_dir(i).name));
	[PSNR(i), SSIM(i), IFC(i)] = evaluate_SR(HR, Results, scale, ifc);
end

PSNR(count + 1) = mean(PSNR(:));
SSIM(count + 1) = mean(SSIM(:));

fprintf("Average PSNR is %f\n", PSNR(count + 1));
fprintf("Average SSIM is %f\n", SSIM(count + 1));

PSNR_path = fullfile('/4TB/datasets/LR-GOPRO/Validation_4x', 'PSNR-GFN_4x_HR.txt');
SSIM_path = fullfile('/4TB/datasets/LR-GOPRO/Validation_4x', 'SSIM-GFN_4x_HR.txt');
save_matrix(PSNR, PSNR_path);
save_matrix(SSIM, SSIM_path);


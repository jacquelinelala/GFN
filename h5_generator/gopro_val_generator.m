% -------------------------------------------------------------------------
%   Description:
%       A function to generate test dataset: blur_lr input images and hr
%       images.
%
%   Parameter:
%       - The download dataset GOPRO_Large's directory
%
%   Output:
%       - The generated images are stored in the directory of
%       GOPRO_Large/Validation_4x/LR_Blur and GOPRO_Large/Validation_4x/HR
%       
%   Citation: 
%       Gated Fusion Network for Joint Image Deblurring and Super-Resolution
%       The British Machine Vision Conference(BMVC2018 oral)
%       Xinyi Zhang, Hang Dong, Zhe Hu, Wei-Sheng Lai, Fei Wang and Ming-Hsuan Yang
%
%   Contact:
%       cvxinyizhang@gmail.com
%   Project Website:
%       http://xinyizhang.tech/bmvc2018
%       https://github.com/jacquelinelala/GFN

%%
function gopro_val_generator(folder)
%% scale factors
scale = 4;
test_folder = fullfile(folder,'test');
blur_root = fullfile(folder, sprintf('Validation_%dx/LR_Blur', scale));
hr_root = fullfile(folder, sprintf('Validation_%dx/HR', scale));

if ~isdir(blur_root)
    mkdir(blur_root)
end
if ~isdir(hr_root)
    mkdir(hr_root)
end

%% generate data
val_sets = dir(fullfile(test_folder,'GO*'));
parts = length(val_sets)
name_index = 0;
for n=1:parts
HR_image_path = fullfile(val_sets(n).folder, val_sets(n).name, 'sharp')
Blur_image_path = fullfile(val_sets(n).folder, val_sets(n).name, 'blur_gamma')
filepaths_HR = dir(fullfile(HR_image_path, '0*.png')); 
filepaths_BLur = dir(fullfile(Blur_image_path, '0*.png'));  

index = 0;
margain = 0;

for index = 1 : length(filepaths_HR)

                name_index = name_index + 1;
                image = imread(fullfile(filepaths_HR(index).folder,filepaths_HR(index).name));
                image_Blur = imread(fullfile(filepaths_BLur(index).folder,filepaths_BLur(index).name));
                if size(image,3)==3
                    image = im2double(image);
                    image_Blur = im2double(image_Blur);
                    HR_label1 = modcrop(image, 16);%save
                    Blur_label1 = modcrop(image_Blur, 16);
                    filepaths_HR(index).name
                            %Crop Blur patch
                            LR_BLur_input = imresize(Blur_label1,1/scale,'bicubic');%save
                            imwrite(LR_BLur_input, fullfile(blur_root,sprintf('%04d.png',name_index)));%save
                            imwrite(HR_label1, fullfile(hr_root,sprintf('%04d.png',name_index)));%save
                end
end
end


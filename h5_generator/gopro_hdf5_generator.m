% -------------------------------------------------------------------------
%   Description:
%       A function to generate train hdf5 patches.
%
%   Parameter:
%       - The download dataset GOPRO_Large's directory
%
%   Output:
%       - The generated patches are stored in hdf5 files in the directory of
%       GOPRO_Large/GOPRO_train256_4x_HDF5.
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
function gopro_hdf5_generator(folder)

scale = 4;
size_label = 256;
size_input = size_label/scale;
stride = 128;

train_folder = fullfile(folder, 'train');
save_root =fullfile(folder, sprintf('GOPRO_train%d_%dx_HDF5', size_label, scale));

if ~isdir(save_root)
    mkdir(save_root)
end

%% generate data
train_sets = dir(fullfile(train_folder,'GO*'));
parts = length(train_sets)
downsizes= [0.5,0.7,1];
for n=1:parts/2 %n
data = zeros(size_input, size_input, 3, 1);
label_db = zeros(size_input, size_input, 3, 1);
label = zeros(size_label, size_label, 3, 1);
count = 0;
margain = 0;    

for index = (n-1)*2 +1 :n*2 
HR_image_path = fullfile(train_folder, train_sets(index).name, 'sharp')
Blur_image_path = fullfile(train_folder, train_sets(index).name, 'blur_gamma')
filepaths_HR = dir(fullfile(HR_image_path, '0*.png')); 
filepaths_BLur = dir(fullfile(Blur_image_path, '0*.png')); 

for i = 1 : length(filepaths_HR)
    for downsize = 1:length(downsizes)
                image = imread(fullfile(HR_image_path,filepaths_HR(i).name));
                image_Blur = imread(fullfile(Blur_image_path,filepaths_BLur(i).name));
                image = imresize(image,downsizes(downsize),'bicubic');
                image_Blur = imresize(image_Blur,downsizes(downsize),'bicubic');
                if size(image,3)==3
                    image = im2double(image);
                    image_Blur = im2double(image_Blur);
                    HR_label = modcrop(image, scale);
                    Blur_label = modcrop(image_Blur, scale);
                    [hei,wid, c] = size(HR_label);
                    filepaths_HR(i).name
                    for x = 1 + margain : stride : hei-size_label+1 - margain
                        for y = 1 + margain :stride : wid-size_label+1 - margain
                            %Crop HR patch
                            HR_patch_label = HR_label(x : x+size_label-1, y : y+size_label-1, :);
                            [dx,dy] = gradient(HR_patch_label);
                            gradSum = sqrt(dx.^2 + dy.^2);
                            gradValue = mean(gradSum(:));
                            if gradValue < 0.005
                                continue;
                            end    
                            %Crop Blur patch
                            Blur_patch_label = Blur_label(x : x+size_label-1, y : y+size_label-1, :);
                            LR_BLur_input = imresize(Blur_patch_label,1/scale,'bicubic');
                            Deblur_label = imresize(HR_patch_label,1/scale,'bicubic');
                            count=count+1;
                            data(:, :, :, count) = LR_BLur_input;
                            label_db(:, :, :, count) = Deblur_label;
                            label(:, :, :, count) = HR_patch_label;
                        end % end of y 
                    end % end of x
                end % end of if
    end %end of downsize
end % end of i
end % end of index

order = randperm(count);
data = data(:, :, :, order);
label_db = label_db(:, :, :, order); 
label = label(:, :, :, order); 

%% writing to HDF5
savepath = fullfile(save_root ,sprintf('LR-GOPRO_x4_Part%d.h5', n))
chunksz = 64;
created_flag = false;
totalct = 0;

for batchno = 1:floor(count/chunksz)
    last_read=(batchno-1)*chunksz;
    batchdata = data(:,:,:,last_read+1:last_read+chunksz); 
    batchlabs_db = label_db(:,:,:,last_read+1:last_read+chunksz);
    batchlabs = label(:,:,:,last_read+1:last_read+chunksz);
    startloc = struct('dat',[1,1,1,totalct+1], 'lab_db', [1,1,1,totalct+1], 'lab', [1,1,1,totalct+1]);
    curr_dat_sz = store2hdf5(savepath, batchdata, batchlabs_db, batchlabs, ~created_flag, startloc, chunksz); 
    created_flag = true;
    totalct = curr_dat_sz(end);
end

h5disp(savepath);
end % index fo n
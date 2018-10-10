% -------------------------------------------------------------------------
%   Description:
%       Delete some images limited with some formats
%   Input:
%       - Validation/Results
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
%% deleting
function delete_image(folder)
path = fullfile(folder, 'Validation_4x/Results');
%path = fullfile(folder, 'Validation_4x/HR');
result_dir = dir(fullfile(path, '*GFN_4x.png'));
len = length(result_dir);

for i = 1 : len
    delete(fullfile(result_dir(i).folder, result_dir(i).name));
end
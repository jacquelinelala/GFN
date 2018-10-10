% -------------------------------------------------------------------------
%   Description:
%       Make a image get a modulo operation.
%
%   Parameter:
%
%   Output:
%       - A moduloed image.
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
%%
function imgs = modcrop(imgs, modulo)
if size(imgs,3)==1
    sz = size(imgs);
    sz = sz - mod(sz, modulo);
    imgs = imgs(1:sz(1), 1:sz(2));
else
    tmpsz = size(imgs);
    sz = tmpsz(1:2);
    sz = sz - mod(sz, modulo);
    imgs = imgs(1:sz(1), 1:sz(2),:);
end


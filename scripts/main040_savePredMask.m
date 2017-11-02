%LD_LIBRARY_PATH=/usr/local/cuda-7.5/lib64:local matlab -nodisplay
%{
run(fullfile(path_to_matconvnet, 'matlab', 'vl_setupnn'));
addpath(fullfile(path_to_matconvnet, 'matlab'));
vl_compilenn('enableGpu', true, ...
               'cudaRoot', '/usr/local/cuda-7.5', ...
               'cudaMethod', 'nvcc', ...
               'enableCudnn', true, ...
               'cudnnRoot', '/usr/local/cuda-7.5/cudnn-v5') ;

%}
clear
close all
clc;
rng(777);
addpath(genpath('../libs'))
path_to_matconvnet = './libs/matconvnet';
path_to_model = '../models/';

run(fullfile(path_to_matconvnet, 'matlab', 'vl_setupnn'));
addpath(genpath(fullfile('dependencies', 'matconvnet','examples')));

mean_r = 123.68; %means to be subtracted and the given values are used in our training stage
mean_g = 116.779;
mean_b = 103.939;
mean_bgr = reshape([mean_b, mean_g, mean_r], [1,1,3]);
mean_rgb = reshape([mean_r, mean_g, mean_b], [1,1,3]);
%% load imdb and dataset file
load('imdb_merge4cls.mat');

badAnchor = 35266;
imdb.labelList = [imdb.labelList(1:badAnchor-1), imdb.labelList(badAnchor+1:end)];
imdb.imgList = [imdb.imgList(1:badAnchor-1), imdb.imgList(badAnchor+1:end)];
imdb.maskList = [imdb.maskList(1:badAnchor-1), imdb.maskList(badAnchor+1:end)];
imdb.set = [imdb.set(1:badAnchor-1), imdb.set(badAnchor+1:end)];

imdb.meta.height = 512;
imdb.meta.width = 512;
imdb.meta.classNum = length(imdb.meta.className);
%% read matconvnet model
% set GPU
gpuId = 1; %[1, 2];
gpuDevice(gpuId);
flagSaveFig = true; % {true false} whether to store the result

saveFolder = 'main021_finegrained_segDet_res50';
modelName = 'softmax_net-epoch-27.mat';
%% setup network
netMat = load( fullfile('./exp', saveFolder, modelName) );
netMat = netMat.net;
netMat = dagnn.DagNN.loadobj(netMat);

rmLayerName = 'obj_seg';
if ~isnan(netMat.getLayerIndex(rmLayerName))
    sName = netMat.layers(netMat.getLayerIndex(rmLayerName)).inputs{1};
    netMat.removeLayer(rmLayerName); % remove layer

    baseName = 'res6';
    upsample_fac = 8;
    filters = single(bilinear_u(upsample_fac*2, 2, 2));
    crop = ones(1,4) * upsample_fac/2;
    deconv_name = [baseName, '_interp'];
    var_to_up_sample = sName;
    netMat.addLayer(deconv_name, ...
        dagnn.ConvTranspose('size', size(filters), ...
        'upsample', upsample_fac, ...
        'crop', crop, ...
        'opts', {'cudnn','nocudnn'}, ...
        'numGroups', 2, ...
        'hasBias', false), ...
        var_to_up_sample, deconv_name, {[deconv_name  '_f']}) ;
    ind = netMat.getParamIndex([deconv_name  '_f']) ;
    netMat.params(ind).value = filters ;
    sName = deconv_name;

    layerTop = sprintf('SoftMaxLayer');
    netMat.addLayer(layerTop, dagnn.SoftMax(),sName, layerTop);
    netMat.vars(netMat.layers(netMat.getLayerIndex(layerTop)).outputIndexes).precious = 1;
end

netMat.move('gpu');
netMat.mode = 'test' ;
% netMat.mode = 'normal' ;
netMat.conserveMemory = 1;
%% test 
saveFolder = [strrep(saveFolder,'/', '') '_visualization'];

dstFolder = '/home/skong2/pollenProject_dataset_predSegMask';
imdb.predSegMaskList = {};
for imgIdx = 1:length(imdb.imgList)
    imgPathName = imdb.imgList{imgIdx};
    imOrg = single(imread(imgPathName));
    
    [subpath, imgName, imgExt] = fileparts(imgPathName);
    [~, tmpClassName] = fileparts(subpath);
    if ~isdir(fullfile(dstFolder, tmpClassName))
        mkdir( fullfile(dstFolder, tmpClassName) );
    end
    dstPath = fullfile(dstFolder, tmpClassName, [imgName, imgExt]);
    %% feed into the network
    fprintf('image-%03d %s ... \n', imgIdx, imdb.imgList{imgIdx});
    imFeed = bsxfun(@minus, imOrg, mean_rgb);                    
    inputs = {'data', gpuArray(imFeed)};
    netMat.eval(inputs) ;
    %% gather the output 
    SoftMaxLayer = gather(netMat.vars(netMat.layers(netMat.getLayerIndex('SoftMaxLayer')).outputIndexes).value);        
    [~, predSeg] = max(SoftMaxLayer,[],3);
    %% visualization        
%     imgFig = figure(1);
%     subWindowH = 1; 
%     subWindowW = 2;
%     set(imgFig, 'Position', [100 100 1500 800]) % [1 1 width height]
%     windowID = 1;
%     
%     % ground-truth
%     subplot(subWindowH,subWindowW,windowID); windowID = windowID + 1;
%     imagesc(uint8(imOrg)); title(sprintf('image-%04d', imgIdx)); axis off image;
%     subplot(subWindowH,subWindowW,windowID); windowID = windowID + 1;
%     imagesc(predSeg); title(sprintf('predSeg')); axis off image; caxis([1,2]);
    imdb.predSegMaskList{imgIdx} = dstPath;
    imwrite(uint8(predSeg-1), dstPath);
%     %% save?   
%     if flagSaveFig && ~isdir(saveFolder)
%         mkdir(saveFolder);
%     end
%     if flagSaveFig
%         [~, curfilename, ~] = fileparts(imdb.val{imgIdx});
%         export_fig( sprintf('%s/visualization_%s.jpg', saveFolder,  curfilename) );
%     end
end

for imgIdx = 1:length(imdb.imgList)
    imgPathName = imdb.imgList{imgIdx};        
    [subpath, imgName, imgExt] = fileparts(imgPathName);
    [~, tmpClassName] = fileparts(subpath);    
    dstPath = fullfile(dstFolder, tmpClassName, [imgName, imgExt]);
    imdb.predSegMaskList{imgIdx} = dstPath;
end

save('imdb_merge4cls_predSeg.mat', 'imdb');
%% leaving blank


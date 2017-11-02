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
% load('imdb_segDet.mat');
% load('imdb_segDet_whole.mat');
load('imdb_cls_part1.mat');
imdb = imdb_cls;
imdb.meta.classNum = 2;
imdb.meta.height = 1000;
imdb.meta.width = 1000;

numTrainImg = 15000;
randIdx = randperm(length(imdb.imgList));
imdb.train = imdb.imgList(randIdx(1:numTrainImg));
imdb.val = imdb.imgList(randIdx(1+numTrainImg:end));
imdb.sets.name = {'train', 'val'};
%% read matconvnet model
% set GPU
gpuId = 3; %[1, 2];
gpuDevice(gpuId);
flagSaveFig = true; % {true false} whether to store the result

saveFolder = 'main021_finegrained_segDet_res50';
modelName = 'softmax_net-epoch-14.mat';
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
    filters = single(bilinear_u(upsample_fac*2, imdb.meta.classNum, imdb.meta.classNum));
    crop = ones(1,4) * upsample_fac/2;
    deconv_name = [baseName, '_interp'];
    var_to_up_sample = sName;
    netMat.addLayer(deconv_name, ...
        dagnn.ConvTranspose('size', size(filters), ...
        'upsample', upsample_fac, ...
        'crop', crop, ...
        'opts', {'cudnn','nocudnn'}, ...
        'numGroups', imdb.meta.classNum, ...
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
for imgIdx = 1:30%:length(imdb.val)
    imgPathName = fullfile(imdb.path_to_image, imdb.val{imgIdx});
    maskpath = strrep(imgPathName, imdb.path_to_image, imdb.path_to_mask);
    imOrg = single(imread(imgPathName));
    gtOrg = single(imread(maskpath));    
    
    %% feed into the network
    fprintf('image-%03d %s ... \n', imgIdx, imdb.val{imgIdx});
    imFeed = bsxfun(@minus, imOrg, mean_rgb);                    
    inputs = {'data', gpuArray(imFeed)};
    netMat.eval(inputs) ;
    %% gather the output 
    SoftMaxLayer = gather(netMat.vars(netMat.layers(netMat.getLayerIndex('SoftMaxLayer')).outputIndexes).value);        
    [~, predSeg] = max(SoftMaxLayer,[],3);
    %% visualization        
    imgFig = figure(1);
    subWindowH = 1; 
    subWindowW = 3;
    set(imgFig, 'Position', [100 100 1500 800]) % [1 1 width height]
    windowID = 1;
    
    % ground-truth
    subplot(subWindowH,subWindowW,windowID); windowID = windowID + 1;
    imagesc(uint8(imOrg)); title(sprintf('image-%04d', imgIdx)); axis off image;
    subplot(subWindowH,subWindowW,windowID); windowID = windowID + 1;
    imagesc(gtOrg+1); title(sprintf('gtSeg')); axis off image; caxis([1,2]);
    subplot(subWindowH,subWindowW,windowID); windowID = windowID + 1;
    imagesc(predSeg); title(sprintf('predSeg')); axis off image; caxis([1,2]);
    
    %% save?   
    if flagSaveFig && ~isdir(saveFolder)
        mkdir(saveFolder);
    end
    if flagSaveFig
        [~, curfilename, ~] = fileparts(imdb.val{imgIdx});
        export_fig( sprintf('%s/visualization_%s.jpg', saveFolder,  curfilename) );
    end
end
%% leaving blank


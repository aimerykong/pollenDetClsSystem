%% add path and setup configuration
%LD_LIBRARY_PATH=/usr/local/cuda-7.5/lib64:local matlab -nodisplay
% clear
% close all
clc;
rng(777);
addpath('./libs/exportFig')
addpath('./libs/layerExt')
% addpath('./libs/matconvnet')
addpath('./libs/myFunctions')

path_to_model = '/home/data/skong2/MarkovCNN/basemodels/';
path_to_dataset = '../dataset/';

path_to_matconvnet = './libs/matconvnet';
run(fullfile(path_to_matconvnet, 'matlab', 'vl_setupnn'));
addpath(genpath(fullfile('dependencies', 'matconvnet','examples')));

% set GPU 
gpuId = 2; %[1, 2];
gpuDevice(gpuId);
%% load imdb and dataset file
% load('imdb_segDet.mat');
load('imdb_segDet_whole.mat');
imdb.meta.height = 1000;
imdb.meta.width = 1000;
%% sepcify model
% netbasemodel = load(fullfile(path_to_model, 'imagenet-vgg-verydeep-16.mat' ));
netbasemodel = load(fullfile(path_to_model, 'imagenet-resnet-50-dag.mat' ));
netbasemodel = dagnn.DagNN.loadobj(netbasemodel);

for i = 1:length(netbasemodel.layers)
    if isfield(netbasemodel.layers(i).block, 'bnorm_moment_type_trn')
        netbasemodel.layers(i).block.bnorm_moment_type_trn = 'global';
        netbasemodel.layers(i).block.bnorm_moment_type_tst = 'global';
    end
end
RFinfo = netbasemodel.getVarReceptiveFields('data');
%% modify the basemodel to fit segmentation task
% add objective function layer
scalingFactor = 1;

netbasemodel.meta.normalization.averageImage = reshape([123.68, 116.779,  103.939],[1,1,3]); % imagenet mean values
netbasemodel.meta.normalization.imageSize = [imdb.meta.height, imdb.meta.width, 3, 1];
netbasemodel.meta.normalization.border = [8, 8]; % 720x720
netbasemodel.meta.normalization.stepSize = [1, 1];
%% modify the network
netbasemodel.removeLayer('prob'); % remove layer
netbasemodel.removeLayer('fc1000'); % remove layer
netbasemodel.removeLayer('pool5'); % remove layer

% move res3's pooling
netbasemodel.layers(37).block.stride = [1 1]; %res3a_branch1
netbasemodel.layers(39).block.stride = [1 1]; %res3a_branch2a
% modify res4's stride from 2 to 1
netbasemodel.layers(79).block.stride = [1 1]; % res4a_branch1
netbasemodel.layers(81).block.stride = [1 1]; % res4a_branch2a


sName = 'res5cx';
lName = 'res6_conv';
block = dagnn.Conv('size', [1 1 2048 2], 'hasBias', true, 'stride', 1, 'pad', 0, 'dilate', 1);
netbasemodel.addLayer(lName, block, sName, lName, {[lName '_f'], [lName '_b']});
ind = netbasemodel.getParamIndex([lName '_b']);
netbasemodel.params(ind).value = zeros([1 2], 'single');
netbasemodel.params(ind).learningRate = 20;
ind = netbasemodel.getParamIndex([lName '_f']);
weights = randn(1, 1, 2048, 2, 'single')*sqrt(2/5);
netbasemodel.params(ind).value = weights;
netbasemodel.params(ind).learningRate = 10;
sName = lName;


obj_name = sprintf('obj_seg');
gt_name =  sprintf('gt_seg');
input_name = sName;
netbasemodel.addLayer(obj_name, ...
    SegmentationLossLogistic('loss', 'softmaxlog'), ... softmaxlog logistic
    {input_name, gt_name}, obj_name)

%% show learning rates for all layers
for i = 1:numel(netbasemodel.layers)            
    curLayerName = netbasemodel.layers(i).name;
    if ~isempty(strfind(curLayerName, 'bn'))
        netbasemodel.layers(i).block.bnorm_moment_type_trn = 'global';
        netbasemodel.layers(i).block.bnorm_moment_type_tst = 'global';
        netbasemodel.params(netbasemodel.layers(i).paramIndexes(3)).learningRate = 0;
%         fprintf('%d \t%s, \t\t%.2f\n',i, netbasemodel.params(netbasemodel.layers(i).paramIndexes(3)).name, netbasemodel.params(netbasemodel.layers(i).paramIndexes(3)).learningRate);        
    end
end 

% netbasemodel.params(netbasemodel.getParamIndex('res5_1_projBranch_f')).learningRate = 1;
% netbasemodel.params(netbasemodel.getParamIndex('res4_1_projBranch_f')).learningRate = 1;
% for i = 221:230
%     netbasemodel.params(i).learningRate = netbasemodel.params(i).learningRate / 4;    
% end

for i = 1:numel(netbasemodel.params)            
    fprintf('%d \t%s, \t\t%.2f\n',i, netbasemodel.params(i).name, netbasemodel.params(i).learningRate);   
end  
%% configure training environment
batchSize = 1;
totalEpoch = 50;
learningRate = 1:totalEpoch;
learningRate = (2.5e-4) * (1-learningRate/totalEpoch).^0.9; % epoch 12

weightDecay=0.0005; % weightDecay: usually use the default value

opts.batchSize = batchSize;
opts.learningRate = learningRate;
opts.weightDecay = weightDecay;
opts.momentum = 0.9 ;

% opts.withDepth = false ;
% opts.withDepthRegression = false ;
% opts.withDepthClassification = false ;

opts.scalingFactor = scalingFactor;

opts.expDir = fullfile('./exp', 'main011_segDet_res50');
if ~isdir(opts.expDir)
    mkdir(opts.expDir);
end

opts.numSubBatches = 1 ;
opts.continue = true ;
opts.gpus = gpuId ;
%gpuDevice(opts.train.gpus); % don't want clear the memory
opts.prefetch = false ;
opts.sync = false ; % for speed
opts.cudnn = true ; % for speed
opts.numEpochs = numel(opts.learningRate) ;
opts.learningRate = learningRate;


numTrainImg = 15000;
randIdx = randperm(length(imdb.imgList));
imdb.train = imdb.imgList(randIdx(1:numTrainImg));
imdb.val = imdb.imgList(randIdx(1+numTrainImg:end));
imdb.trainMask = imdb.maskList(randIdx(1:numTrainImg)); 
imdb.valMask = imdb.maskList(randIdx(1+numTrainImg:end)); 

imdb.sets.name = {'train', 'val'};
for i = 1:2
    curSetName = imdb.sets.name{i};
    opts.(curSetName) = imdb.(curSetName);
end

opts.checkpointFn = [];
mopts.classifyType = 'softmax';

rng(777);
bopts = netbasemodel.meta.normalization;
bopts.numThreads = 12;
bopts.imdb = imdb;
%% train
fn = getImgBatchWrapper_pollen4DetSeg(bopts);

opts.backPropDepth = inf; % could limit the backprop
prefixStr = [mopts.classifyType, '_'];
opts.backPropAboveLayerName = 'res5a_branch1';% ??

trainfn = @cnnTrainDetSeg;
[netbasemodel, info] = trainfn(netbasemodel, prefixStr, imdb, fn, 'derOutputs', {...
    sprintf('obj_seg'), 1},...
    opts);
%% leaving blank
%{
%}




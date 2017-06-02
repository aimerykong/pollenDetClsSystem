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
gpuId = 1; %[1, 2];
gpuDevice(gpuId);
%% load imdb and dataset file
load('imdb_merge4seg.mat');
imdb.meta.height = 512;
imdb.meta.width = 512;
%% sepcify model
netbasemodel = load(fullfile('./exp', 'main021_finegrained_segDet_res50', 'softmax_net-epoch-29.mat' ));
netbasemodel = netbasemodel.net;
netbasemodel = dagnn.DagNN.loadobj(netbasemodel);
%% show learning rates for all layers
scalingFactor = 1;
for i = 1:numel(netbasemodel.params)            
    fprintf('%d \t%s, \t\t%.2f\n',i, netbasemodel.params(i).name, netbasemodel.params(i).learningRate);   
end  
%% configure training environment
batchSize = 1;
totalEpoch = 30;
learningRate = 1:totalEpoch;
learningRate = (1e-5) * (1-learningRate/totalEpoch).^0.9; % epoch 12

weightDecay=0.0005; % weightDecay: usually use the default value

opts.batchSize = batchSize;
opts.learningRate = learningRate;
opts.weightDecay = weightDecay;
opts.momentum = 0.9 ;

% opts.withDepth = false ;
% opts.withDepthRegression = false ;
% opts.withDepthClassification = false ;

opts.scalingFactor = scalingFactor;

opts.expDir = fullfile('./exp', 'main023_finegrained_segDet_res50_mergeimdb');
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
% fn = getImgBatchWrapper_pollen4DetSeg(bopts);
% fn = getImgBatchWrapper_pollen4fgDetSeg(bopts);
fn = getImgBatchWrapper_pollen4fgDetSeg_mergeimdb(bopts);

opts.backPropDepth = inf; % could limit the backprop
prefixStr = [mopts.classifyType, '_'];
opts.backPropAboveLayerName = 'res5a_branch1';% ??
% opts.backPropAboveLayerName = 'res4a_branch1';% ??

trainfn = @cnnTrainDetSeg;
[netbasemodel, info] = trainfn(netbasemodel, prefixStr, imdb, fn, 'derOutputs', {...
    sprintf('obj_seg'), 1},...
    opts);
%% leaving blank
%{
%}




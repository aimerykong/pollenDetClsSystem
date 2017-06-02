%% add path and setup configuration
%LD_LIBRARY_PATH=/usr/local/cuda-7.5/lib64:local matlab -nodisplay
% clear
% close all
clc;
rng(777);
addpath('./libs/exportFig')
addpath('./libs/layerExt')
addpath('./libs/myFunctions')

path_to_model = '/home/data/skong2/MarkovCNN/basemodels/';
path_to_dataset = '../dataset/';

path_to_matconvnet = './libs/matconvnet';
run(fullfile(path_to_matconvnet, 'matlab', 'vl_setupnn'));
addpath(genpath(fullfile('dependencies', 'matconvnet','examples')));

% set GPU 
gpuId = 3; %[1, 2];
gpuDevice(gpuId);
%% load imdb and dataset file
load('imdb_gtPatch4cls.mat');
imdb.meta.height = 512;
imdb.meta.width = 512;
imdb.meta.classNum = length(imdb.meta.className);
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
% netbasemodel.layers(37).block.stride = [1 1]; %res3a_branch1
% netbasemodel.layers(39).block.stride = [1 1]; %res3a_branch2a
% modify res4's stride from 2 to 1
% netbasemodel.layers(79).block.stride = [1 1]; % res4a_branch1
% netbasemodel.layers(81).block.stride = [1 1]; % res4a_branch2a


sName = 'res5cx';
lName = 'GlobalPooling';
netbasemodel.addLayer( lName, ...
    dagnn.Pooling('poolSize', [16 16], 'method', 'avg'), ...
    sName, ...
    lName) ;
sName = lName;


lName = 'res6_conv';
block = dagnn.Conv('size', [1 1 2048 imdb.meta.classNum], 'hasBias', true, 'stride', 1, 'pad', 0, 'dilate', 1);
netbasemodel.addLayer(lName, block, sName, lName, {[lName '_f'], [lName '_b']});
ind = netbasemodel.getParamIndex([lName '_b']);
netbasemodel.params(ind).value = zeros([1 imdb.meta.classNum], 'single');
netbasemodel.params(ind).learningRate = 20;
ind = netbasemodel.getParamIndex([lName '_f']);
weights = randn(1, 1, 2048, imdb.meta.classNum, 'single')*sqrt(imdb.meta.classNum/5);
netbasemodel.params(ind).value = weights;
netbasemodel.params(ind).learningRate = 10;
sName = lName;

obj_name = sprintf('obj_cls');
gt_name =  sprintf('label');
input_name = sName;
netbasemodel.addLayer(obj_name, ...
    SegmentationLossLogistic('loss', 'softmaxlog'), ... softmaxlog logistic
    {input_name, gt_name}, obj_name)

netbasemodel.addLayer('top1err', dagnn.Loss('loss', 'classerror'), ...
      {sName, 'label'}, 'error') ;

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
learningRate = (2.5e-6) * (1-learningRate/totalEpoch).^0.9; % epoch 12

weightDecay=0.0005; % weightDecay: usually use the default value

opts.batchSize = batchSize;
opts.learningRate = learningRate;
opts.weightDecay = weightDecay;
opts.momentum = 0.9 ;

% opts.withDepth = false ;
% opts.withDepthRegression = false ;
% opts.withDepthClassification = false ;

opts.scalingFactor = scalingFactor;

opts.expDir = fullfile('./exp', 'main030_cls_res50_gtOnly_avgPool_masking');
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
    tmp = find(imdb.set==i);
    opts.(curSetName) = tmp;
end

opts.checkpointFn = [];
mopts.classifyType = 'softmax';

rng(777);
bopts = netbasemodel.meta.normalization;
bopts.numThreads = 12;
bopts.imdb = imdb;
%% train
fn = getImgBatchWrapper_pollen4cls_masking(bopts);

opts.backPropDepth = inf; % could limit the backprop
prefixStr = [mopts.classifyType, '_'];
% opts.backPropAboveLayerName = 'res5a_branch1';% ??
% opts.backPropAboveLayerName = 'res4a_branch1';% ??
opts.backPropAboveLayerName = 'res3a_branch1';% ??

trainfn = @cnnTrainPollenCls_masking;
[netbasemodel, info] = trainfn(netbasemodel, prefixStr, imdb, fn, 'derOutputs', {...
    sprintf('obj_cls'), 1},...
    opts);
%% leaving blank
%{
%}




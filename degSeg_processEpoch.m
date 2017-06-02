function [stats, prof] = degSeg_processEpoch(net, state, scalingFactor, opts, mode, totalEpoch)
% -------------------------------------------------------------------------

%% initialize empty momentum
if strcmp(mode,'train')
    state.momentum = num2cell(zeros(1, numel(net.params))) ;
end

%% move CNN  to GPU as needed
numGpus = numel(opts.gpus) ;
if numGpus >= 1
    net.move('gpu') ;
    if strcmp(mode,'train')
        state.momentum = cellfun(@gpuArray,state.momentum,'UniformOutput',false) ;
    end
end
if numGpus > 1
    mmap = map_gradients(opts.memoryMapFile, net, numGpus) ;
else
    mmap = [] ;
end

%% profile
if opts.profile
    if numGpus <= 1
        profile clear ;
        profile on ;
    else
        mpiprofile reset ;
        mpiprofile on ;
    end
end

subset = state.(mode) ;
num = 0 ;
stats.num = 0 ; % return something even if subset = []
stats.time = 0 ;
adjustTime = 0 ;

start = tic ;
% A = cell(1,2);
% Amax = zeros(1,2);
% B = cell(1,2);
% Bmax = zeros(1,2);
% C = cell(1,2);
% Cmax = zeros(1,2);
% D = cell(1,2);
% Dmax = zeros(1,2);
% E = cell(1,2);
% Emax = zeros(1,2);
% F = cell(1,2);
% Fmax = zeros(1,2);
for t=1:opts.batchSize:numel(subset)
    fprintf('%s: epoch %02d/%03d: %3d/%3d:', mode, state.epoch, totalEpoch, ...
        fix((t-1)/opts.batchSize)+1, ceil(numel(subset)/opts.batchSize)) ;
    batchSize = min(opts.batchSize, numel(subset) - t + 1) ;
    
    for s=1:opts.numSubBatches
        % get this image batch and prefetch the next
        batchStart = t + (labindex-1) + (s-1) * numlabs ;
        batchEnd = min(t+opts.batchSize-1, numel(subset)) ;
        batch = subset(batchStart : opts.numSubBatches * numlabs : batchEnd) ;
        num = num + numel(batch) ;
        if numel(batch) == 0, continue ; end
                
        [image, mask, imageOrg] = state.getBatch(batch, mode, scalingFactor) ;    
        mask = imresize(mask, 1/8, 'nearest');
        %% fetch data for train/test        
%         figure(1); subplot(2,3,1); imagesc(uint8(imo)); axis off image;
        inputs = {};
        if numGpus >= 1
            image = gpuArray(image) ;
            inputs{end+1} = 'data';
            inputs{end+1} = image;  
            inputs{end+1} = 'gt_seg';  
            inputs{end+1} = mask;               
        end
        
        if opts.prefetch
            if s == opts.numSubBatches
                batchStart = t + (labindex-1) + opts.batchSize ;
                batchEnd = min(t+2*opts.batchSize-1, numel(subset)) ;
            else
                batchStart = batchStart + numlabs ;
            end
            nextBatch = subset(batchStart : opts.numSubBatches * numlabs : batchEnd) ;
            state.getBatch(nextBatch, mode, scalingFactor) ;
%             state.getBatch(state.imdb, nextBatch) ;
        end
        %% feedforward and/or backprop
        if strcmp(mode, 'train')
%             net.mode = lower('trainGlobalBN') ; % trainGlobalBN, trainLocalBN, test, normal
            net.mode = 'normal' ;
            net.accumulateParamDers = (s ~= 1) ;
            
%             outputIdx = net.layers(net.getLayerIndex('res7_interp')).outputIndexes;
%             net.vars(outputIdx).precious = 1;            
            net.eval(inputs, opts.derOutputs) ;
%             net.eval(inputs, opts.derOutputs, 'backPropAboveLayerName', opts.backPropAboveLayerName) ;
           %{
            img = inputs{2};
            img = bsxfun(@plus, img, net.meta.normalization.averageImage);
            
            labelDict = load('labelDictionary.mat');
            validIndex = find(labelDict.ignoreInEval==0);
            colorLabel = labelDict.colorLabel(validIndex,:);       
            
            pred = gather(net.vars(outputIdx).value);
            [~, pred] = max(pred, [], 3);
            pred = index2RGBlabel(pred-1, colorLabel);
            
            gt = inputs{4};
%             [~, gt] = max(gt, [], 3);
            gt = index2RGBlabel(gt-1, colorLabel);
            
            figure; 
            subplot(1,3,1); imshow(uint8(img)); title('image');
            subplot(1,3,2); imshow(uint8(gt)); title('gt');
            subplot(1,3,3); imshow(uint8(pred)); title('pred');
            disp('done');
            %}
        else
            net.mode = 'test' ;
%             net.mode = 'normal' ;
%             net.conserveMemory = 0;
%             outputIdx = net.layers(net.getLayerIndex('res7_interp')).outputIndexes;
%             net.vars(outputIdx).precious = 1;
            net.eval(inputs) ;
            %{
            img = inputs{2};
            img = bsxfun(@plus, img, net.meta.normalization.averageImage);
            
            labelDict = load('labelDictionary.mat');
            validIndex = find(labelDict.ignoreInEval==0);
            colorLabel = labelDict.colorLabel(validIndex,:);       
            
%             outputIdx = net.layers(net.getLayerIndex('res7_interp')).outputIndexes;
            pred = gather(net.vars(outputIdx).value);
            [~, pred] = max(pred, [], 3);
            pred = index2RGBlabel(pred-1, colorLabel);
            
            gt = inputs{4};
%             [~, gt] = max(gt, [], 3);
            gt = index2RGBlabel(gt-1, colorLabel);
            
            figure; 
            subplot(1,3,1); imshow(uint8(img)); title('image');
            subplot(1,3,2); imshow(uint8(gt)); title('gt');
            subplot(1,3,3); imshow(uint8(pred)); title('pred');
            disp('done');
            %}
        end
    end
    
%     Amax(t,1) = max(abs(gather(net.params(210).value(:)))); % res5c_branch2c_filter
%     Amax(t,2) = max(abs(gather(net.params(210).der(:))));
%     A{t,1} = net.params(210).value;
%     A{t,2} = net.params(210).der;    
% %     fprintf('\tder-res5c_branch2c_filter: %.5f\n', max(abs(gather(net.params(210).der(:)))));
%     
%     Bmax(t,1) = max(abs(gather(net.params(211).value(:)))); % bn5c_branch2c_mult
%     Bmax(t,2) = max(abs(gather(net.params(211).der(:))));
%     B{t,1} = net.params(211).value;
%     B{t,2} = net.params(211).der;
%     
%     Cmax(t,1) = max(abs(gather(net.params(212).value(:)))); % bn5c_branch2c_bias
%     Cmax(t,2) = max(abs(gather(net.params(212).der(:))));
%     C{t,1} = net.params(212).value;
%     C{t,2} = net.params(212).der;
%     
%     Dmax(t,1) = max(abs(gather(net.params(213).value(:)))); % bn5c_branch2c_moments
%     Dmax(t,2) = max(abs(gather(net.params(213).der(:))));
%     D{t,1} = net.params(213).value;
%     D{t,2} = net.params(213).der;
%     
%     Emax(t,1) = max(abs(gather(net.params(214).value(:)))); % res6_conv_f
%     Emax(t,2) = max(abs(gather(net.params(214).der(:))));
%     E{t,1} = net.params(214).value;
%     E{t,2} = net.params(214).der;
%     
%     Fmax(t,1) = max(abs(gather(net.params(215).value(:)))); % res6_conv_b
%     Fmax(t,2) = max(abs(gather(net.params(215).der(:))));
%     F{t,1} = net.params(215).value;
%     F{t,2} = net.params(215).der;
    
%     for iiii = 1:length(net.params)
%         fprintf('%03d %s %f\n', iiii, net.params(iiii).name, max(abs(net.params(iiii).der(:))));
%     end

%     if t == 1
%         disp(t);
%         disp(' ');
%         disp(Amax);
%         disp(Bmax);
%         disp(Cmax);
%         disp(Dmax);
%         disp(Emax);
%         disp(Fmax);
%     end

    %% accumulate gradient
    if strcmp(mode, 'train')
        if ~isempty(mmap)
            write_gradients(mmap, net) ;
            labBarrier() ;
        end
        state = accumulate_gradients(state, net, opts, batchSize, mmap) ;
    end
    
    % get statistics
    time = toc(start) + adjustTime ;
    batchTime = time - stats.time ;
    stats = opts.extractStatsFn(net) ;
    stats.num = num ;
    stats.time = time ;
    currentSpeed = batchSize / batchTime ;
    averageSpeed = (t + batchSize - 1) / time ;
    if t == opts.batchSize + 1
        % compensate for the first iteration, which is an outlier
        adjustTime = 2*batchTime - time ;
        stats.time = time + adjustTime ;
    end
    
    fprintf(' %.1f (%.1f) Hz', averageSpeed, currentSpeed) ;
    for f = setdiff(fieldnames(stats)', {'num', 'time'})
        f = char(f) ;
        fprintf(' %s:', f) ;
        fprintf(' %.4f', stats.(f)) ;
    end
    fprintf('\n') ;
end

if ~isempty(mmap)
    unmap_gradients(mmap) ;
end

if opts.profile
    if numGpus <= 1
        prof = profile('info') ;
        profile off ;
    else
        prof = mpiprofile('info');
        mpiprofile off ;
    end
else
    prof = [] ;
end

net.reset() ;
net.move('cpu') ;

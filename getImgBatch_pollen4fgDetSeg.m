function [imt, mask, imo] = getImgBatch_pollen4fgDetSeg(images, mode, scaleFactor, varargin)
% CNN_IMAGENET_GET_BATCH  Load, preprocess, and pack images for CNN evaluation
opts.imageSize = [512, 512] ;
opts.border = [0, 0] ;
opts.stepSize = [1, 1] ;
opts.lambda = 1 ;
opts.keepAspect = true ;
opts.numAugments = 1 ; % flip?
opts.transformation = 'none' ;  % 'stretch' 'none'
opts.averageImage = [] ;
% opts.rgbVariance = 1*ones(1,1,'single') ; % default: zeros(0,3,'single') ;
opts.rgbVariance = zeros(0,3,'single') ;
opts.classNum = 19;

opts.interpolation = 'bilinear' ;
opts.numThreads = 1 ;
opts.prefetch = false ;
opts.imdb = [];

opts = vl_argparse(opts, varargin);

% global dataset
%% read mat (hdf5) file for image and label

if strcmpi(mode, 'train') || strcmpi(mode, 'val')
    imt = zeros(opts.imageSize(1)-opts.border(1), opts.imageSize(2)-opts.border(2), 3, numel(images), 'single') ;
    mask = zeros((opts.imageSize(1)-opts.border(1))/scaleFactor, (opts.imageSize(2)-opts.border(2))/scaleFactor, 1, numel(images), 'single') ;
    imo = zeros(opts.imageSize(1)-opts.border(1), opts.imageSize(2)-opts.border(2), 3, numel(images), 'single') ;
else
    imt = zeros(opts.imageSize(1), opts.imageSize(2), 3, numel(images), 'single') ;
    mask = zeros(opts.imageSize(1)/scaleFactor, opts.imageSize(2)/scaleFactor, 1, numel(images), 'single') ;    
end

for img_i = 1:1%:numel(images)
    if  strcmpi(mode, 'val')        
        flag_flip = 0;
%         xstart = 1;
%         ystart = 1;
%         
%         xend = opts.imageSize(2) - (opts.border(2) - xstart+1);
%         yend = opts.imageSize(1) - (opts.border(1) - ystart+1);
    else
        flag_flip = rand(1)>0.5;
%         xstart = randperm(opts.border(2) / opts.stepSize(2) + 1,1)*opts.stepSize(2) - opts.stepSize(2) + 1;
%         ystart = randperm(opts.border(1) / opts.stepSize(1) + 1,1)*opts.stepSize(1) - opts.stepSize(1) + 1;
%         
%         xend = opts.imageSize(2) - (opts.border(2) - xstart+1);
%         yend = opts.imageSize(1) - (opts.border(1) - ystart+1);
    end
    %% read the image and annotation    
%     maskpath = strrep(images{1}, opts.imdb.path_to_image, opts.imdb.path_to_mask);
%     maskpath = strrep(maskpath, 'png', 'bmp');
    maskpath = fullfile(opts.imdb.path_to_mask, images{1});
    imgOrg = single(imread(fullfile(opts.imdb.path_to_image, images{1}) ));
    gtOrg = single(imread(maskpath));    
            
    gtOrg(gtOrg~=0) = 2;
    gtOrg(gtOrg==0) = 1;
    
    %% augmentation
    if strcmpi(mode, 'train') 
        if flag_flip
            %% flip augmentation
            imgOrg = fliplr(imgOrg);
            gtOrg = fliplr(gtOrg);
        end        
        %% crop augmentation
%         imgOrg = imgOrg(ystart:yend, xstart:xend,:);              
%         gtOrg = gtOrg(ystart:yend, xstart:xend,:);    
        %% random rotation        
        angle = 90 * randperm(4, 1);        
        imgOrg = imrotate(imgOrg, angle, 'bicubic');
        gtOrg = imrotate(gtOrg, angle, 'nearest');
    elseif strcmpi(mode, 'val') 
        %% crop augmentation
%         imgOrg = imgOrg(ystart:yend, xstart:xend,:);    
%         gtOrg = gtOrg(ystart:yend, xstart:xend,:);                   
    end        
    %% return
%     figure(1);
%     subplot(1,2,1); imshow(uint8(imgOrg));
%     subplot(1,2,2); imagesc(gtOrg); axis off image; colorbar; caxis([0,2]);
    
    imo = imgOrg;
    imt = bsxfun(@minus, imo(:,:,:,img_i), opts.averageImage) ;   
    mask = gtOrg;%curGT_Stack;          
end
% finishFlag = true;
%% leaving blank



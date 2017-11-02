function [imt, mask, imo, label] = getImgBatch_pollen4cls_masking(images, mode, scaleFactor, varargin)
% CNN_IMAGENET_GET_BATCH  Load, preprocess, and pack images for CNN evaluation
opts.imageSize = [1000, 1000] ;
opts.border = [8, 8] ;
opts.stepSize = [32, 32] ;
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

% if strcmpi(mode, 'train') || strcmpi(mode, 'val')
%     imt = zeros(opts.imageSize(1)-opts.border(1), opts.imageSize(2)-opts.border(2), 3, numel(images), 'single') ;
%     mask = zeros((opts.imageSize(1)-opts.border(1))/scaleFactor, (opts.imageSize(2)-opts.border(2))/scaleFactor, 1, numel(images), 'single') ;
%     imo = zeros(opts.imageSize(1)-opts.border(1), opts.imageSize(2)-opts.border(2), 3, numel(images), 'single') ;
% else
%     imt = zeros(opts.imageSize(1), opts.imageSize(2), 3, numel(images), 'single') ;
%     mask = zeros(opts.imageSize(1)/scaleFactor, opts.imageSize(2)/scaleFactor, 1, numel(images), 'single') ;    
% end

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
    gtOrg = 0;
    imgOrg = single(imread( opts.imdb.imgList{images} ));
    gtOrg = single(imread( opts.imdb.maskList{images} ));                
%     gtOrg(gtOrg~=0) = 2;
%     gtOrg(gtOrg==0) = 1;
    
    label = opts.imdb.labelList(images);
    
    
%     figure(1);
%     subplot(2,2,1); imshow(uint8(imgOrg));
%     subplot(2,2,2); imagesc(gtOrg); axis off image; colorbar; caxis([0,2]);
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
    imo = imgOrg;
    imt = bsxfun(@minus, imo(:,:,:,img_i), opts.averageImage) ;  
    
    if opts.imageSize(1)<size(imo,1)
        tmp = zeros(opts.imageSize, 'single');
        yxNew = floor((size(imo) - size(tmp))/2); yxNew(1) = max(yxNew(1),1); yxNew(2) = max(yxNew(2),1);
        imo = imo(yxNew(1):yxNew(1)+opts.imageSize(1)-1, yxNew(2):yxNew(2)+opts.imageSize(2)-1, :);
        imt = imt(yxNew(1):yxNew(1)+opts.imageSize(1)-1, yxNew(2):yxNew(2)+opts.imageSize(2)-1, :);
        gtOrg = gtOrg(yxNew(1):yxNew(1)+opts.imageSize(1)-1, yxNew(2):yxNew(2)+opts.imageSize(2)-1, :);
                
%         figure(1);
%         subplot(1,2,1);
%         imshow(uint8(imo));
%         subplot(1,2,2);
%         imshow((gtOrg));
%         a = 1;
    elseif opts.imageSize(1)>size(imo,1)
        tmp = zeros(opts.imageSize, 'single');
        yxNew = floor((size(tmp) - size(imo))/2); yxNew(1) = max(yxNew(1),1); yxNew(2) = max(yxNew(2),1);
        tmp(yxNew(1):yxNew(1)+size(imo,1)-1, yxNew(2):yxNew(2)+size(imo,2)-1, :) = imo;
        imo = tmp;
        
        tmp = zeros(opts.imageSize, 'single');
        tmp(yxNew(1):yxNew(1)+size(imt,1)-1, yxNew(2):yxNew(2)+size(imt,2)-1, :) = imt;
        imt = tmp;
                 
        tmp = zeros(opts.imageSize(1), opts.imageSize(2), 'single');
        tmp(yxNew(1):yxNew(1)+size(gtOrg,1)-1, yxNew(2):yxNew(2)+size(gtOrg,2)-1, :) = gtOrg;
        gtOrg = tmp;        
    else
    end    
    mask = gtOrg;%curGT_Stack;          
end
% finishFlag = true;
%% leaving blank



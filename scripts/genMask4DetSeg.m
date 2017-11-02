function [imgWindowList, maskWindowList, ulrbWindowList, centerWindowList] = genMask4DetSeg(netMat, imgList, mean_rgb)

netMat.move('gpu');
netMat.mode = 'test' ;
netMat.conserveMemory = 1;
%% feed into the network
% predSegList = cell(1,length(imgList));
imgWindowList = cell(1,length(imgList));
maskWindowList = cell(1,length(imgList));
ulrbWindowList = cell(1,length(imgList));
centerWindowList = cell(1,length(imgList));

for i = 1:length(imgList)
    %         fprintf('image: %s ... \n', titleList{i});
    imFeed = bsxfun(@minus, single(imgList{i}), mean_rgb);
    inputs = {'data', gpuArray(imFeed)};
    netMat.eval(inputs) ;
    %% gather the output
    SoftMaxLayer = gather(netMat.vars(netMat.layers(netMat.getLayerIndex('SoftMaxLayer')).outputIndexes).value);
    respMap = SoftMaxLayer(:,:,2);
%     [~, predSeg] = max(SoftMaxLayer,[],3);
    %% NMS for attention-aware patches
    respMapMask = (respMap>0.5);
    se = strel('disk', 10);
    respMapMask = imdilate(respMapMask, se); % for connected component, get radius
    

%     maxResp_x8 = imRespNMS( imresize(respMap, 1/8, 'bilinear') );
%     maxResp_x4 = imRespNMS( imresize(maxResp_x8, 2, 'bilinear') );
%     maxResp_x2 = imRespNMS( imresize(maxResp_x4, 2, 'bilinear') );
%     maxResp = imRespNMS( imresize(maxResp_x2, 2, 'bilinear') );

%     maxResp = nonmax(respMap, 10);

    maxResp = imRespNMS(respMap);
    maxResp(maxResp<0.5) = 0;
    maxResp = maxResp .* respMapMask; % for the grain centers
    
    %         CC = bwconncomp(respMapMask); %% connected component
    ccLabel = bwlabel(respMapMask);
    centerList = find(maxResp);
    
    imgWindowList{i} = {};
    maskWindowList{i} = {};
    ulrbWindowList{i} = {};
    centerWindowList{i} = {};
    
    for centerIdx = 1:length(centerList)
        curCenter = centerList(centerIdx);
        [cur_relativeY, cur_relativeX] = ind2sub(size(respMapMask), curCenter);        
        centerWindowList{i}{centerIdx} = [cur_relativeY, cur_relativeX];
        
        cur_winSize = length(ccLabel==ccLabel(curCenter));
        cur_winSize = sqrt(cur_winSize)+10;
        if cur_winSize < 100
            cur_winSize = 100;
        elseif cur_winSize>500
            cur_winSize = 512;
        end
        UL = round([cur_relativeY, cur_relativeX]-cur_winSize/2);
        BR = UL + cur_winSize-1;
        if UL(1)<1
            UL(1) = 1;
            BR(1) = cur_winSize;
        end
        if UL(2)<1
            UL(2) = 1;
            BR(2) = cur_winSize;
        end
        if BR(1)>size(respMapMask,1)
            BR(1) = size(respMapMask,1);
            UL(1) = BR(1)-cur_winSize+1;
        end
        if BR(2)>size(respMapMask,2)
            BR(2) = size(respMapMask,2);
            UL(2) = BR(2)-cur_winSize+1;
        end
        
        
        curImgWindow = imgList{i}(UL(1):BR(1), UL(2):BR(2), :);
        curMaskWindow = respMapMask(UL(1):BR(1), UL(2):BR(2), :);        
        imgWindowList{i}{centerIdx} = curImgWindow;
        maskWindowList{i}{centerIdx} = curMaskWindow;
        ulrbWindowList{i}{centerIdx} = [UL(1),BR(1),UL(2),BR(2)];
    end      
end

netMat.move('cpu');


% return a get batch function
% -------------------------------------------------------------------------
function fn = getImgBatchWrapper_pollen4DetSeg(opts)
% -------------------------------------------------------------------------
    fn = @(images,mode,scaleFactor) getBatch_dict(images, mode, scaleFactor, opts) ;
end

% -------------------------------------------------------------------------
function [im, mask, imo] = getBatch_dict(images, mode, scaleFactor, opts)
% -------------------------------------------------------------------------
    [im, mask, imo] = getImgBatch_pollen4DetSeg(images, mode, scaleFactor, opts, 'prefetch', nargout == 0) ;
end

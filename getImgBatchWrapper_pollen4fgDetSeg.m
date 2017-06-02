% return a get batch function
% -------------------------------------------------------------------------
function fn = getImgBatchWrapper_pollen4fgDetSeg(opts)
% -------------------------------------------------------------------------
    fn = @(images,mode,scaleFactor) getBatch_dict(images, mode, scaleFactor, opts) ;
end

% -------------------------------------------------------------------------
function [im, mask, imo] = getBatch_dict(images, mode, scaleFactor, opts)
% -------------------------------------------------------------------------
    [im, mask, imo] = getImgBatch_pollen4fgDetSeg(images, mode, scaleFactor, opts, 'prefetch', nargout == 0) ;
end

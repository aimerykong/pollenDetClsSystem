% return a get batch function
% -------------------------------------------------------------------------
function fn = getImgBatchWrapper_pollen4cls(opts)
% -------------------------------------------------------------------------
    fn = @(images,mode,scaleFactor) getBatch_dict(images, mode, scaleFactor, opts) ;
end

% -------------------------------------------------------------------------
function [im, mask, imo, label] = getBatch_dict(images, mode, scaleFactor, opts)
% -------------------------------------------------------------------------
    [im, mask, imo, label] = getImgBatch_pollen4cls(images, mode, scaleFactor, opts, 'prefetch', nargout == 0) ;
end

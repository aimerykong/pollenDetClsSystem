function im = drawCircle(im, y, x, radius, thickness)

if ~exist('thickness', 'var')
    thickness = 1;
end

im(y-1:y+1, x-1:x+1, 1) = 255;
im(y-1:y+1, x-1:x+1, 2) = 0;
im(y-1:y+1, x-1:x+1, 3) = 0;

value = bsxfun(@plus, ((1:size(im,2)) - x) .^ 2, ((1:size(im,1)).' - y) .^ 2);
mask  = value < (radius + thickness)^2 & value > (radius - thickness)^2;
imSingleChannel = im(:,:,1); imSingleChannel(mask) = 255; im(:,:,1) = imSingleChannel;
imSingleChannel = im(:,:,2); imSingleChannel(mask) = 0  ; im(:,:,2) = imSingleChannel;
imSingleChannel = im(:,:,3); imSingleChannel(mask) = 0  ; im(:,:,3) = imSingleChannel;



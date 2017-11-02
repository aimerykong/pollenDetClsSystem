function mask = drawDisk(mask, y, x, radius)

mask(y:y, x:x, 1) = 1;

value = bsxfun(@plus, ((1:size(mask,2)) - x) .^ 2, ((1:size(mask,1)).' - y) .^ 2);
newDisk  = value <= (radius)^2 ;
mask(newDisk)  = 1;



function maxiResp2 = imRespNMS(Resp)
% Non-Maxima Suppression over a response map.
% By comparing each response's 8 neighbors, and preserve the local maxima
% ones.
%
%
% Shu Kong
% 05/20/2015

%% Thresholding 
p = 99; % p-th percentile of the intensities in the patch as the threshold
thresh = Resp(:);
threshold = prctile(thresh, p);

%% above-below
L = ( Resp(2:end-1) > Resp(1:end-2));
R = ( Resp(2:end-1) > Resp(3:end));
T =  Resp(2:end-1) > threshold; 
maxiResp = R & L & T;
maxiResp = [0, maxiResp, 0];
maxiResp = reshape(maxiResp, [size(Resp,1), size(Resp,2)]);

%% left-right
RespTranspose = Resp';
L = ( RespTranspose(2:end-1) > RespTranspose(1:end-2));
R = ( RespTranspose(2:end-1) > RespTranspose(3:end));
T =  RespTranspose(2:end-1) > threshold;
maxiRespTranspose = R & L & T;
maxiRespTranspose = [0, maxiRespTranspose, 0];
maxiRespTranspose = reshape(maxiRespTranspose, [size(RespTranspose,1), size(RespTranspose,2)]);

%{
% shift up 1 unit
RespShift = Resp;
for i = 1:size(Resp,2)
    if mod(i,2) == 1
        RespShift(:, i) = [RespShift(2:end,i); 0];
    end
end
L = ( RespShift(2:end-1) > RespShift(1:end-2));
R = ( RespShift(2:end-1) > RespShift(3:end));
T =  RespShift(2:end-1) > treshold; 
maxiResp1 = R & L & T;
maxiResp1 = [0, maxiResp1, 0];
maxiResp1 = reshape(maxiResp1, [size(Resp,1), size(Resp,2)]);

%% shift down 1 unit
RespShift = Resp;
for i = 1:size(Resp,2)
    if mod(i,2) == 1
        RespShift(:, i) = [0; RespShift(1:end-1,i)];
    end
end
L = ( RespShift(2:end-1) > RespShift(1:end-2));
R = ( RespShift(2:end-1) > RespShift(3:end));
T =  RespShift(2:end-1) > treshold; 
maxiResp2 = R & L & T;
maxiResp2 = [0, maxiResp2, 0];
maxiResp2 = reshape(maxiResp2, [size(Resp,1), size(Resp,2)]);

%}

%% return the final NMS response map
maxiRespMask = maxiRespTranspose' & maxiResp;% & maxiResp1 & maxiResp2;

maxiResp2 = zeros(size(Resp,1), size(Resp,2));
maxiResp2(maxiRespMask==1) = Resp(maxiRespMask==1);



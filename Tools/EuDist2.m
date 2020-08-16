function D = EuDist2(X,isSqrt)
%EUDIST2 Euclidean Distance matrix of a singal view dataset
%   Input parameters:
%       X     : Original data from singal view, d*n
%       isSqrt: isSqrt=1 denotes that D is EuDist; isSqrt=0 denotes that D is EuDist.^2
%   
%   Output:
%       D     : Euclidean Distance matrix

if ~exist('isSqrt','var')
    isSqrt = 1;
end

[~,n] = size(X);
XX = sum(X.*X,1); 
XTX = X'*X;

XX = full(XX);
XTX = full(XTX);

if isSqrt 
    D = sqrt(repmat(XX,n,1)+repmat(XX',1,n)-2*XTX);
    D = real(D);
else
    D = repmat(XX,n,1)+repmat(XX',1,n)-2*XTX;
end

D = max(D,D');
D = D - diag(diag(D));
D = abs(D);
end


function L = LaplacianConstruction(X,s,sigma)
%LAPLACIANCONSTRUCTION Construct unnormalized graph Laplacian by Guassian Kernel, L = D-W
%   Constuct by Guassian Kernel, which sigma is the median of EuDists(X)
%   Input parameters:
%       X    : Original data from singal view, d*n
%       s    : Constructed with the most s nearest neighbors, default n-1
%       sigma: Parameter of Guassian Kernel
%   Output parameter:
%       L    : Unnormalized graph Laplacian

[~,n] = size(X);
if ~exist('sigma','var')
    sigma = MedianSigma(X);
end

Dist = EuDist2(X,0);
if exist('s','var') && s<n-1
    for i = 1:n
        tmp = sort(Dist(i,:));
        Dist(i,(Dist(i,:)>tmp(s+1))) = 0;
    end
    Dist = max(Dist,Dist');
end

W = exp(-Dist/(2*sigma^2));
W(W>=1) = 0;
W = max(W,W');

D = diag(sum(W));
L = D-W;
end


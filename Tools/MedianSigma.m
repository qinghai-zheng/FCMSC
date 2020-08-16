function sigma = MedianSigma(X)
%MEDIANSIGMA Calcaulate sigma by the median value
%   Input parameter:
%       X    : Data matrix, d*n
%   
%   Output parameter:
%       sigma: Median of the EuDist2(X)

[~,n] = size(X);
D = EuDist2(X);
D = reshape(D,1,n*n);
sigma = median(D);
end
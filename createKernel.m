function K = createKernel(X,Y,params)
%createKernel     constructs a kernel matrix
%
%usage
%  K = createKernel(X,Y,params)
%
%input
%  X       an N x n matrix of N data points in dimension n
%  Y       an M x n matrix of M data points in dimension n
%                       the rows of Y. If Y is given as an empty matrix (equals []), use X
%                       instead of Y.
%
% params  a struct determinining the kernel type, with fields
%           ktype   one of 'linear' ,'polynomial','gauss', identifies the kernel type
%                    further parameters need to be specified depending on kernel
%                      linear:         k(x,y) = x' A y
%                          default: A = eye(n)
%                      polynomial:     k(x,y) = (theta <x,y>)^deg
%                          default: theta = 1/sqrt(2), deg = 2
%                      gauss:          k(x,y) = exp(-1/2 |x-y|^2/sigma^2)
%                          default: sigma = 1
%                      default/other:  k(x,y) = x'y
%
% OUTPUT:
%   K       the N x M kernel matrix with entries k(X_i, Y_j),
%                         where X_i, Y_j are the rows of X,Y
%
%
%authors
%  Franz Király <f.kiraly@ucl.ac.uk>
%  Martin Kreuzer <martin.kreuzer@uni-passau.de>
%  Louis Theran <theran@math.fu-berlin.de>
%
% Copyright (c) 2014, Franz J. Kiraly, Martin Kreuzer, Louis Theran
% All rights reserved.
% see LICENSE

[N,n] = size(X);
[M,nY] = size(Y);

if n~=nY
    error('dimensions of X and Y do not match')
end


%-- initialization with default parameters start
if ~exist('params','var')
    type = 'linear';
else
    if ~isfield(params,'ktype')
        type = 'linear';
    else
        type = params.ktype;
    end
end

if ~isfield(params,'A')
    A = eye(n);
else
    A = params.A;
end

if ~isfield(params,'theta')
    theta = 1/sqrt(2);
else
    theta = params.theta;
end

if ~isfield(params,'deg')
    deg = 2;
else
    deg = params.deg;
end

if ~isfield(params,'sigma')
    sigma = 1;
else
    sigma = params.sigma;
end

%-- end initialization with default parameters


%-- construction of kernel matrix

switch type
    case 'linear'
        K = X*A*Y';
        
    case 'polynomial'
        K = (theta*X*Y' + 1).^deg;
        
    case 'gauss'
        XX = sum(X.*X, 2);
        YY = sum(Y.*Y, 2);
        D = repmat(XX, 1, M) + repmat(YY', N, 1) - 2*X*Y';
        K = exp(-D/(2*sigma^2));
        
    otherwise
        K = X*Y';
end

end
function IPCAresult = IPCA(X,IPCAparams)
%IPCA       Ideal PCA
%
%usage
%  [optional:
%  IPCAparams.Z = ...
%  IPCAparams.kernelparams = struct(...);
%  ...       ]
%
%  IPCAresult = IPCA(X,params)
%
%input
%  X          an N x n matrix of N training points
%              (N points in n-space)
%
% optional, but recommended:
%  IPCAparams    a struct with any subset of the following fields:
%
%    M               the number of feature spanning points
%                     default is n^2
%    Z               an M x n matrix which contains the basis for the feature space
%                     default is randn(M x n)
%    thresholdtype   a string which specifies the kind of thresholding in the SVD
%                     'eps': all singular values >= threshold are retained
%                     'num': the first threshold singular values are retained,
%                            but only those which are bigger-than-machineeps
%                     other/default: all bigger-than-machineeps are retained
%    threshold       the singular value threshold
%                     thresholdtype = 'eps': default is 0.01
%                     thresholdtype = 'num': default is 5
%    machineeps      the threshold for machine zero
%                     default is 1e-12
%    kernelparams    a struct or parameters for createKernel (see there for format)
%                     default is inhomogenous polynomial kernel of degree 2 with theta = 1/sqrt(2)
%                     if kernelparams is changed, M should be adapted as well
%
%
%output
% a struct of matrices:
%  U         an N x m matrix of m leading left principal kernel vectors
%  V         an M x m matrix of m leading right principal kernel vectors
%  S         an m-vector of leading singular values
%  Z         an N x n matrix which contains the basis for the feature space
%             if Z was provided in params, these are equal, otherwise Z is
%                                                         randomly generated
%  dims      a struct containing m,n,M,N
%  kmats     a struct containing pre-comnputed kernel matrices for further use
%             Kxz = k(X,Z), Kzz = K(Z,Z), Kzzinv = pinv(Kzz),
%             Kzzinvsqrt = real(sqrtm(pinv(Kzz))), Kzzsqrt = real(sqrtm(Kzz))
%             KzzV, KzzD eigendecomposition of Kzz, Kzz = KzzV*diag(KzzD)*KzzV'           
%  kernelparams   the kernel parameters which were used above
%  IPCAtime   the time needed to compute the matrices above
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


%-- initialization with default parameters start
if ~exist('IPCAparams','var')
    deg = 2;
    thresholdtype = 'eps';
    threshold = 0.01;
    M = min(n^2,N);
    Z = randn(M,n);
    kernelparams = struct('ktype','polynomial','theta',1/sqrt(2),'deg',2);
else
    
    if ~isfield(IPCAparams,'thresholdtype')
        thresholdtype = 'eps';
    else
        thresholdtype = IPCAparams.thresholdtype;
    end
    
    if ~isfield(IPCAparams,'machineeps')
        machineeps = 1e-12;
    else
        machineeps = params.machineeps;
    end
    
    if ~isfield(IPCAparams,'threshold')
        switch thresholdtype
            case 'eps'
                %threshold = 0.01;
                threshold = machineeps;
            case 'num'
                threshold = 5;
            otherwise
                thresholdtype = 'eps';
                threshold = machineeps;
        end
    else
        threshold = IPCAparams.threshold;
    end

    if ~isfield(IPCAparams,'Z')
        if ~isfield(IPCAparams,'M')
            M = min(n^2,N);
        else
            M = IPCAparams.M;
        end
        Z = randn(M,n);
    else
        [M,nZ] = size(IPCAparams.Z);
        if nZ ~= n
            error('dimensions of X and Z do not match')
        end
        Z = IPCAparams.Z;
    end
    
    if ~isfield(IPCAparams,'kernelparams')
        kernelparams = struct('ktype','polynomial','theta',1/sqrt(2),'deg',2);
    else
        kernelparams = IPCAparams.kernelparams;
    end
    
end
%-- end initialization with default parameters



%-- IPCA = thresholded SVD on the cross-kernel

Kxz = createKernel(X,Z,kernelparams);
Kzz = createKernel(Z,Z,kernelparams);

tic;

[KzzV,KzzD] = eig(Kzz);
Kzzinv = KzzV*pinv(KzzD)*KzzV';
Kzzsqrt = KzzV*sqrt(KzzD)*KzzV';
Kzzinvsqrt = KzzV*sqrt(pinv(KzzD))*KzzV';

IPCAtime = toc;


[U,S,V] = svd(Kxz*real(sqrtm(pinv(Kzz))));

switch thresholdtype
    case 'eps'
        cutoff = sum(diag(S)>= threshold);
    case 'num'
         machinecutoff = sum(diag(S)>= machineeps);
         [sSd,~] = size(diag(S));
         cutoff = min([threshold,sSd,machinecutoff]);
end

IPCAresult = struct;
IPCAresult.U = U(:,1:cutoff);
IPCAresult.V = V(:,1:cutoff);
IPCAresult.S = diag(S(1:cutoff,1:cutoff));
IPCAresult.Sinv = diag(pinv(S(1:cutoff,1:cutoff)));
IPCAresult.Z = Z;
IPCAresult.dims.m = cutoff;
IPCAresult.dims.n = n;
IPCAresult.dims.M = M;
IPCAresult.dims.N = N;
IPCAresult.kmats.Kxz = Kxz;
IPCAresult.kmats.Kzz = Kzz;
IPCAresult.kmats.Kzzinv = Kzzinv;
IPCAresult.kmats.Kzzsqrt = Kzzsqrt;
IPCAresult.kmats.Kzzinvsqrt = Kzzinvsqrt;
IPCAresult.kmats.KzzV = KzzV;
IPCAresult.kmats.KzzD = diag(KzzD);
IPCAresult.kernelparams = kernelparams;
IPCAresult.IPCAtime = IPCAtime;

end
function IPCA_evalresult = IPCA_eval(Xtest,IPCAresult,evalparams)
%IPCA_eval    Ideal PCA model evaluation
%
%usage
%  IPCAresult = IPCA(X,IPCAparams)
%  IPCAeval = IPCA_eval(Xtest,IPCAresult,evalparams)
%
%input
%  Xtest         an Ntest x n matrix of Ntest test points
%                   (Ntest points in n-space)
%  IPCAresult    output of IPCA, learnt on training data
%
% optional, but recommended:
%  evalparams    a struct which determines which outputs are computed.
%                true = output is computed; other/omitted: output is not computed
%   possible output fields:
%    Principal Components:
%     rightProj     projection onto the right (Z-type) principal components
%     rightProjN    projection onto the right (Z-type) principal components, 
%                    divided by singular vector (whitened)
%     leftProj      projection onto the left (X-type/ordinary kernel PCA) principal components
%     leftProjN     projection onto the left (X-type/ordinary kernel PCA) principal components
%                    divided by singular vector (whitened)
%    Manifold Components/Dual Principal Components:
%     rightDist     distance vector to data manifold
%     rightDistSq   squared length of distance vector
%     leftDist      distance vector to data manifold projected onto X-span
%     leftDistSq    squared length of disgance vector
%
%   default: rightProjN = true, rightDist = true, all others are not computed
%
%
%output
% a struct of matrices IPCA_evalresult 
% Which of these are computed is determined by evalparams.
%  fields have the same name as evalparams, but contain matrices
% default: only rightProjN, rightDist are computed
%
%    Principal Components:
%     rightProj     an Ntest x m matrix of right principal projections
%     rightProjN    an Ntest x m matrix of whitened right principal projections
%     leftProj      an Ntest x m matrix of left principal projections
%     leftProjN     an Ntest x m matrix of whitened left principal projections
%    Manifold Components/Dual Principal Components:
%     rightDist     an Ntest x m matrix of residual right projection vectors
%     rightDistSq   an Ntest-vector of their squared lengths
%     leftDist      an Ntest x m matrix of residual left projection vectors
%     leftDistSq    an Ntest-vector of their squared lengths
%
%    [outputname]_time contains the total time needed for IPCA and evaluation
%
%authors
%  Franz Király <f.kiraly@ucl.ac.uk>
%  Martin Kreuzer <martin.kreuzer@uni-passau.de>
%  Louis Theran <theran@math.fu-berlin.de>
%
% Copyright (c) 2014, Franz J. Kiraly, Martin Kreuzer, Louis Theran
% All rights reserved.
% see LICENSE

[~,n] = size(Xtest);

if n ~= IPCAresult.dims.n
    error('dimensions of Xtest and X do not match')
end


%-- initialization with default parameters start
if ~exist('evalparams','var')
   rightProj = false;
   rightProjN = true;
   leftProj = false;
   leftProjN = false;
   rightDist = true;
   leftDist = false;
else
    
    if ~isfield(evalparams,'rightProj')
        rightProj = false;
    else
        if ~islogical(evalparams.rightProj)
            rightProj = false;
        else
            rightProj = evalparams.rightProj;
        end        
    end
    
    if ~isfield(evalparams,'rightProjN')
        rightProjN = false;
    else
        if ~islogical(evalparams.rightProjN)
            rightProjN = false;
        else
            rightProjN = evalparams.rightProjN;
        end        
    end

    if ~isfield(evalparams,'leftProj')
        leftProj = false;
    else
        if ~islogical(evalparams.leftProj)
            leftProj = false;
        else
            leftProj = evalparams.leftProj;
        end        
    end
        
    if ~isfield(evalparams,'leftProjN')
        leftProjN = false;
    else
        if ~islogical(evalparams.leftProjN)
            leftProjN = false;
        else
            leftProjN = evalparams.leftProjN;
        end        
    end
    
    if ~isfield(evalparams,'rightDist')
        rightDist = false;
    else
        if ~islogical(evalparams.rightDist)
            rightDist = false;
        else
            rightDist = evalparams.rightDist;
        end        
    end
    
    if ~isfield(evalparams,'rightDistSq')
        rightDistSq = false;
    else
        if ~islogical(evalparams.rightDistSq)
            rightDistSq = false;
        else
            rightDistSq = evalparams.rightDistSq;
        end        
    end
        
    if ~isfield(evalparams,'leftDist')
        leftDist = false;
    else
        if ~islogical(evalparams.leftDist)
            leftDist = false;
        else
            leftDist = evalparams.leftDist;
        end        
    end
    
    if ~isfield(evalparams,'leftDistSq')
        leftDistSq = false;
    else
        if ~islogical(evalparams.leftDistSq)
            leftDistSq = false;
        else
            leftDistSq = evalparams.leftDistSq;
        end        
    end
        
end

if rightProj + rightProjN + leftProj + leftProjN + leftDist + leftDistSq + rightDist + rightDistSq == 0
   error('no outputs have been requested in evalparams, aborting computation'); 
end
%-- end initialization with default parameters



%-- load the model estimation results from IPCA
kernelparams = IPCAresult.kernelparams;

IPCAtime = IPCAresult.IPCAtime;

Z = IPCAresult.Z;

m = IPCAresult.dims.m;
M = IPCAresult.dims.M;
N = IPCAresult.dims.N;

U = IPCAresult.U;
V = IPCAresult.V;
S = diag(IPCAresult.S);
Sinv = diag(IPCAresult.Sinv);

Kxz = IPCAresult.kmats.Kxz;
Kzz = IPCAresult.kmats.Kzz;
Kzzinv = IPCAresult.kmats.Kzzinv;
Kzzinvsqrt = IPCAresult.kmats.Kzzinvsqrt;


%-- eval.IPCA
IPCA_evalresult = struct;

Kxtz = createKernel(Xtest,Z,kernelparams);
%Kxtx = createKernel(Xtest,X,kernelparams);

if rightProj
    tic;
    IPCA_evalresult.rightProj = Kxtz*V;
    IPCA_evalresult.rightProj_time = toc + IPCAtime;
end

if rightProjN
    if rightProj
        tic;
        IPCA_evalresult.rightProjN = IPCA_evalresult.rightProj*Sinv;
        IPCA_evalresult.rightProjN_time = IPCA_evalresult.rightProj_time + toc + IPCAtime;
    else
        tic;
        IPCA_evalresult.rightProjN = Kxtz*V*Sinv;
        IPCA_evalresult.rightProjN_time = toc + IPCAtime;
    end
end

if leftProj
    tic;
    IPCA_evalresult.leftProj = Kxtz*Kzzinv*Kxz'*U;
    IPCA_evalresult.leftProj_time = toc + IPCAtime;
end

if leftProjN
    if leftProj
        tic;
        IPCA_evalresult.leftProjN = IPCA_evalresult.leftProj*Sinv';
        IPCA_evalresult.leftProjN_time = IPCA_evalresult.leftProj_time + toc + IPCAtime;
    else
        tic;
        IPCA_evalresult.leftProjN = Kxtz*Kzzinv*Kxz'*U*Sinv';
        %IPCA_evalresult.leftProjN = Kxtx*U*Sinv';
        IPCA_evalresult.leftProjN_time = toc + IPCAtime;
    end
end

if rightDist
    tic;
    PV = eye(M) - V*V';
    rd = Kxtz*Kzzinvsqrt*PV;
    IPCA_evalresult.rightDist = rd;
    IPCA_evalresult.rightDist_time = toc;
end

if rightDistSq
    if rightDist
        tic;
        rd = IPCA_evalresult.rightDist;
        rdsq = sum(rd.^2,2);
        IPCA_evalresult.rightDistSq = rdsq;
        IPCA_evalresult.rightDistSq_time = toc + IPCA_evalresult.rightDist_time;
    else
        tic;
        PV = eye(M) - V*V';
        rd = Kxtz*Kzzinvsqrt*PV;
        rdsq = sum(rd.^2,2);
        IPCA_evalresult.rightDistSq = rdsq;
        IPCA_evalresult.rightDistSq_time = toc;
    end
end


if leftDist
    tic;
    PU = eye(N)-U*U';
    IPCA_evalresult.leftDist = Kxtz*Kzzinv*Kxz'*PU;
    IPCA_evalresult.leftDist_time = toc + IPCAtime;
end

if leftDistSq
    if leftDist
        tic;
        IPCA_evalresult.leftDistSq = IPCA_evalresult.leftDist.*IPCA_evalresult.leftDist';
        IPCA_evalresult.leftDistSq_time = IPCA_evalresult.leftDist_time + toc + IPCAtime;
    else
        tic;
    	PU = eye(N)-U*U';
        IPCA_evalresult.leftDistSq = Kxtz*Kzzinv*Kxz'*PU*Kxz*Kzzinv*Kxtz;
        IPCA_evalresult.leftDistSq_time = toc + IPCAtime;
    end
end

end
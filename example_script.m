
% example template for the usage of IPCA
% needs to be completed before use


% STEP 1 - load data.

data_to_learn_features = %read in data from which features are learnt
data_to_evaluate_features = %read in data for which to extract features, these canbut need not be the same


% STEP 2 - learn IPCA features.

IPCAparams = struct;
IPCAparams.Z = %initialize the feature spanning points, or remove this line for random initialization

IPCAparams.kernelparams = struct(%choose the type of kernel used, see createKernel


IPCAresult = IPCA(data_to_learn_features,IPCAparams);


% STEP 3 - evaluate the IPCA features

evalparams = struct (%specify which features should be evaluated

IPCA_evalresult = IPCA_eval(data_to_evaluate_features,IPCAresult,evalparams);


% STEP 4 - optional - use the evaluated features in further learning

furtherlearningalgorithm(IPCA_evalresult.oneoftheoutputmatrices)






% This scripts sets parameters for parameter estimation and 
% hyperparameter search

% set bounds for the uniform prior:
% proton density [a.u.]:
minPd = 0;
maxPd = 1e5;
% T1 [s]:
minT1 = 0;
maxT1 = 10;

parameterMax = [maxPd, maxT1];
parameterMin = [minPd, minT1];

% Typical parameters [Pd, T1] for algorithm initialization.
typicalParameters = [10000, 1];

%kappa_search = -0.5:0.01:0.99; % kappa grid search window
kappa_search = 0.0:0.1:0.3;
nSamples = 128; % number of samples per MCMC walkers
nWalkers = 10; % Number of MCMC walkers
nThinning = 1; % thinning 
inner_thinning = 8; % inner thinning
step_size=4;
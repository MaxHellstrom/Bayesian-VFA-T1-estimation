function [WAIC, X, time_struct] = calculate_WAIC_curve(Y, B1corr, typicalParameters, parameterMin, parameterMax, mask, kappa_search, TR, FAs, nWalkers, inner_thinning, step_size)


%% Construct the estimator and set properties
MCMC = MCMCSpatial2DModel(@(x,z)(SPGRModel(x, z, TR, FAs)), Y, B1corr, typicalParameters, mask, 0);

MCMC.TVWeight = 1;%weightsFromGradients(Y, mask);
MCMC.parameterMin = parameterMin;
MCMC.parameterMax = parameterMax;
MCMC.parameterNames = {'PD', 'T1'};
MCMC.convergenceTestWindow = 100;
MCMC.useSpatialModel = true;
MCMC.plotFigure = figure(); clf
MCMC.convergenceTestFraction = 0.5;
MCMC.thinning = 1;
MCMC.plotConvergence = figure(); clf
MCMC.convergenceThreshold = 0.99;
MCMC.verbose = true;

% Optimal settings from an extensive search.
MCMC.innerThinning = inner_thinning;
MCMC.setGWStepSize(step_size);
MCMC.setNoOfWalkers(nWalkers);

%% Calculate the WAIC values

time_struct = {};
time_struct.burn_in = zeros(1, numel(kappa_search));
time_struct.samples = zeros(1, numel(kappa_search));

X = zeros([size(B1corr),2,numel(kappa_search)]);
for ii = 1:numel(kappa_search)

    MCMC.kappa = kappa_search(ii);
    
    tic;
    MCMC.burnIn();
    time_struct.burn_in(ii) = toc;
    
    tic;
    [tmp,~,~, WAIC(ii)] = MCMC.sample(100); %#ok<AGROW>
    time_struct.samples(ii) = toc;
    X(:,:,:,ii) = mean(tmp(:,:,:,:),4); % Store the mean image.
end


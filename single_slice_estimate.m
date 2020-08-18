function [X, Lambda, Sigma, waic, ess, thinning, timer] = ...
    single_slice_estimate(model, Y, FA, TR, mask, B1_corr, kappa, ...
    typicalParameters, parameterMin, parameterMax, nThinning, nWalkers, ...
    nSamples, inner_thinning, step_size)

    %% Construct the estimator and set properties
    MCMC = MCMCSpatial2DModel(@(x,z)(SPGRModel(x, z, TR, FA)), Y, B1_corr, typicalParameters, mask, kappa);

    MCMC.parameterMin = parameterMin;
    MCMC.parameterMax = parameterMax;
    MCMC.parameterNames = {'PD', 'T1'};
    MCMC.verbose = true;

if model == "B_unif" || model == "B_TV"

    MCMC.TVWeight = 1;
    MCMC.convergenceTestWindow = 100;
    MCMC.useSpatialModel = (model == "B_TV");
    MCMC.plotFigure = figure('Name', 'current MCMC sample'); clf
    MCMC.convergenceTestFraction = 0.5;
    MCMC.thinning = nThinning;
    MCMC.plotConvergence = figure('Name', 'Burn-in calculation'); clf
    MCMC.convergenceThreshold = 0.99;

    % Optimal settings from an extensive search.
    MCMC.innerThinning = inner_thinning;
    MCMC.setGWStepSize(step_size);
    MCMC.setNoOfWalkers(nWalkers);
    
    % MCMC estimate
    tic
    MCMC.burnIn();
    timer.burn_in=toc;
    tic
    [X, Lambda, Sigma, waic] = MCMC.sample(nSamples);
    timer.samples=toc;
    X = single(X); % Save memory
    [ess, thinning] = MCMC.ESS(X);
end

if model == "ML"
    
    Lambda = nan; Sigma = nan; waic = nan; ess = nan; thinning=nan;
    
    tic;
    X = MCMC.MLfit();
    timer.samples = toc;
    X = single(X); % Save memory
end
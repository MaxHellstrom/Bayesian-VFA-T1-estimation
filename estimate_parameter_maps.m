function map = estimate_parameter_maps(...
    model, Y, FA, TR, mask, B1_corr, kappa, typicalParameters, ...
    parameterMin, parameterMax, nThinning, nWalkers, nSamples, ...
    inner_thinning, step_size)



% Create struct to store data and results
map = {};
timer = {};

% Store model parameters
map.model_params = {};
map.model_params.name = model; % model name (ML/B_unif/B_TV)
map.model_params.Y = Y; % store input VFA data;
map.model_params.FA = FA; % flip angle
map.model_params.TR = TR; % repetition time
map.model_params.mask = mask; % mask
map.model_params.B1_corr = B1_corr; % B1 correction

% store sampler parameters (for B_unif and B_TV)
if model == "B_unif" || model == "B_TV"
    map.sampler_params = {};
    map.sampler_params.kappa = kappa;
    map.sampler_params.nWalkers = nWalkers;
    map.sampler_params.nSamples = nSamples;
    map.sampler_params.nThinning = nThinning;
end

% conduct tissue parameter estimation
[X, Lambda, Sigma, waic, ess, thinning, timer] = single_slice_estimate(...
    model, Y, FA, TR, mask, B1_corr, kappa, typicalParameters, ...
    parameterMin, parameterMax, nThinning, nWalkers, nSamples, ...
    inner_thinning, step_size);

% Store the results
map.results = {};
map.results.PD = X(:,:,1,:);
map.results.T1 = X(:,:,2,:);
map.results.timer = timer;


if model == "B_unif" || model == "B_TV"
    map.results.Lambda(:,:) = reshape(Lambda,[size(Lambda,1),1,size(Lambda,2)*size(Lambda,3)]);
    map.results.Sigma(:,:) = reshape(Sigma,[size(Sigma,1),1,size(Sigma,2)*size(Sigma,3)]);
    map.results.waic{1} = waic;
    map.results.ESS_T1(:,:) = ess(:,:,2);
    map.results.ESS_PD(:,:) = ess(:,:,1);
    map.results.required_thinning_T1 = thinning(:,:,2);
    map.results.required_thinning_PD = thinning(:,:,1);
end
end

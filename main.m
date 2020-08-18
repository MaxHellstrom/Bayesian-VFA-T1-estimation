%% Set paths, settings, and load input data

% Paths
addpath('estimators')
addpath('plotting')
addpath('models')

data_path = "data/";
results_path = 'estimation results/';
data_name = 'synthetic_data.mat';

% Load dataset
load(data_path + data_name);

% Set model parameters
set_model_parameters

plot_input_data(FA, Y, mask, ref, B1_corr)

%% Conduct hyperparameter search for the models only hyperparameter, kappa.
% Note: This is already conducted for the synthetic dataset provided in the
% data folder. the Kappa value is loadd when using plot_input_data(FA, Y, mask, ref, B1_corr)

conduct_kappa_search = true;

if conduct_kappa_search

    [WAIC, X, time_struct] = calculate_WAIC_curve(Y, B1_corr, typicalParameters, parameterMin, parameterMax, mask, kappa_search, TR, FA, nWalkers, inner_thinning, step_size);
    
    for ii = 1:numel(WAIC)
        w(ii) = WAIC(ii).computed_waic;
    end
    % Save results
    save('WAIC_results.mat', 'kappa_search', 'w');

    
end
%% Load hyperparameter search result

load('example_WAIC_results.mat')

% Analyse kappa search to extract min value
WAIC_smooth = movmean(w, 30);
[min_3, ind_3] = min(WAIC_smooth);
kappa = round(kappa_search(ind_3),2);

figure()
plot(kappa_search, w,'.')
hold on
plot(kappa_search, WAIC_smooth,'k')
plot(kappa, WAIC_smooth(ind_3),'r*')
legend('WAIC', 'WAIC smoothed',['kappa_{min} = ', num2str(kappa)])



%% Conduct parameter estimation

% estimate using maximum likelihood estimator
ML = estimate_parameter_maps("ML", Y, FA, TR, mask, B1_corr, kappa, typicalParameters, parameterMin, parameterMax, nThinning, nWalkers, nSamples, inner_thinning, step_size);
% estimate using the B_unif estimator
B_unif = estimate_parameter_maps("B_unif", Y, FA, TR, mask, B1_corr, kappa, typicalParameters, parameterMin, parameterMax, nThinning, nWalkers, nSamples, inner_thinning, step_size);
% estimate using the B_TV estimator
B_TV = estimate_parameter_maps("B_TV", Y, FA, TR, mask, B1_corr, kappa, typicalParameters, parameterMin, parameterMax, nThinning, nWalkers, nSamples, inner_thinning, step_size);

% Save results
save([results_path,'example_1.mat'], 'ML', 'B_unif','B_TV', 'ref');



%% Plot and compare estimation results
results_path = 'estimation results/';
results_name = 'example_1.mat';

load([results_path, results_name]);
plot_estimation_results(ML, B_unif, B_TV, ref, [0, 8], [-25, 25], -90)
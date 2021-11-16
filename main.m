%% Set paths, settings, and load input data

% Paths
addpath('estimators')
addpath('plotting')
addpath('models')

% load to dataset
dataset_path = "data/";
dataset_name = 'synthetic_data.mat';

kappa_results_path = 'kappa_search_results/';
kappa_results_name= 'kappa_results.mat';

estimation_results_path = 'estimation_results/';
estimation_results_name = 'estimation_results';



% Load dataset
disp('loading dataset...')
load(dataset_path + dataset_name);

% Set model parameters
set_model_parameters


if plot_input
    disp('plotting dataset...')
    plot_input_data(FA, Y, mask, ref, B1_corr)
end




%% Conduct hyperparameter search for the models only hyperparameter, kappa.

if conduct_kappa_search
    disp('conducting kappa_search...')
    
    [WAIC, X, time_struct] = calculate_WAIC_curve(Y, B1_corr, typicalParameters, parameterMin, parameterMax, mask, kappa_search, TR, FA, nWalkers, inner_thinning, step_size);

    for ii = 1:numel(WAIC)
        w(ii) = WAIC(ii).computed_waic;
    end
    
    % Save results
    save([kappa_results_path, kappa_results_name], 'kappa_search', 'w');
else
    disp('Kappa search avoided...')
    disp('loading precalculated kappa_min... ');
    load([kappa_results_path, 'kappa_results_example'])
end 


WAIC_smooth = movmean(w, 30);
[min_3, ind_3] = min(WAIC_smooth);
kappa = round(kappa_search(ind_3),2);

if plot_kappa_search_results
    figure('Name', 'WAIC results')
    plot(kappa_search, w,'.')
    hold on
    plot(kappa_search, WAIC_smooth,'k')
    plot(kappa, WAIC_smooth(ind_3),'r*')
    legend('WAIC', 'WAIC smoothed',['kappa_{min} = ', num2str(kappa)])
end

disp(['using kappa = ', num2str(kappa)])


%% Conduct parameter estimation

if conduct_parameter_estimation
    disp('conducting parameter estimation with B_TV, B_unif and ML')
    
    % estimate using the B_TV estimator
    B_TV = estimate_parameter_maps("B_TV", Y, FA, TR, mask, B1_corr, kappa, typicalParameters, parameterMin, parameterMax, nThinning, nWalkers, nSamples, inner_thinning, step_size);
    
    % estimate using the B_unif estimator
    B_unif = estimate_parameter_maps("B_unif", Y, FA, TR, mask, B1_corr, kappa, typicalParameters, parameterMin, parameterMax, nThinning, nWalkers, nSamples, inner_thinning, step_size);

    % estimate using maximum likelihood estimator
    ML = estimate_parameter_maps("ML", Y, FA, TR, mask, B1_corr, kappa, typicalParameters, parameterMin, parameterMax, nThinning, nWalkers, nSamples, inner_thinning, step_size);
    
    % Save results
    save([estimation_results_path, estimation_results_name], 'ML', 'B_unif','B_TV', 'ref');

else
    disp('no parameter estimation conducted')
    disp('loading precalculated estimation data')
    load([estimation_results_path, 'estimation_results_example']);
end



%% Plot and compare estimation results
plot_estimation_results(ML, B_unif, B_TV, ref, [0, 8], [-25, 25], -90)

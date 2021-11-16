% WAIC - Widely Applicable Information Criterion
%
% Parameters
% ----------
% logp : N-by-n float matrix
%     The log-likelihood values for each sample. N is the number of MCMC
%     samples, and n is the number of data samples.
%
% method : 1 or 2
%     Whether to compute the estimated number of parameters using the
%     variance (2) or the original method (1).
% normalize : bool
%     Whether or not to normalize the WAIC estimate. This is done by
%     dividing by n, i.e. the mean WAIC is returned in that case. Default
%     is false.

function [computed_waic, p_waic, waic_se, ok] = waic(logp, method, normalize)

    if nargin < 2
        method = 2;
    end
    if nargin < 3
        normalize = false;
    end

    lppd_i = logmeanexp(logp,1); %log(mean(exp(logp), 1));  % Implement log-sum-exp instead!!

    if (method == 1)
        vars_lpd_i = 2 * (lppd_i - mean(logp, 1));
    else
        vars_lpd_i = var(logp, 1, 1);
    end

    ok = ~any(vars_lpd_i > 0.4);

    waic_i = -2 * (lppd_i - vars_lpd_i);

    % TODO!!!
    if length(waic_i) > 1
        waic_se = sqrt(length(waic_i) * var(waic_i));
    else
        waic_se = NaN;
    end

    % Effective number of parameters
    p_waic = sum(vars_lpd_i);

    % Final WAIC value
    computed_waic = sum(waic_i);

    if normalize
        computed_waic = computed_waic / length(waic_i);
    end

end


function lme = logmeanexp(x,dim)
 n = numel(x);
 x_star = max(x,[],dim);
 dx = bsxfun(@plus,x,-x_star);
 lse = log(sum(exp(dx), dim));
 lme = x_star - log(n) + lse;
end
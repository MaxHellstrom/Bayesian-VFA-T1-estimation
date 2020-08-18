classdef MCMCSpatial2DModel < handle
    % This class enables MCMC sampling from the hidden parameters in a 2D spatial
    % model. Uses a Gibbs sampling algorithm combined with an ensamble
    % sampler [1].
    %
    % [1] Goodman, J. & Weare, J. Ensemble samplers with affine invariance. Commun. Appl. Math. Comput. Sci. 5, 65???80 (2010)

    properties (SetAccess = private)
        % Function handle to the model at each spatial position. Must be on
        % the form. Yh = model(theta) where theta is the parameters to the model
        % which are shaped like [N, p], where N is number of spatial positions and p
        % is the number of parameters per site. The output must be on the
        % form [N, k] where k is the number of images in the data.
        model;
        
        % The observed data. This is automatically populated by the
        % constructor. Is shaped like [N, k].
        data;
        
        % A mask telling where estimation should be performed. Use to avoid
        % spending time in background regions. Logical image on the form
        % [Nx, Ny]. 
        mask;
        
        % The number of pixels (sites).
        V;
        
        % The number of images in the data
        M;
        
        % Spatial parameters that are assumed to be known
        fixedParams;

    end
    
    
    properties (Access = public)
        % Set to true if verbose output is desired.
        verbose = true;
        
        % The hyperparameter of the statistical model
        kappa;
        
        % Set to false if no spatial model should be used. If false pixel
        % by pixel estimation will be performed.
        useSpatialModel = true;
        
        % The amount of thinning at each spatial position when using the
        % GW algorithm. 
        innerThinning = 8;
        
        % The thinning used 
        thinning = 1;
        
        % The thinning can be calculated using data from the burn-in. If
        % set to true this is performed.
        autoThinning = true;
        
        % Will be equal to true when the chain has converged. I.e. after
        % burn-in has been performed.
        converged = false;
        
        % In burn-in the 'convergenceTestCallback' function will use this
        % threshold to judge if the chains has converged.
        convergenceThreshold = 0.99;
        
        % Callback function to test if the chain has converged. Must be on
        % the format: [tf, value] = testcallback(p1, p2, MCMC). p1 and p2
        % should be shaped like [N,p,m]. Where N is the number of spaital
        % sites, p is the number of parameters per site and m is the number of
        % samples. Default: the static method
        % 'testBayesianDistributionConvergence'. The method
        % testHistogramConvergence is also available. Not recommended though.
        convergenceTestCallback;

        % To judge if the chains have converged samples from a window at
        % the end of he chain is used. This property gives the size of
        % theis window.
        convergenceTestWindow = 100;
        
        % How often to perform the convergence test. If 10 (default) it is performed
        % every tenth iteration.
        convergenceTestSkipFrequency = 10;
        
        % The fraction of pixels used in the convergance test.
        convergenceTestFraction = 1;
        
        % Internal use.
        convergenceTestMask;
        
        % What variables to test for convergance. Leave empty for all (default).
        % Use the idex of the variables to specify. E.g. [1 2]
        convergenceTestVariables = [];
        
        % The algorithm must start from a noisy initial guess. The initial 
        % guess mean is set by fitting the model to the mean of all spatial
        % sites. The randomness is introduced by adding a fraction of the
        % mean randomly to each site parameter. 
        initRandomnessFraction = 0.1;
        
        % The algorithm uses a weighted total variation to make the model more
        % general. If the input data is on the form [Nx,Ny,Nd] then TVWeight 
        % must be on the form [Nx,Ny] or a scalar in which case it is not used. 
        TVWeight = 1; 
        
        % Callback used to view the progress. Must be of the same format as
        % MCMCSpatial2DModel.defaultViewCallback.
        viewCallback;
        
        % Maximum parameter values on the format [Np, 1]
        parameterMax;
        
        % Minimum parameter values on the format [Np, 1]
        parameterMin;
        
        % Parameter names
        parameterNames;
        
        % A vector of typical parameters for a single spatial position
        typicalParameters;
        
        % The figure in which to plot
        plotFigure = []
        
        % The figure into which the convergence should be plotted. No plots
        % if not available.
        plotConvergence = [];
        
    end
    
    properties (SetAccess = private, GetAccess = public)
        % The stepsize of the Goodman & Weare algorithm. Default 2. Use the
        % setGWStepSize to modify this property.
        stepSize = 2;
        
        % The last acceptence ratio of the GW algorithm should be around 0.23-0.5 ideally.
        GWAcceptenceRatio;
        
        % The number of walkers used
        nWalkers = 10;
    end
    
    properties (Access = private)
        % The number of disjoint subsets
        nSubsets = 9;
        
        % The data reshaped as cell array of [Npixels_persubset, Nimages]
        y;
        
        % Spatial parameters that are assumed to be known
        fixedParameters;
        
        % A description of the neighbourhood around a pixel.
        neighbourStruct;
        
        % The TVWeight flattened out for calculations. Has the shape [N,1].
        weight = [];
        
        % The parameters of the model to be sampled
        theta; % [N, Nparams, nWalkers]
        lambda; % [Nparams, nWalkers]
        sigma; % [1, nWalkers]
        
        % Ensambe MCMC samplers
        mcSpatial;
    end
    
    
    methods (Access = public)
        function obj = MCMCSpatial2DModel(model, data, fixedParams, typicalParameters, mask, kappa)
            % Constructs a MCMCSpatial2DModel.
            %
            % Input parameters:
            % model             -   Function handle to the model at each spatial position. Must be on
            %                       the form. Yh = model(theta) where theta is the parameters to the model
            %                       which are shaped like [N, p], where N is number of spatial positions and p
            %                       is the number of parameters per site. The output must be on the
            %                       form [N, k] where k is the number of images in the data.
            % data              -   Observed data on the format [nx, ny,
            %                       k]. Where (nx, ny) is the size of the images in the data and k is the number of images. 
            % fixedParams       -   Spatial parameters that are assumed to
            %                       be know. [nx, ny, nf] 
            % typicalParameters -   A typical value on the parameters in
            %                       the model (for a single position). Used as initial starting position for the algorithm.
            % mask              -   A mask (nx,ny) of logicals telling where to estimate the parameters.
            % kappa             -   The regularization hyper-parameter. 
            %
            % Output parameters:
            % obj               -   The constructed object.
            %
            
            
            % Set public info
            nParameters = numel(typicalParameters);
            obj.typicalParameters = typicalParameters;
            obj.model   = model;
            obj.data    = data;
            obj.fixedParams = fixedParams;
            obj.mask    = mask;
            obj.parameterMax = inf(nParameters,1);
            obj.parameterMin = -inf(nParameters,1);
            obj.kappa = kappa;
            obj.V = sum(mask(:));
            obj.M = size(data, 3);
            obj.convergenceTestCallback = @MCMCSpatial2DModel.testBayesianDistributionConvergence;
            obj.viewCallback = @MCMCSpatial2DModel.defaultViewCallback;
            
            % Split the pixels to be estimated into 9 sets. To ensure independence in
            % the Gibbs sampler.
            pixelSetMasks = cell(1,9);
            n1 = size(mask,1);
            n2 = size(mask,2);
            ii = 0;
            for i1 = 1:3
                for i2 = 1:3
                    ii = ii + 1;
                    pixelSetMasks{ii} = false(size(mask));
                    pixelSetMasks{ii}(i1:3:n1,i2:3:n2) = true;
                    pixelSetMasks{ii} = pixelSetMasks{ii} & mask;
                end
            end
            
            % Create the neighbourhood structure
            obj.neighbourStruct = [];
            for ii = 1:numel(pixelSetMasks)
                obj.neighbourStruct = [obj.neighbourStruct, obj.getNeighbourInfo(pixelSetMasks{ii})]; 
            end

            % Organize the data and fixed parameters into the 9 subsets.
            obj.y = cell(1,obj.nSubsets);
            obj.fixedParameters = cell(1,obj.nSubsets);
            for ii = 1:numel(obj.y)
                pixels = obj.neighbourStruct(ii).pixelSetIndex;
                n = numel(pixels);
                obj.y{ii} = zeros(n,obj.M);
                
                if (~isempty(fixedParams))
                   obj.fixedParameters{ii} = zeros(n, size(fixedParams,3)); 
                   for m = 1:size(fixedParams,3)
                       tmp = fixedParams(:,:,m);
                       tmp = tmp(obj.mask);
                       obj.fixedParameters{ii}(:,m) = tmp(pixels);
                   end
                end
                   
                for m = 1:obj.M
                   tmp = obj.data(:,:,m);
                   tmp = tmp(obj.mask);
                   obj.y{ii}(:,m) = tmp(pixels);
                end
            end
            
            % Create initial guess. Will set theta, lambda , sigma.
            obj.initialGuess();
            
            % Setup a sampler per subset
            obj.mcSpatial = cell(1,obj.nSubsets);
            for ii = 1:numel(obj.mcSpatial)
               obj.mcSpatial{ii} = GWMCMC(@(x,wix)(obj.logPFun(x,wix,ii)),obj.theta(obj.neighbourStruct(ii).pixelSetIndex,:,:));
               obj.mcSpatial{ii}.StepSize = obj.stepSize;
            end  
            
            % Set default parameter names
            obj.parameterNames = cell(1,nParameters);
            for ii = 1:nParameters
               obj.parameterNames{ii} = ['P',num2str(ii)]; 
            end
            
        end
        
        function X = MLfit(obj)
           % Perform a pixelwise fit to the data
           Y = zeros(obj.V, obj.M);
           Z = zeros(obj.V, size(obj.fixedParams,3));
           for m = 1:obj.M
                tmp = obj.data(:,:,m);
                Y(:,m) = tmp(obj.mask);
           end
           
           if (isempty(obj.fixedParams))
               cost = @(x, v) sum( (obj.model(x, []) - Y(v,:)).^2 );
           else
               for m = 1:size(obj.fixedParams,3)
                    tmp = obj.fixedParams(:,:,m);
                    Z(:,m) = tmp(obj.mask);
               end

                % Cost function
                cost = @(x, v) sum( (obj.model(x, Z(v,:)) - Y(v,:)).^2 ); 
           end
            % Output
            np = numel(obj.typicalParameters(:));
            xx = zeros(obj.V, np);
            
            % Set optimization structure
            
            % Estimate fmincon(fun,x0,A,b,Aeq,beq,lb,ub)
            options = optimoptions(@fmincon,'display','off', 'TypicalX', obj.typicalParameters(:)');
            h = waitbar(0,'Perform ML-fit');
            for v = 1:obj.V
                xx(v,:) = fmincon(@(x)cost(x,v), obj.typicalParameters(:)',[],[],[],[],obj.parameterMin, obj.parameterMax, [], options);
                if (mod(v,100)==0)
                   waitbar(v/obj.V); 
                end
            end
            close(h);
            
            % Rearange the results
            X = zeros([size(obj.mask), np]);
            for ii = 1:np
               tmp = zeros(size(obj.mask));
               tmp(obj.mask) = xx(:,ii);
               X(:,:,ii) = tmp;
            end
        end
        
        function setGWStepSize(obj,ss)
            % Set the stepsize of the GW algorithm. Must be > 1.
            %
            % Input:
            % obj   - A MCMCSpatial2DModel object.
            % ss    - The step size > 1.
            obj.stepSize = ss;
            for ii = 1:numel(obj.mcSpatial)
               obj.mcSpatial{ii}.StepSize = ss;
            end  
        end
        
        function setNoOfWalkers(obj, nW)
            % Set the number of walkers. Note that this will reset the
            % state of the Markov Chain.
            %
            % Input:
            % obj   - A MCMCSpatial2DModel object.
            % nW    - The number of walkers.
            
            
            obj.nWalkers = nW;
            % Create initial guess. Will set theta, lambda , sigma.
            obj.initialGuess();
            
            % Setup a sampler per subset
            obj.mcSpatial = cell(1,obj.nSubsets);
            for ii = 1:numel(obj.mcSpatial)
               obj.mcSpatial{ii} = GWMCMC(@(x,wix)(obj.logPFun(x,wix,ii)),obj.theta(obj.neighbourStruct(ii).pixelSetIndex,:,:));
               obj.mcSpatial{ii}.StepSize = obj.stepSize;
            end     
        end
        
        function [hasConverged, WAIC, X, Lambda, Sigma] = burnIn(obj, maxNoOfIterations)
            % Perform burn-in until covergence or maxNoOfIterations has
            % been reached. 
            %
            % Input parameters:
            % obj               -   A MCMCSpatial2DModel object.
            % maxNoOfIterations -   Maximum samples used in burn-in.
            %                       Optional parameter with default value = inf.
            %
            %
            % Output parameters:
            % hasConverged      -   True if the burn-in phase has convered.
            % WAIC              -   Information about the Widely Applicable
            %                       Information Criteria for the converged samples obtained
            %                       during burn-in.
            % X                 -   Samples obtained during burn-in
            %                       organised as [nx, ny, p, w, ns] where p
            %                       is the number of parameters per site, w
            %                       is the number of walkers and ns is the
            %                       number of samples. 
            % Lambda            -   The regularization parameters organized as 
            %                       [2, w, ns]
            % Sigma             -   The standard deviation organized as [1, w, ns]
            %
            
            if (nargin < 2)
                maxNoOfIterations = inf;
            end
            
            % Setup the weights if not already done.
            if (isempty(obj.weight))
               obj.setupWeights();
            end
            
            if (obj.convergenceTestFraction < 1)
                nvars = sum(obj.mask(:));
               obj.convergenceTestMask = rand(nvars,1) < obj.convergenceTestFraction;
            end
            

            % Preallocate
            nParams = numel(obj.parameterMax);
            allocationChunk = obj.convergenceTestWindow*5;
            Theta = zeros([size(obj.theta),obj.convergenceTestWindow]);
            s = zeros(1,obj.nWalkers,obj.convergenceTestWindow);
            l = zeros(2,obj.nWalkers,obj.convergenceTestWindow);
            
            if (nargout > 2)
               % Allocate X, Lambda, Sigma 
                X = zeros(size(obj.mask,1),size(obj.mask,2), nParams, obj.nWalkers, allocationChunk);
                Sigma = zeros(1, obj.nWalkers, allocationChunk);
                Lambda = zeros(nParams, obj.nWalkers, allocationChunk);
                allocatedSize = allocationChunk;
            end
            
            iIterations = 0;
            obj.converged = false;
            ringBufferIndex = 0;
            while (~obj.converged && (iIterations < maxNoOfIterations))
                   for ii = 1:obj.thinning
                     obj.gibbsSampler();
                   end
                   ringBufferIndex = mod(ringBufferIndex, obj.convergenceTestWindow)+1;
                   Theta(:,:,:,ringBufferIndex)     = obj.theta;
                   s(:,:,ringBufferIndex) = obj.sigma;
                   l(:,:,ringBufferIndex) = obj.lambda;
                   
                   iIterations = iIterations + 1;
                
                % If the output is desired. Store the s
                if (nargout > 2)
                    if (iIterations >= allocatedSize)
                        allocatedSize = allocatedSize + allocationChunk;
                        X = cat(5,X,zeros(size(obj.mask,1),size(obj.mask,2), nParams, obj.nWalkers, allocationChunk));
                        Lambda = cat(4,Lambda,zeros(nParams, obj.nWalkers, allocationChunk));
                        Sigma = cat(4,Sigma,zeros(1, obj.nWalkers, allocationChunk));
                    end
                    X(:,:,:,:,iIterations)  = obj.maskExpand(obj.theta);
                    Lambda(:,:,iIterations) = obj.lambda;
                    Sigma(:,:,iIterations)  = obj.sigma;
                end
                

                
                
                % Visual feedback to the user
                if (~isempty(obj.viewCallback))
                    x = obj.maskExpand(obj.theta(:,:,1));
                    obj.viewCallback(x, obj.parameterNames, obj.plotFigure);
                end

                if (obj.verbose)
                    disp('---------------------------------------');
                    disp('Burn-in in progress...')
                    
                    disp(['@ sample #', num2str(iIterations)]);
                    disp(['Sigma = ', num2str(mean(obj.sigma,2))]);
                    if (obj.useSpatialModel)
                        disp(['Using a spatial model with: Kappa = ', num2str(obj.kappa)]);
                        
                        disp(['Lambda = [', num2str(mean(obj.lambda,2)'),']']);
                    else
                        disp('Using no spatial model');
                    end
                end
                


                % Check convergence of the chain.
                if (iIterations >= obj.convergenceTestWindow) % At least a full window must be collected
                    if (ringBufferIndex == obj.convergenceTestWindow)
                        indexing = 1:obj.convergenceTestWindow;
                    else
                       start = ringBufferIndex + 1;
                       stop = ringBufferIndex; 
                       indexing = [start:obj.convergenceTestWindow,1:stop];
                    end
                    
                    if (mod(iIterations, obj.convergenceTestSkipFrequency) == 0)
                        half_window1 = 1:round(obj.convergenceTestWindow/2);
                        half_window2 = (round(obj.convergenceTestWindow/2)+1):obj.convergenceTestWindow;
                        Theta1 = Theta(:,:,:,indexing(half_window1));
                        Theta2 = Theta(:,:,:,indexing(half_window2));
                        [obj.converged, metric] = obj.convergenceTestCallback(Theta1, Theta2, obj);
                        if (obj.verbose)
                           disp(['Metric = ', num2str(metric), ' vs Limit = ', num2str(obj.convergenceThreshold)]);
                        end
                    end
                end
            end
          
            hasConverged = obj.converged;   
            
            
            % Calculate the WAIC
            s = s(:,:,indexing);
            logp = obj.loglikelihood(obj.maskExpand(cat(4,Theta1, Theta2)), s);
            [WAIC.computed_waic, WAIC.p_waic, WAIC.waic_se, WAIC.ok] = waic(logp, 2);
            
            if (nargout > 2)
                % Remove unused space in X, Lambda and Sigma
                X = X(:,:,:,:,1:iIterations);
                Lambda = Lambda(:,:,1:iIterations);
                Sigma = Sigma(:,:,1:iIterations);
            end
            
        end
        
        function [X, Lambda, Sigma, WAIC] = sample(obj, nSamples)
            % Perform burn-in until covergence or maxNoOfIterations has
            % been reached. 
            %
            % Input parameters:
            % obj               -   A MCMCSpatial2DModel object.
            % nSamples          -   Number of samples/per walker to be obtained. 
            %
            %
            % Output parameters:
            % X                 -   Samples obtained during burn-in
            %                       organised as [nx, ny, p, w, ns] where p
            %                       is the number of parameters per site, w
            %                       is the number of walkers and ns is the
            %                       number of samples. 
            % Lambda            -   The regularization parameters organized as 
            %                       [2, w, ns]
            % Sigma             -   The standard deviation organized as [1, w, ns]
            % WAIC              -   Information about the Widely Applicable
            %                       Information Criteria for the samples obtained.
            %
            %
            
            % Setup the weights if not already done.
            if (isempty(obj.weight))
               obj.setupWeights();
            end
            
            % Preallocate output
            nParams = numel(obj.parameterMax);
            X = zeros(size(obj.mask,1),size(obj.mask,2), nParams, obj.nWalkers, nSamples);
            Sigma = zeros(1, obj.nWalkers, nSamples);
            Lambda = zeros(nParams, obj.nWalkers, nSamples);
            
            % Perform Gibb's sampling
            for iMC = 1:nSamples*obj.thinning
                obj.gibbsSampler();

                % Store results in X every nThinning passage
                if (mod(iMC,obj.thinning)==0)
                    X(:,:,:,:,iMC/obj.thinning) = obj.maskExpand(obj.theta);
                    Sigma(:,:,iMC/obj.thinning) = obj.sigma;
                    Lambda(:,:,iMC/obj.thinning) = obj.lambda;

                    if (~isempty(obj.viewCallback))
                        obj.viewCallback(X(:,:,:,1,iMC/obj.thinning), obj.parameterNames, obj.plotFigure);
                    end
                    
                    if (obj.verbose)
                        disp('---------------------------------------');
                        disp('Burn-in completed,')
                        disp('Sampling in progress...')
                        disp(['@ sample #', num2str(iMC/obj.thinning)]);
                        disp(['Sigma = ', num2str(mean(obj.sigma,2))]);
                        if (obj.useSpatialModel)
                            disp(['Using a spatial model with: Kappa = ', num2str(obj.kappa)]);
                            disp(['Lambda = [', num2str(mean(obj.lambda,2)'),']']);
                        else
                            disp('Using no spatial model');
                        end
                    end
                end 
            end
            
            % Calculate WAIC
            logp = obj.loglikelihood(X, Sigma);
            [WAIC.computed_waic, WAIC.p_waic, WAIC.waic_se, WAIC.ok] = waic(logp, 2);
        end
        
        function WAIC = calculateWAIC(obj, X, Sigma)
        % Calculate the WAIC.
        %
             logp = obj.loglikelihood(X, Sigma);
            [WAIC.computed_waic, WAIC.p_waic, WAIC.waic_se, WAIC.ok] = waic(logp, 2);
        end
        
        function [ess, thinning] = ESS(obj, X)
        % Estimate the effetive sample size and propose a thinning
        % parameter. One suggestion per (pixel). Uses a AR(1) model.
        %
        % Input:
        % X     - Samples [Nx, Ny, Npara, Nwalkers, Nsamples].
            siz = size(X);
            ess = zeros(siz(1:3));
            thinning = zeros(siz(1:3));
            
            X = obj.maskContract(X);
            
            for ii = 1:siz(3)
            
                % Detrend theta
                tmp =  squeeze(X(:,ii,:,:));
                Theta = bsxfun(@plus, tmp, -mean(tmp,3));

                % The model is now:
                % X[t] = phi*X[t-1] + e

                Xt_1 = Theta(:,:,1:end-1);
                Xt = Theta(:,:,2:end);

                % Average over chains
                phi = mean(sum(Xt_1.*Xt,3)./sum(Xt_1.^2,3),2);
              %  s2 = mean(var(Xt - bsxfun(@times,phi,Xt_1),0,3),2);

                alpha = 1./(1-phi).^2;

                thinning(:,:,ii) = obj.maskExpand(alpha);
                ess(:,:,ii) = obj.maskExpand(size(Theta,3)./alpha);
            end
        end
    end
    
    methods (Access = private)   
        
        function setupWeights(obj)
           if (numel(obj.TVWeight)==1)
              obj.weight = ones(sum(obj.mask(:)),1); 
           else
              obj.weight = obj.TVWeight(obj.mask);
           end
        end
        
        function y = maskExpand(obj, x)
            s = size(x);
            m = size(obj.mask);
            y = zeros([m,s(2:end)]);
            n = prod(s(2:end));
            tmp = zeros(m);
            for ii = 1:n
                tmp(obj.mask) = x(:,ii);
                y(:,:,ii) = tmp;
            end
        end
        
        function y = maskContract(obj, x)
            s = size(x);
            if (numel(s) < 3)
               s = [s, 1]; 
            end
            m = sum(obj.mask(:));
            y = zeros([m,s(3:end)]);
            n = prod(s(3:end));
            for ii = 1:n
                tmp = x(:,:,ii);
                y(:,ii) = tmp(obj.mask);
            end
        end
             
        function logp = loglikelihood(obj, X, Sigma)
            X = obj.maskContract(X);
            Y = obj.maskContract(obj.data);
            X = X(:,:,:); % Reshape to have all samples at the end
            
            if (isempty(obj.fixedParams))
                Z = [];
            else
                Z = obj.maskContract(obj.fixedParams);
            end

            nMCSamples = size(X,3);
            nDataSamples = size(Y,2);
            logp = zeros(nMCSamples, nDataSamples);

            % Get the loglikelihood for each MC-smaple
            np = size(X,1);
            for ii = 1:nMCSamples
               Yhat = obj.model(X(:,:,ii), Z);
               lp = -(np/2)*log(2*pi*Sigma(ii).^2) - (1/(2*Sigma(ii).^2)) * sum( (Y-Yhat).^2, 1);
               logp(ii,:) = lp(:);
            end         
        end
        
        function gibbsSampler(obj) 
            % Loop over subsets
            ar = zeros(obj.nSubsets,1);
            for iSub = 1:obj.nSubsets
                [theta_i, ar(iSub)] = obj.mcSpatial{iSub}.sample(obj.innerThinning, obj.innerThinning);
                currentPixels = obj.neighbourStruct(iSub).pixelSetIndex;
                obj.theta(currentPixels,:) = theta_i(:,:);
            end
            obj.GWAcceptenceRatio = mean(ar);

            % Need the quadratic and TV-terms
            Aterm = zeros(1,obj.nWalkers);
            Bterm = zeros(2,obj.nWalkers);
            % Quadratic-term
            for iSub = 1:9
                currentPixels = obj.neighbourStruct(iSub).pixelSetIndex;
                for iWalker = 1:obj.nWalkers
                    th = obj.theta(currentPixels,:,iWalker);
                    residual = (obj.model(th, obj.fixedParameters{iSub})-obj.y{iSub});
                    Aterm(iWalker) = Aterm(iWalker) + sum(residual(:).^2);
                end
            end

            % TV-terms
             if (obj.useSpatialModel)
                for iSub = 1:9
                    currentPixels = obj.neighbourStruct(iSub).pixelSetIndex;
                    for iWalker = 1:obj.nWalkers
                        th      = obj.theta(currentPixels,:,iWalker);
                        th01    = obj.theta(obj.neighbourStruct(iSub).neighbour_x0_yp1,:,iWalker);
                        th10    = obj.theta(obj.neighbourStruct(iSub).neighbour_xp1_y0,:,iWalker);
                        ex01    = obj.neighbourStruct(iSub).exist_neighbour_x0_yp1;
                        ex10    = obj.neighbourStruct(iSub).exist_neighbour_xp1_y0;
                        w       = obj.weight(currentPixels);
                        for ip = 1:2
                            tv = w.*sqrt(ex10.*(th10(:,ip) - th(:,ip)).^2 + ex01.*(th01(:,ip) - th(:,ip)).^2);
                            Bterm(ip,iWalker) = Bterm(ip,iWalker) + sum(tv(:));
                        end
                    end
                end
             end


            % Sample sigma and lambda.     
            obj.lambda = zeros(2,obj.nWalkers);

            for iWalker = 1:obj.nWalkers
                s2 = MCMCSpatial2DModel.invGammaSampler((obj.M*obj.V-1)/2,Aterm(iWalker)/2,1);
                obj.sigma(iWalker) = sqrt(s2);

                if (obj.useSpatialModel)
                    for ip = 1:2
                        % obj.lambda(ip,iWalker) = MCMCSpatial2DModel.gammaSampler(obj.kappa*(obj.V-1),Bterm(ip,iWalker),1);
                        obj.lambda(ip,iWalker) = MCMCSpatial2DModel.gammaSampler(obj.V*(1-obj.kappa)+1,1/Bterm(ip,iWalker),1);
                    end 
                end    
            end
        end
 
        function lp = logPFun(obj, theta_i, wix, iSubSet)
        % The logprobability used by GWMCMC algoritmen
                nParams = numel(obj.parameterMax);
            
                % Get the neigboors for this subset
                if (obj.useSpatialModel)
                    ns          = obj.neighbourStruct(iSubSet);
                    theta_11    = obj.theta(ns.neighbour_xn1_yp1,:,wix);
                    exist_11    = ns.exist_neighbour_xn1_yp1;
                    theta1_1    = obj.theta(ns.neighbour_xp1_yn1,:,wix);
                    exist1_1    = ns.exist_neighbour_xp1_yn1;
                    theta01     = obj.theta(ns.neighbour_x0_yp1,:,wix);
                    exist01    = ns.exist_neighbour_x0_yp1;
                    theta10     = obj.theta(ns.neighbour_xp1_y0,:,wix);
                    exist10    = ns.exist_neighbour_xp1_y0;
                    theta0_1    = obj.theta(ns.neighbour_x0_yn1,:,wix);
                    exist0_1    = ns.exist_neighbour_x0_yn1;
                    theta_10    = obj.theta(ns.neighbour_xn1_y0,:,wix);
                    exist_10    = ns.exist_neighbour_xn1_y0;
                    w_10        = obj.weight(ns.neighbour_xn1_y0);
                    w00         = obj.weight(ns.pixelSetIndex);
                    w0_1        = obj.weight(ns.neighbour_x0_yn1);
                end

                % The quadratic term
                res = obj.model(theta_i, obj.fixedParameters{iSubSet})-obj.y{iSubSet};
                A = -1/(2*obj.sigma(wix)^2)*sum(res.^2,2);

                % The TV-terms
                B = cell(1,nParams);
                if (obj.useSpatialModel)
                    for jj = 1:nParams
                        B1 = w00.*sqrt(exist10.*(theta10(:,jj) - theta_i(:,jj)).^2 + exist01.*(theta01(:,jj) - theta_i(:,jj)).^2);
                        B2 = w_10.*sqrt(exist_10.*( theta_i(:,jj) - theta_10(:,jj)).^2 + exist_11.*exist_10.*(theta_11(:,jj) - theta_10(:,jj)).^2);
                        B3 = w0_1.*sqrt(exist1_1.*exist0_1.*(theta1_1(:,jj) - theta0_1(:,jj)).^2 + exist0_1.*(theta_i(:,jj) - theta0_1(:,jj)).^2);
                        B{jj} = B1 + B2 + B3;
                    end
                end
                
                lp = A - obj.M*log(2*pi*obj.sigma(wix));
                
                if (obj.useSpatialModel)
                    for jj = 1:nParams
                        lp = lp - obj.lambda(jj,wix)*B{jj};
                    end
                end

                % Set zero probability (-inf log-prob) if outside limits.
                
                zeroProbablityPixels = false(size(theta_i,1),1);
                for jj = 1:nParams
                    zeroProbablityPixels = zeroProbablityPixels | (theta_i(:,jj) <= obj.parameterMin(jj));
                    zeroProbablityPixels = zeroProbablityPixels | (theta_i(:,jj) >= obj.parameterMax(jj));
                end
                lp(zeroProbablityPixels) = -inf;
                
            end
        
        function initialGuess(obj)
            % Get a single curve for all pixels.
            Y = zeros(1,size(obj.data,3));
            
            for ii = 1:numel(Y)
               tmp = obj.data(:,:,ii);
               Y(ii) = mean(tmp(obj.mask));
            end
            
            if (isempty(obj.fixedParams))
                cost = @(x) sum( (obj.model(x, []) - Y).^2 ); 
            else
                Z = zeros(1,size(obj.fixedParams,3));
                for ii = 1:numel(Z)
                   tmp = obj.fixedParams(:,:,ii);
                   Z(ii) = mean(tmp(obj.mask));
                end
                cost = @(x) sum( (obj.model(x, Z) - Y).^2 ); 
            end
            
            % Search for the ML estimate
            
            nParams = numel(obj.parameterMax);
            parameter_mean = fminsearch(cost, obj.typicalParameters(:)');
            signal_mean = obj.model(parameter_mean, Z);
            
            

            
            obj.theta = zeros(obj.V, nParams, obj.nWalkers);
            
            % Add some randomness
            for ii = 1:nParams
               obj.theta(:,ii,:) = parameter_mean(ii) * (1+obj.initRandomnessFraction * (rand(obj.V, 1, obj.nWalkers) - 0.5));
            end
            
            % Get a reasonable guess on sigma           
            r = bsxfun(@plus, obj.data, -reshape(signal_mean,[1,1,numel(Y)]));
            r = obj.maskContract(r);
            sigma_mean = mean(std(r,[],2));
            obj.sigma = sigma_mean * (1+obj.initRandomnessFraction*(rand(1,obj.nWalkers)-0.5));
            
            % Set lambda to 0
            obj.lambda = zeros(nParams,obj.nWalkers);
            % Set lambda to 1
            %obj.lambda = ones(nParams,obj.nWalkers);    
        end
        
        function nInfo = getNeighbourInfo(obj, pixelSetMask)
            % Return for the pixel-set the indexes of the neighbours and if these
            % exist. The indexes are relative to the mask. I.e. if getNeighbourIndex
            % returns [a, b, c] then:
            %
            % N = sum(mask(:));
            % v = false(N,1);
            % v(c) = true;
            % w(a) = true;
            % L = zeros(size(mask));
            % L(v) = true; % This will be an image with the neigbours of the
            % pixelSet set to true. 
            % L(w) = true; % This will be an image with the the
            % pixelSet set to true.

            [nInfo.neighbour_xn1_yp1, nInfo.exist_neighbour_xn1_yp1, nInfo.pixelSetIndex] = obj.getNeighbourIndex(pixelSetMask,-1,1);
            [nInfo.neighbour_xp1_yn1, nInfo.exist_neighbour_xp1_yn1] = obj.getNeighbourIndex(pixelSetMask,1,-1);
            [nInfo.neighbour_xp1_y0, nInfo.exist_neighbour_xp1_y0] = obj.getNeighbourIndex(pixelSetMask,1,0);
            [nInfo.neighbour_xn1_y0, nInfo.exist_neighbour_xn1_y0] = obj.getNeighbourIndex(pixelSetMask,-1,0);
            [nInfo.neighbour_x0_yp1, nInfo.exist_neighbour_x0_yp1] = obj.getNeighbourIndex(pixelSetMask,0,1);
            [nInfo.neighbour_x0_yn1, nInfo.exist_neighbour_x0_yn1] = obj.getNeighbourIndex(pixelSetMask,0,-1);
        end

        function [neighbourIndex, exist_neighbour, pixelSetIndex] = getNeighbourIndex(obj, pixelSetMask,dx,dy)
            maskIndex = 1:sum(obj.mask(:));
            maskIndexImage = zeros(size(obj.mask));
            maskIndexImage(obj.mask) = maskIndex;

           n1 = size(obj.mask,1);
           n2 = size(obj.mask,2);
           neighbourIndex = ones(sum(pixelSetMask(:)),1);
           exist_neighbour = false(sum(pixelSetMask(:)),1);
           pixelSetIndex = zeros(sum(pixelSetMask(:)),1);
           ii = 0;
           for i1 = 1:n1
               for i2 = 1:n2
                   if (pixelSetMask(i1,i2))
                        ii = ii + 1;
                        pixelSetIndex(ii) =  maskIndexImage(i1,i2);
                        exist_neighbour(ii) = false;
                        if (inRange(i1+dx,i2+dy))
                            if (obj.mask(i1+dx,i2+dy))
                                neighbourIndex(ii) = maskIndexImage(i1+dx,i2+dy);
                                exist_neighbour(ii) = true;
                            end
                        end
                   end
               end
           end

            function tf = inRange(x,y)
               tf = (x > 0) & (y > 0) & (x <= n1) & (y <= n2);  
            end

        end
    end
    
    methods  (Access = public, Static = true) 
        function [test, metric] = testBayesianDistributionConvergence(W1, W2, MCMC)
            % Input shapes: [n, p, c, N]. Where n is the number of spatial sites, p is
            % the numbers of parameters at each site, c is the number of samplers, and
            % N is the number of samples drawn from the posterior.
            if (isempty(MCMC.convergenceTestVariables))
               variables = 1:size(W1,2);
            else
               variables = MCMC.convergenceTestVariables;
            end

            pmax = zeros(size(variables));
            pmax_value = zeros(size(variables));


            if (~isempty(MCMC.plotConvergence))
                figure(MCMC.plotConvergence);
            end


            for ii = 1:numel(variables)
                idxVariable = variables(ii);

                if (MCMC.convergenceTestFraction < 1)
                    mask = MCMC.convergenceTestMask;
                    A = squeeze(W1(mask,idxVariable,:,1:end));  % [pixels, chains, samples]
                    B = squeeze(W2(mask,idxVariable,:,1:end));  % [pixels, chains, samples]
                else
                    A = squeeze(W1(:,idxVariable,:,1:end));  % [pixels, chains, samples]
                    B = squeeze(W2(:,idxVariable,:,1:end));  % [pixels, chains, samples]
                end

                % OBS - Need to implement min too. Requires new likelihood
                % functions.
                lambda_max = MCMC.parameterMax(idxVariable);
                lambda_min = MCMC.parameterMin(idxVariable);

                sigma2_1 = AR1_variance_estimator_(A);
                a = mean(A(:, :), 2);  % [pixels, samples]
                sigma2_2 = AR1_variance_estimator_(B);
                b = mean(B(:, :), 2);  % [pixels, samples]

                if (~isempty(MCMC.plotConvergence))
                    m_plot = 301;  % 301 plot points
                else
                    m_plot = 11;  % Use few if not plotted anyway...
                end
                [pmax(ii), pmax_value(ii), p_plot, pis] = posterior_max(m_plot, a, b, lambda_max, lambda_min, sigma2_1, sigma2_2);

                if (~isempty(MCMC.plotConvergence))
                    % Shape of the subplot 
                    nCol = ceil(sqrt(numel(variables)));
                    nRow = ceil(numel(variables) / nCol) + 1;

                    figure(MCMC.plotConvergence)
                    % View the estimates the converged fraction

                    ax = subplot(nRow, nCol, 1:nCol);
                    plot(pis, p_plot,'LineWidth', 2);
                    hold on

                    % View the histograms
                    subplot(nRow, nCol, nCol + ii)
                    nBins = ceil(sqrt(numel(a)));
                    histogram(a, nBins, 'Normalization', 'pdf', 'DisplayStyle', 'stairs', 'EdgeColor', 'k', 'LineWidth', 2);
                    hold on
                    histogram(b, nBins, 'Normalization', 'pdf', 'DisplayStyle', 'stairs', 'EdgeColor', 'b', 'LineWidth', 2);
                    hold off
                    if (~isempty(variables))
                        varName = MCMC.parameterNames{variables(ii)};
                    else
                        varName = ['Var', num2str(variables(ii))];
                    end

                    title(['Pixel value distribution: ', varName]);
                    legend('First half', 'Last half')
                end

            end 

            % Common plot stuff
            if (~isempty(MCMC.plotConvergence))
                axes(ax);

                if (~isempty(variables))
                    texts = cell(size(variables));
                    for ii = 1:numel(variables)
                        texts{ii} = [MCMC.parameterNames{variables(ii)}, ' (', num2str(pmax(ii)), ')'];
                    end
                    legend(texts);
                end
                title(['Estimated converged fraction [Acceptence criteria: ', num2str(MCMC.convergenceThreshold), ']']);
                hold off
                drawnow;
            end

            if (all(pmax_value > 1.0))
                test = (min(pmax) >= MCMC.convergenceThreshold);
            else
                test = false;
            end
            metric = min(pmax);

            % Helper functions
            function [mean_s2, phi] = AR1_variance_estimator_(theta)
                % See details in:
                %     Sokal (1996). "Monte Carlo Methods in Statistical
                %     Mechanics: Foundations and New Algorithms"

                % theta [Nvariables, Nchains, Nsamples]

                % Detrend theta
                theta = bsxfun(@plus, theta, -mean(theta, 3));

                % The model is now:
                % X[t] = phi * X[t - 1] + e

                Xt_1 = theta(:, :, 1:(end - 1));
                Xt = theta(:, :, 2:end);

                % Average over chains
                phi = mean(sum(Xt_1 .* Xt, 3) ./ sum(Xt_1.^2, 3), 2);  % [Nvariables]
                % s2 = var(Xt - bsxfun(@times, phi, Xt_1), 0, 3);  % [Nvariables, Nchains]
                s2 = mean(var(Xt - bsxfun(@times, phi, Xt_1), 0, 3), 2);  % [Nvariables]

                s = size(theta);
                % mean_s2 = mean(s2 ./ (1 - phi).^2 ./ prod(s(2:3)), 2);
                % Variance of the mean
                mean_s2 = (s2 ./ (1 - phi).^2) ./ prod(s(2:3));
            end

            function [pmax, pmax_value, p_plot, pis] = posterior_max(N_, m, n, lambda_max, lambda_min, sigma2_1, sigma2_2)

                % Find the peak
                objective = @(x)(-log_likelihood(m, n, x, lambda_max, lambda_min, sigma2_1, sigma2_2));
                [pmax, pmax_value] = fminbnd(objective,0,1);
                pmax_value = -pmax_value;
                f1 = -objective(1.0);
                if (f1 > pmax_value)
                    pmax = 1;
                    pmax_value = f1;
                end

                % Find the width of the peak as the distance to reduced hight by 1/e.
                cost = @(x)((abs(objective(x)+pmax_value)-1)^2);
                [p_width, fval_width] = fminbnd(cost,0,pmax); %#ok<ASGLU>

                % Calculate values in a neighbourhood around pmin 5 widths large atmost.
                p1 =  pmax - 5*abs(p_width-pmax);
                p2 =  pmax + 5*abs(p_width-pmax);
                p1 = max(0,p1);
                p2 = min(1,p2);

                pis = linspace(p1,p2,N_);
                evidence = zeros(N_, 1);
                w = zeros(N_,1);

                % Simpson's rule:
                dx_3 = (p2-p1) / (N_ * 3);  % Interval is [0, 1], hence 1 [= (1 - 0)].

                % First point
                pi_i = pis(1);
                evidence(1) = log_likelihood(m, n, pi_i, lambda_max, lambda_min, sigma2_1, sigma2_2);
                w(1) = log(dx_3);

                % Points 2, ..., N - 1

                for i = 2:(N_ - 1)
                    pi_i = pis(i);
                    if mod(i, 2) == 0
                        param = 4;
                    else
                        param = 2;
                    end
                    w(i) = log(param * dx_3);
                    evidence(i) = log_likelihood(m, n, pi_i, lambda_max, lambda_min, sigma2_1, sigma2_2);
                end

                % Last point
                evidence(end) = log_likelihood(m, n, pis(end), lambda_max, lambda_min, sigma2_1, sigma2_2);
                w(end) =  log(dx_3);

                p_plot = exp(evidence - logsumexp(evidence + w));

                % The pmax value is a logarithm and unnormalized
                pmax_value = exp(pmax_value - logsumexp(evidence + w));
            end

            function p = log_likelihood(m, n, pi_, lambda_max, lambda_min, sigma2_1, sigma2_2)
                AA = pi_ * evidence_same(m, n, lambda_max, lambda_min, 0.5 * (sigma2_1 + sigma2_2)) + ...
                        (1 - pi_) * evidence_different(m, n, lambda_max, lambda_min, sigma2_1, sigma2_2);
                    p = sum(log(AA));
            end

            function p = evidence_same(m, n, lambda_max, lambda_min, sigma2)

                AA = exp((-(m - n).^2) ./ (4 * sigma2)) ./ (4 * lambda_max * sqrt(pi * sigma2));
                BB = erf((m - 2 * lambda_max + n) ./ (2 * sqrt(sigma2))) - erf((m + n -2 * lambda_min) ./ (2 * sqrt(sigma2)));

                p = -AA .* BB;
            end

            function p = evidence_different(m, n, lambda_max, lambda_min, sigma2_1, sigma2_2)
                AA = (1 / (4 * (lambda_max-lambda_min)^2));
                BB = erf((lambda_max - m) ./ sqrt(2 * sigma2_1)) - erf((lambda_min - m) ./ sqrt(2 * sigma2_1));
                C = erf((lambda_max - n) ./ sqrt(2 * sigma2_2)) - erf((lambda_min - n) ./ sqrt(2 * sigma2_2));

                p = AA .* BB .* C;
            end

            function lse = logsumexp(lps)
                max_val = max(lps);
                lse = max_val + log(sum(exp(lps - max_val)));
            end   
        end 
  
        function defaultViewCallback(theta, parameterNames, fig)
            nParams = numel(parameterNames);
            
            % Shape of the subplot 
            n = ceil(sqrt(nParams));
            m = ceil(nParams/n);
            
            if (~isempty(fig))
                figure(fig);
            end
            for ii = 1:nParams
               subplot(m,n,ii);
               img = theta(:,:,ii);
               v = prctile(img(:), [1, 99]);
               a = v(1) - 0.1*(v(2)-v(1));
               b = v(2) + 0.1*(v(2)-v(1));
               if (all([a,b]==0))
                   b = 1;
               end
               imagesc(theta(:,:,ii),[a, b]);
               axis off
               title(parameterNames{ii});   
               drawnow;
            end   
        end
    end

    
    
    methods (Static = true, Access = private)
        function [converged, metric] = testHistogramConvergence(x1, x2, MCMC)
            minMetric = inf;
            for ii = 1:size(x1,2)
               v1 = mean(squeeze(x1(:,ii,:)),2);
               v2 = mean(squeeze(x2(:,ii,:)),2);
               [~, p] = kstest2(v1, v2); 
               if (p<minMetric)
                   minMetric = p;
               end
            end

            converged = minMetric > MCMC.convergeLimit;
            metric = minMetric;
        end
        
        % Sample sigma^2
        function s2 = invGammaSampler(alpha,beta,n)
            shape = alpha;
            scale = 1/beta;
            s2_inv = gamrnd(shape,scale,n);
            s2 = 1./s2_inv;
        end

%         % Sample sigma
%         function s = invGammaSampler(N,A,n)
%             shape = N/2+1;
%             scale = 2/A;
%             tau = gamrnd(shape,scale,n);
%             s = 1./sqrt(tau);
%         end
        function l = gammaSampler(alpha,beta,n)
            shape = alpha;
            scale = beta;
            l = gamrnd(shape,scale,n);
        end

        % Sample lambda
%         function l = gammaSampler(N,B,n)
%             shape = N+1;
%             scale = 1/B;
%             l = gamrnd(shape,scale,n);
%         end

        function [computed_waic, p_waic, waic_se, ok] = waic(logp, method, normalize)
            if nargin < 2
                method = 2;
            end
            if nargin < 3
                normalize = false;
            end

            lppd_i = MCMCSpatial2DModel.logmeanexp(logp,1); %log(mean(exp(logp), 1));  % Implement log-sum-exp instead!!

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
    end
        

        
end
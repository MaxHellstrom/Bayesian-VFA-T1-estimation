classdef GWMCMC < handle
    % A MCMC sampler based on the stretch move method in: 
    % Goodman & Weare (2010), Ensemble Samplers With Affine Invariance, Comm. App. Math. Comp. Sci., Vol. 5, No. 1, 65–80
    
    properties (SetAccess = public)
        % Handle to the log(P) function
        logPFun;
        % Current state of the chain. 
        Xt;
        % Number of ensable walkers
        nWalkers;
        % Number of parameters
        nParams;
        % Number of parallel systems
        nSystems;
        % The step size of the algorithm. Default = 2.
        StepSize = 2;
        callback = [];
    end
    
    
    
    methods
        function obj = GWMCMC(logPFun,X0)
            % Create an ensable sampler.
            %
            % Input:
            % logPFun   - A function handle to the logarithm of the probablity to 
            %           be sampled from. Note that this function should
            %           take 2 inputs. The current state of N-systems and an integer 
            %           representing the current walker. If N parallel
            %           systems are used and each system has M variables
            %           then the first argument should be a N x M matrix.
            % X0        - The initial state of the N parallel systems. This 
            %           should be a N x M x W  matrix where N is the # of parallel systems, 
            %           M is the # of variables and W is the # of walkers.
            %
            %
            % Example: Sample 100 parallel 2D Gaussians with mean 0 and cov
            % = [1 0.5; 0.5 2] using 10 walkers.
            %
            % % Initial state
            % X0 = randn(100,2,10)
            %
            % % log(P)-function
            % icov = inv([1 0.5; 0.5 2]);
            % logPFun = @(x,wix)(-1/2*sum((x*icov).*x,2));
            %
            % % Setup the system
            % mc = GWMCMC(logPFun,X0);
            %
            % % Burnin (500 iterations)
            % mc.sample(500,500);
            % 
            % % 1000 samples with thinning = 10.
            % theta = mc.sample(1000*10,10);
            
            
           obj.logPFun = logPFun;
           obj.Xt = X0;
           
           [obj.nSystems, obj.nParams, obj.nWalkers] = size(X0);
        end
        
        function [X, acceptanceRatio] = sample(obj,nSteps,thinning)
            % Performs sampling.
            %
            % Input:
            % obj       - A GWMCMC object
            % nSteps    - The number of simulated samples
            % thinning  - The amount of thinning
            %
            % Output:
            % X                 - The samples of size (N,M,W,nSteps/thinning).
            %                   See constructor for information about N,M,W.
            % acceptanceRatio   - The fraction of samples accepted. Should
            %                   in general be between 23% and 50%.
            
            
            
            % Get states
            curm = obj.Xt;
            
            % Get the current probabilities
            curlogP = zeros(obj.nSystems, obj.nWalkers);
            for iWalker = 1:obj.nWalkers
              curlogP(:,iWalker) = obj.logPFun(curm(:,:,iWalker),iWalker); 
            end 
            
            % Total accepted moves
            nAccepted = 0;
            
            % Allocate output
            X = zeros(obj.nSystems, obj.nParams, obj.nWalkers, floor(nSteps/thinning));
            
            % The stretch move implementation of the GWGW algorithm.
            for iStep = 1:nSteps
                
                % Generate proposals for all walkers
                rix = mod((1:obj.nWalkers) + floor(rand * (obj.nWalkers - 1)), obj.nWalkers) + 1;  % Pick a random partner
                zz = ((obj.StepSize - 1) * rand(obj.nSystems, 1, obj.nWalkers) + 1).^2 / obj.StepSize; % Use the inverse transform method to sample from Eq. 9 in the paper.
                proposedm = curm(:,:, rix) - bsxfun(@times, (curm(:,:, rix) - curm), zz);
                
                % Loop over walkers
                for wix = 1:obj.nWalkers
                    % Calculate the log(P) of the proposed move 
                    proposedlogP = obj.logPFun(proposedm(:,:, wix),wix); % 2017-09-15 AG test
                    
                    % Check if moves are accepted. Eq. 7 in the paper.
                    lr = log(rand(obj.nSystems,  1));
                    acceptfullstep = lr < ((obj.nParams-1) * log(zz(:,wix)) + proposedlogP - curlogP(:, wix));

                    % The number of accepted moves
                    ac = sum(acceptfullstep);

                    % Update the current log(P) and current state.  
                    if (ac > 0) 
                        curm(acceptfullstep,:,wix) = proposedm(acceptfullstep,:, wix);
                        curlogP(:,wix) = obj.logPFun(curm(:,:,wix),wix); 
                    end
                    
                    % Upate acceptance statistics
                    nAccepted = nAccepted + ac;
                    
                    % Callback for e.g. visualization.
                    if (~isempty(obj.callback))
                            obj.callback(acceptfullstep,curm(:,:,wix),proposedlogP,wix,iStep);
                    end
                end
                
                % Store the result in the output nd-vector.
                if (0 == mod(iStep,thinning))
                    X(:, :, :, floor(iStep/thinning)) = curm;
                end 
            end
            
            % Store state.
            obj.Xt = curm;
            
            % Calculate the acceptance ratio 
            acceptanceRatio =  nAccepted / (nSteps*obj.nSystems*obj.nWalkers);
        end
    end
    
    
end


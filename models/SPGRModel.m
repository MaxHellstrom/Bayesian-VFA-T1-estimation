%% Define the model function for the VFA method
function y = SPGRModel(theta, B1corr, TR, FA)
   Pd = theta(:,1);
   T1 = theta(:,2);
   V = numel(Pd);
   M = numel(FA);
   y = zeros(V,M);

   E1 = exp(-TR./T1);
   if (T1<0)
      return; 
   end
   for ii = 1:M
      y(:,ii) = Pd.*sin(B1corr.*FA(ii)/180*pi).*(1-E1)./(1-cos(B1corr.*FA(ii)/180*pi).*E1);  
   end 
end
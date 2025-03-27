function [S] = spectralmatrix_nonoverlap(u);

%makes spectral density matrix of u by using nonoverlapping data
% presumes that u is formed from e_t+1*x_t of overlapping annual data
% does it all 12 ways and takes the average
% calculates S under the null, thus no lags are required. 
% demean = 1 if subtract off means in S (usually a good idea) 

demean = 1; 
[T,N]=size(u);

S = 0; 
for i = 0:11;
    umo = u(i+1:12:T,:);
    if demean; 
        umo = umo-ones(size(umo,1),1)*mean(umo);
    end; 
    S = S+umo'*umo/(T/12);
end; 

      
    

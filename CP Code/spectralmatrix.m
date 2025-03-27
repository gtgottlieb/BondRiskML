function [S] = spectralmatrix(u, lags, weight);

%  lags number of lags to include in GMM corrected standard errors
%  weight: 1 for newey-west weighting 0 for even weighting
% demean = 1 if subtract off means in S (usually a good idea) 

demean = 1; 
[T,N]=size(u);

  
	S= u'*u/T;

    for indx = (1:lags);
      sadd1 = u(1+indx:T,:)  - demean*ones(T-indx,1)*mean(u(1+indx:T,:));
      sadd2 = u(1:T-indx,:)  - demean*ones(T-indx,1)*mean(u(1:T-indx,:));
     
      sadd = sadd1'*sadd2/T;
      sadd = sadd + sadd'; 
      S = S + (1-weight*indx/(lags+1))*sadd;
	    
    end;
      
    

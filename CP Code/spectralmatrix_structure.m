function [S] = spectralmatrix_structured(e,x);

%  forms structured S matrix, presuming e is error and x is rhv from a one year horizon regression
% using overlapping monthly data. Assumes no chs so E(x x * e e ) = E(xx)*E(ee) and assumes that all serial 
% correlation is due to overlap so E(et et-j) = j/12 * E(et et)

% order of moments, as in fbregs -- error(1)* x(1), x(2).... then error(2)*x(1), x(2) ... etc.
% thus S is [ e(1)^2 x'x  e(1) e(2) x'x ...
%             e(2) e(1) x'x ....

% u = x_t.*e_t+12
%       S = sum j = -12 to 12 E(x_t .* e_t+12  x_t-j .* e_t+12-j)
%       assume chs so 
%       E(x(i)_t .*  x(k)_t-j) * E(e(m)_t * e(n)_t+12-j)
%       assume serial correlation of errors is only due to overlap so 
%       E(e(m)_t * e(n)_t+12-j) = #months overlap / 12 * E(e(m_t) e(n_t))

demean = 1; 
[T,Ne]=size(e);
[T,Nx] = size(x); 

covee = e-ones(T,1)*mean(e); 
covee = covee'*covee/T; 

S = kron(covee,x'*x/T); % note use x'x matrix not demeaned. 

for indx = 1:12; 
    S = S + kron((12-indx)/12*covee,x(1+i:T,:)'*x(1:T-i,:)/T) ...
          + kron((12-indx)/12*covee,x(1:T-i,:)'*x(1+i:T,:)/T); % uses 1/T in x as elsewhere
end; 
      
    

function X = pinv2(A,r)
%PINV   Pseudoinverse.
% This version allows you to impose how many nonzero eigenvalues there are rather than use the tolerance function
% This is important; in several cases in "bond risk premia", the tolerance picked up eigenvalues that really should have been zero
%   X = PINV(A) produces a matrix X of the same dimensions
%   as A' so that A*X*A = A, X*A*X = X and A*X and X*A
%   are Hermitian. The computation is based on SVD(A) and any
%   singular values less than a tolerance are treated as zero.
%   The default tolerance is MAX(SIZE(A)) * NORM(A) * EPS.
%
%   PINV(A,TOL) uses the tolerance TOL instead of the default.
%
%   See also RANK.

%   Copyright 1984-2000 The MathWorks, Inc. 
%   $Revision: 5.9 $  $Date: 2000/06/01 02:04:17 $

[U,S,V] = svd(A,0);
[m,n] = size(A);
if m > 1, s = diag(S);
   elseif m == 1, s = S(1);
   else s = 0;
end

   s = diag(ones(r,1)./s(1:r));
   X = V(:,1:r)*s*U(:,1:r)';





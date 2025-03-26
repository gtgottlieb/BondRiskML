close all; clear all;

% some value for gamma
% tent-shaped
%gamma=[-1;-1/2;0;-1/2;-1]; 
% jc adds: no level effect; gamma'1 = 0

gamma=[-1/2;0;1;0;-1/2]; 

% vector in paper
a=[-2; -1; 0; 1; 2];

disp('Check whether gamma*a = 0');
disp(gamma'*a);

% forward vector = D* yield vector
D=[1 0 0 0 0; ...
  -1 2 0 0 0; ...
   0 -2 3 0 0; ...
   0 0 -3 4 0; ...
   0 0 0 -4 5];

% JC adds: compute forward vector corresponding to linear yield

disp('forward vector corresponding to linear yield'); 
disp(D*a); 

disp('note it''s still linear, though it is shifted up.'); 


% compute gammastar
gammastar=gamma'*D;
gammastar=gammastar';

disp('Now check whether gammastar*a =0');
disp(gammastar'*a);

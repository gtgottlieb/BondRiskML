   function b_rest = dorest(lhv,rhv1,rhv2,label); 

   % does restricted and unrestricted regressions to test restrictions, e.g. "is slope alone enough?") 
   % note restricted regression should explicity include the constant if you want it. 
   % you can specify rhv1 = [] and do unrestricted regressions with the same program. 
   % usually you will want rhv1 = constant and rhv2 = right hand variables to test joint sig in an unrestricted regression however
   
   N1 = size(rhv1,2); 
   N2 = size(rhv2,2);   

   disp(' '); % blank to separate from last run
   if N2 == 0; 
        disp('dorest: error. no second variables. to do unrestricted regression make rhv1 = [], rhv2 = everything, or rhv1 = const'); 
   end; 
   if rank([rhv1 rhv2]) < size([rhv1 rhv2],2); 
       disp(label); 
       disp('dorest: error. Restricted and rest do not span all yields.'); 
   end; 
   
   % 1. Restricted regression
   if N1 > 0; 
       [b_rest,olsb_rest,R2_rest,R2humpadj,v] = olsgmm(lhv, rhv1,18,1);
       errc = lhv - [rhv1]*b_rest; % constrained errors 
       % do we want to report whether the restricted model is significant, beyond R2? Here it is, I left it off the table. 
       test_rest = b_rest'*inv(v)*b_rest; 
       pval_rest = 100*(1-cdf('chi2',test_rest,N1));  
       fprintf(' restricted coeffs and R2    '); 
       fprintf(' %8.2f ',b_rest,R2_rest); 
       fprintf('\n'); 
   else; 
       R2_rest = 0; 
       b_rest = 0; 
   end;     
   
   % 2. unrestricted regression transformed to special variable, then others
   % doesn't matter what the last ones are so long as we span the unrestricted regression
   
   [gammas_t,olsy_trash,R2hump,R2humpadj,v] = olsgmm(lhv,[rhv1 rhv2],18,1); % t for transformed
   teststat = gammas_t(N1+1:end)'*inv(v(N1+1:end,N1+1:end))*gammas_t(N1+1:end); 
   pval = 100*(1-cdf('chi2',teststat,N2)); 
   fprintf('%23s &',label); 
   fprintf('%8.2f &%8.2f &%8.2f &%8.2f &%8.2f \\\\ \n', [R2_rest teststat N2 chi2inv(0.95,N2) pval]); 
   
   % 3. Still not trusting things, the JT or R2 test
   % unrestricted moments are E(x*error) and equal to zero
   % restricted moments are E( x1.*.(y-x1*b) ) 
   %                        E( x2.*.(y-x1*b) ) 
   % a matrix is ( I 0 ) -- the second set of moments is not set equal to zero. 
   % by gmm, cov(gt) = (I - d(ad)^-1a) S ()'
   % d = dgT/db' = E(x1*x1')
   %               E(x2*x1')
   % Use unrestricted S to test
   
   % Delted: JT and Wald tests are numerically identical. DUH!
       
      T = size(lhv,1); 
      
%   err = lhv - [rhv1 rhv2]*gammas_t; 
%   u = [rhv1 rhv2].*(err*ones(1,N1+N2)); 
%   T = size(lhv,1); 
%   S = (u-ones(T,1)*mean(u))'*(u-ones(T,1)*mean(u))/T;
%   for i = 1:18; 
%       S = S+(19-i)/19*(((u(1+i:end,:)-ones(T-i,1)*mean(u(1+i:end,:)))'*(u(1:end-i,:)-ones(T-i,1)*mean(u(1:end-i,:))))/T+...
%                        ((u(1:end-i,:)-ones(T-i,1)*mean(u(1:end-i,:)))'*(u(1+i:end,:)-ones(T-i,1)*mean(u(1+i:end,:))))/T); 
%   end; 
%   d = [rhv1'*rhv1/T ; rhv2'*rhv1/T]; 
%   a = [eye(N1) zeros(N1,N2)];         
%   covgt = (eye(N1+N2) - d*inv(a*d)*a)*S*(eye(N1+N2) - d*inv(a*d)*a)'/T;
%   gt = mean([rhv1 rhv2].*(errc*ones(1,N1+N2)))'; 
   %disp('gt'); 
   %disp(gt'); check that the first gt really are zero
%   chi2stat = gt(N1+1:N1+N2)'*inv(covgt(N1+1:N1+N2,N1+1:N1+N2))*gt(N1+1:N1+N2);
%   pval = 100*(1-cdf('chi2',chi2stat,N2)); 
%   fprintf('    ", JT test        & %8.2f &%8.2f &%8.2f &%8.2f &%8.2f \\\\ \n', [R2_rest chi2stat N2 chi2inv(0.95,N2) pval]); 
       
   
   % can we impose some structure on the S matrix to give better cov matrix estimation
   % S = E(xx')^-1 sum (E(x_t e_t x_t-j' e_t-j)) E(xx')^-1
   % suppose no CHS
   % S = E(xx')^-1 sum [ (E(x_t x_t-j)' E(e_t e_t-j)]  E(xx')^-1
   % Suppose e_t = sum(u_t-k ut-k+1...u_t)
   % then E(e_t e_t-j) = (# months overlap) / 12 * E(e_t e_t) 

   err = lhv - [rhv1 rhv2]*gammas_t;
   sigerr = err'*err/T; 
   rhv = [ rhv1 rhv2]; 
   
   
   % Deleted: JT and Wald tests are numerically identical. 
   %S = (rhv-ones(T,1)*mean(rhv))'*(rhv-ones(T,1)*mean(rhv))/T * sigerr; ;
   %for i = 1:11; 
   %    S = S+(((rhv(1+i:end,:)-ones(T-i,1)*mean(rhv(1+i:end,:)))'*(rhv(1:end-i,:)-ones(T-i,1)*mean(rhv(1:end-i,:))))/T*sigerr*(12-i)/12+ ...
   %           ((rhv(1:end-i,:)-ones(T-i,1)*mean(rhv(1:end-i,:)))'*(rhv(1+i:end,:)-ones(T-i,1)*mean(rhv(1+i:end,:))))/T*sigerr*(12-i)/12); 
   %end; 
  % covgt = (eye(N1+N2) - d*inv(a*d)*a)*S*(eye(N1+N2) - d*inv(a*d)*a)'/T;
  % chi2stat = gt(N1+1:N1+N2)'*inv(covgt(N1+1:N1+N2,N1+1:N1+N2))*gt(N1+1:N1+N2);
  % pval = 100*(1-cdf('chi2',chi2stat,N2)); 
  % fprintf(' ", JT test, struct. S &%8.2f &%8.2f &%8.2f &%8.2f &%8.2f \\\\ \n', [R2_rest chi2stat N2 chi2inv(0.95,N2) pval]); 

   % 3.5. Actually, it's better to do this with regression standard errors. Only difference is the JT test uses demeaned S. This is a bit more
   % straightforward. This is modified from teh same section in fbregs from gamma'f on all forwards.  
   
   Exx = rhv'*rhv/T ;
   S = rhv'*rhv/T * sigerr; 
   for i = 1:11; 
       S = S+( rhv(1+i:end,:)'*rhv(1:end-i,:)/T*sigerr*(12-i)/12+ ...
               rhv(1:end-i,:)'*rhv(1+i:end,:)/T*sigerr*(12-i)/12); 
   end; 
   v_st = inv(Exx)*S*inv(Exx)/T;   
   se_st = diag(v_st);
   se_st = sign(se_st).*(abs(se_st).^0.5);
   teststat = gammas_t(N1+1:N1+N2)'*inv(v_st(N1+1:N1+N2,N1+1:N1+N2))*gammas_t(N1+1:N1+N2); 
   pval = 100*(1-cdf('chi2',teststat,N2));
   fprintf(' ",wald test, struct.S &%8.2f &%8.2f &%8.2f &%8.2f &%8.2f \\\\ \n', [R2_rest teststat N2 chi2inv(0.95,N2) pval]); 

 
   
   % 4. Same thing with nonoverlapping data -- I am dubious about joint test statistics, still near-singular cov matrix
   % since the results (r2 in particular) depend on starting month, report average result over all starting months
   T = size(lhv,1); 
   ann = 1:12:T; 
   v2avg = zeros(N1+N2,N1+N2);  
   for smonth = 1:12; 
       [b_rest2,olsb_rest2,R2_rest2(smonth),R2humpadj2,v2] = olsgmm(lhv(ann+smonth-1), rhv1(ann+smonth-1,:),0,0);
       [gammas_t2,olsy_trash,R2hump2,R2humpadj2,v2] = olsgmm(lhv(ann+smonth-1),[rhv1(ann+smonth-1,:) rhv2(ann+smonth-1,:)],0,0); % t for transformed    
       % the following version uses the full sample estimate and the monthly covariance matrix. 
       teststat(smonth) = gammas_t(N1+1:end)'*inv(v2(N1+1:end,N1+1:end))*gammas_t(N1+1:end); 
       v2avg = v2avg+v2; 
       % the following version uses the monthly estimate too. 
       % teststat(smonth) = gammas_t2(N1+1:end)'*inv(v2(N1+1:end,N1+1:end))*gammas_t2(N1+1:end); 
       %pval(smonth) = 100*(1-cdf('chi2',teststat(smonth),N2));
       %if 0 ; % make this if 1 to see monthly detail
       %   fprintf('    ", no overlap      &%8.2f &%8.2f &%8.2f &%8.2f &%8.2f \\\\ \n', [R2_rest2(smonth) teststat(smonth) N2 chi2inv(0.95,N2) pval(smonth)]); 
       %end; 
   end; 

   v2avg = v2avg/12;
   R2_rest2 = mean(R2_rest2)'; 
  
   teststatavg = mean(teststat'); 
   pval = 100*(1-cdf('chi2',teststatavg,N2));
   fprintf('",no overlap, avg chi2 &%8.2f &%8.2f &%8.2f &%8.2f &%8.2f \\\\\n', [R2_rest2 teststatavg N2 chi2inv(0.95,N2) pval]);    

   
   teststat = gammas_t(N1+1:end)'*inv(v2avg(N1+1:end,N1+1:end))*gammas_t(N1+1:end); 
   pval = 100*(1-cdf('chi2',teststat,N2));
   fprintf('",no overlap, avg v    &%8.2f &%8.2f &%8.2f &%8.2f &%8.2f \\\\\n', [R2_rest2 teststat N2 chi2inv(0.95,N2) pval]);    
   
   return; 
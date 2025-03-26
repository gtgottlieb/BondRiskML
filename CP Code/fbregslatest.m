% MAIN PROGRAM FOR "BOND RISK PREMIA" 

close all;
clear; 

printgraph = 1; % prints graphs and overwrites old graphs

% load bond price data
% this is the file from CRSP completely unmodified
% sample: 1952 - 2003

load bondprice.dat;
T=length(bondprice);

y=-log(bondprice(:,2:end)/100).*(ones(T,1)*[1/1 1/2 1/3 1/4 1/5]);
famablisyld=[bondprice(:,1) y];

% collect the annual yields, form prices, forwards, hpr
% yields(t), forwards(t) are yields, forwards at time t

yields=famablisyld(:,2:end);
mats=[1 2 3 4 5]'; 
prices=-(ones(T,1)*mats').*yields;
forwards = prices(:,1:4)-prices(:,2:5);
fs = forwards-yields(:,1)*ones(1,4);

% hprx(t) is the holding period return over last year

hpr = prices(13:T,1:4)-prices(1:T-12,2:5);
hprx = hpr - yields(1:T-12,1)*ones(1,4);
hpr = [zeros(12,1)*ones(1,4); hpr];     % pads out the initial values with zeros so same length as other series       
hprx = [zeros(12,1)*ones(1,4); hprx];

% make nice dates series to get nice graphs

dates = famablisyld(:,1); 
yr = floor(dates/10000);
mo = dates-10000*yr;
mo = floor(mo/100);
day = dates-10000*yr-100*mo;
dates = yr+mo/12;

beg  = 140;      % set beginning date 140 = 1964. Same as FB sample, and previous data unreliable per Fama. 

% capitalized variables do not follow the convention of being padded out with initial zeros
% instead, HPRX starts 12 months later, so the first HPRX is in 65 while the first FS, FT, YT is 1964. 
% These are set up so you can regress HPRX, AHPRX on YT, FT, etc. directly
% with no subscripting. They also include a column of ones for use in
% regressions. 


HPRX = 100*hprx(beg+12:T,:);

AHPRX = mean(HPRX')';

Ts   = T-beg-12+1;
FS   = [ones(Ts,1) 100*fs(beg:T-12,:)];     % forward-spot spread
FT   = [ones(Ts,1) 100*yields(beg:T-12,1) 100*forwards(beg:T-12,:)]; % yeilds and forwards
YT   = [ones(Ts,1) 100*yields(beg:T-12,:)];  % all yields

% *********************************************************************
% Table 1 -- regressions of excess returns of each bond on all forward
% rates
% *********************************************************************
    
disp('-----------------------------------------------------');
disp('TABLE 1: returns on all forwards, 1964-2003');

[betas,stbetas,R2,R2adj,v,F_trash] = olsgmm(HPRX,FT,12,0);  % std errors: HH with 12 lags
[betas,stbetas_trash,R2_trash,R2adj_trash,v,F] = olsgmm(HPRX,FT,18,1); % tests: NW with 18 lags
errall = HPRX-FT*betas;
         
% regressions of actual (not log) returns to check Gallant suspicions

lhv = 100*(exp(hprx(beg+12:T,:)+yields(beg:T-12,1)*ones(1,4))-yields(beg:T-12,1)*ones(1,4));
[blevel,stlevel,R2level,R2adjlevel,vlevel,Flevel] = olsgmm(lhv,FT,12,0);

% make the table

fprintf( '      %8s %8s %8s %8s %8s %8s %8s %8s %8s %8s %8s \n',...
         'matur', 'const', 'y1', 'f1->2',  'f2->3' , 'f3->4', 'f4->5' ,'R2', 'R2adj' ,'R2lev' ,'chi2/%p' );
mat = [2 betas(:,1)' R2(1) R2adj(1) R2level(1) F(1,1) ;
       0 stbetas(:,1)' 0     0           0       F(1,3)*100; 
       3 betas(:,2)' R2(2) R2adj(2) R2level(2) F(2,1) ;
       0 stbetas(:,2)' 0     0           0       F(2,3)*100; 
       4 betas(:,3)' R2(3) R2adj(3) R2level(3) F(3,1) ;
       0 stbetas(:,3)' 0     0           0       F(3,3)*100; 
       5 betas(:,4)' R2(4) R2adj(4) R2level(4) F(4,1) ;
       0 stbetas(:,4)' 0     0           0       F(4,3)*100]; 
fprintf('coef   %8.2f %8.2f %8.2f %8.2f %8.2f %8.2f %8.2f %8.2f %8.2f %8.2f %8.2f  \n',mat(1,:)); 
fprintf('s.e.   %8.2f %8.2f %8.2f %8.2f %8.2f %8.2f %8.2f %8.2f %8.2f %8.2f %8.2f  \n',mat(2,:));
fprintf('coef   %8.2f %8.2f %8.2f %8.2f %8.2f %8.2f %8.2f %8.2f %8.2f %8.2f %8.2f  \n',mat(3,:)); 
fprintf('s.e.   %8.2f %8.2f %8.2f %8.2f %8.2f %8.2f %8.2f %8.2f %8.2f %8.2f %8.2f  \n',mat(4,:));
fprintf('coef   %8.2f %8.2f %8.2f %8.2f %8.2f %8.2f %8.2f %8.2f %8.2f %8.2f %8.2f  \n',mat(5,:)); 
fprintf('s.e.   %8.2f %8.2f %8.2f %8.2f %8.2f %8.2f %8.2f %8.2f %8.2f %8.2f %8.2f  \n',mat(6,:));
fprintf('coef   %8.2f %8.2f %8.2f %8.2f %8.2f %8.2f %8.2f %8.2f %8.2f %8.2f %8.2f  \n',mat(7,:)); 
fprintf('s.e.   %8.2f %8.2f %8.2f %8.2f %8.2f %8.2f %8.2f %8.2f %8.2f %8.2f %8.2f  \n',mat(8,:));
  
% graph is below together with restricted model
% monte carlos are run separately 

disp('chi2(5) 5% and 1% critical values'); 
disp(chi2inv([0.95 0.99],5)); 

disp(' How important is the "level factor" -- the fact that gamma prime 1 is not zero? '); 
disp(' R2 using all forward spreads, no level');
[betasd,stbetas,R2,R2adj,v] = olsgmm(HPRX,[FT(:,1) FT(:,3)-FT(:,2) FT(:,4)-FT(:,3) FT(:,5)-FT(:,4) FT(:,6)-FT(:,5)],12,0);
disp(R2');

% ******************************************
% Table 2, single factor model
% ******************************************

disp('---------------------------------------------------------------------');
disp('TABLE 2: Two  step OLS to fit restricted model ');


   olsse = zeros(10,1);
   [gammas,olsse(5:end),R2hump,R2humpadj,v,F] = olsgmm(AHPRX,FT,18,1); % joint tests use NW 18 lags
   [gammas,olsse(5:end),R2hump,R2humpadj,v] = olsgmm(AHPRX,FT,12,0); % std errors using HH
   hump = FT*gammas; % in sample fit. 
   erravg=AHPRX-hump;
   humpall=[ones(T,1) 100*yields(:,1) 100*forwards(:,:)]*gammas; % covers all period (i.e. including the last year)
                                                                 % 100s because FT was multipled by 100.   
   

   % Estimation of b's (without constant)
   
   [bets,olsse(1:4),R2hprx,R2hprxadj,v] = olsgmm(HPRX,hump,12,0);
   bets = bets'; % usual convention of column vectors
   err = HPRX-hump*bets';
       
   % Calculate standard errors of two step by using two step OLS moments
   
   u = [ (erravg*ones(1,6)).*FT  err.*(hump*ones(1,4)) ]; 
   gt = mean(u)'; 
   Eff=FT'*FT/size(FT,1);
   Eef = err'*FT/size(FT,1);
   Erf = HPRX'*FT/size(FT,1);
   d = -[  Eff                       zeros(6,4) ; 
           -Erf+2*bets*gammas'*Eff   gammas'*Eff*gammas*eye(4)];
           
   S = spectralmatrix(u,12,0);   
   gmmsex = diag(inv(d)*S*inv(d)'/size(FT,1)).^0.5;
   gmmse = olsse*0; 
   gmmse(5:end) = gmmsex(1:6);  % formula above was derived with gamma first
   gmmse(1:4) = gmmsex(7:10);   % convention below is bs first
   
   disp(' gammas, ols se, gmm se');
   disp([ (gammas)'; (olsse(5:end))'; (gmmse(5:end))' ])
   disp('R2, chi2stat, dof, crit value, % p value'); 
   disp([ R2hump F(1) F(2) chi2inv(0.95,F(2)) 100*F(3)]); 
   
   disp('bn, ols se, gmm se, R2');;
   disp([(bets) (olsse(1:4)) (gmmse(1:4)) (R2hprx) ]);
  
   
   disp('Interpolation for Table 6 (stock return forecasting): linear');
   a=olsgmm(bets,[ones(4,1) (2:5)' ],0,0);
   disp('for year 6 - 10');
   disp(6:0.5:10); 
   disp(1*a(1)+(6:0.5:10)*a(2) );
  
   disp('Interpolation for Table 6 (stock return forecasting): linear plus square root');
   a2=olsgmm(bets,[ones(4,1) (2:5)' (2:5)'.^0.5 ],0,0);
   disp('for year 6 - 10');
   disp(6:0.5:10); 
   disp(1*a2(1)+(6:0.5:10)*a2(2)+(6:0.5:10).^0.5*a2(3) );
  
   
   disp('Interpolation for Table 6 (stock return forecasting): linear+exponential');
   a3=olsgmm(bets,[ones(4,1) (2:5)' exp(-(2:5)')],0,0);
   disp('for year 6 - 12');
   disp(6:0.5:10); 
   disp(1*a3(1)+(6:0.5:10)*a3(2)+exp(-(6:0.5:10))*a3(3) );
   
   figure; 
   plot((2:12),1*a(1)+(2:12)*a(2),...
        (2:12),1*a2(1)+(2:12)*a2(2)+(2:12).^0.5*a2(3),...
        (2:12),1*a3(1)+(2:12)*a3(2)+exp(-(2:12))*a3(3),...
        (2:5),bets,'v');
   title('b coefficients and interpolation line used to evaluate stock return forecasts'); 
   legend('linear','liner, square rooot','linear, exponential',2); 
   axis([2 12 0 3]); 
   
% *******************************************
% plot of tent shaped forward rate coefficients. 
% *******************************************
   
   bigfont = 20; 
   smallfont = 18; 
    figure;
	subplot(2,1,1);
	plot(  (1:5),betas(2:6,4),'-ok',...
    	   (1:5),betas(2:6,3),'-vb',...
		   (1:5),betas(2:6,2),'-sg',...
           (1:5),betas(2:6,1),'-dm','Linewidth',2,'MarkerSize',12);         
    legend('5','4','3','2');
    set(gca,'xtick',[1 2 3 4 5],'FontSize',smallfont)
    title('Unrestricted','FontSize',bigfont);
    axis([-inf inf -3.1 4.4]);
	
	subplot(2,1,2);
	plot((1:5),bets(4)*gammas(2:end)','-ok',...
		   (1:5),bets(3)*gammas(2:end)','-vb',...
		   (1:5),bets(2)*gammas(2:end)','-sg',...
		   (1:5),bets(1)*gammas(2:end)','-dm','Linewidth',2,'MarkerSize',12);
	legend('5','4','3','2');
	set(gca,'xtick',[1 2 3 4 5],'FontSize',smallfont)
    axis([-inf inf -3.1 4.4]);
    title('Restricted','FontSize',bigfont);
    if printgraph;
   	    print -depsc2 fb1.eps;
    end;
    
          
 % plot of constant restriction -- edited out for length, but pretty to
 % look at. 
   figure; 
   plot((1:4),betas(1,:),'-ok',...
        (1:4),bets*gammas(1),'--vg','Linewidth',1);
   legend('Unrestricted','Restricted');
   set(gca,'xtick',[1 2 3 4]);
   xlabel('lhv maturity'); 
   title('Constant in excess return forecast regression');
   
 %  if printgraph;
 %  print -depsc2 fb2.eps;
 %  end;

 % *****************************
 % GMM tests in Table 12, 13
 % ******************************
 
   disp('--------------------------------'); 
   disp('GMM tests of 2 step model'); 
   
   % calculate wald and JT test of inefficient two step model
   
   u = [(err(:,1)*ones(1,6)).*FT ...
        (err(:,2)*ones(1,6)).*FT ...
        (err(:,3)*ones(1,6)).*FT ...
        (err(:,4)*ones(1,6)).*FT];
   % note this uses the restricted errors. Mean u will be gt. S may be affected
   gt = mean(u)'; 
  
   [betamat,betase,R2unc,R2uncadj,v] = olsgmm(HPRX,FT,12,0);
   errunc=HPRX-FT*betamat;
   betamat = betamat'; % now is the same size and shape as b*gamma 
   
  uu = [(errunc(:,1)*ones(1,6)).*FT ...
        (errunc(:,2)*ones(1,6)).*FT ...
        (errunc(:,3)*ones(1,6)).*FT ...
        (errunc(:,4)*ones(1,6)).*FT];
   
   a = [ kron(ones(1,4),eye(6)) ; 
         kron(eye(3),gammas') zeros(3,6) ]; 
      
   d = [ kron([-eye(3);ones(1,3)],Eff*gammas)  -kron(bets,Eff)];
   
   wtval = [18]';  % report the usual 18 lags newey west as other tests 
  
   for indx = 1:size(wtval,1); % loop allows you to try variations in number of weights. 
       disp('lags'); 
       disp(wtval(indx)); 
       S = spectralmatrix(uu,wtval(indx),1);   % newey west 18 lags spectral matrix. 
       S_no = spectralmatrix_nonoverlap(uu);   % spectral matrix formed from nonoverlapping data
       S_st = spectralmatrix_structure(errunc,FT); % spectral matrix formed from no chs and overlap for serial 
                    
       % JT test 
       
       covgt = (eye(24)-d*inv(a*d)*a)*S*(eye(24)-d*inv(a*d)*a)'/size(FT,1);
       invcovgt = pinv2(covgt,15);
       jtstat = (gt'*invcovgt*gt);
       pval = 100*(1-cdf('chi2',jtstat,15));
       cutoff = chi2inv(0.95,15); 

       covgt_no = (eye(24)-d*inv(a*d)*a)*S_no*(eye(24)-d*inv(a*d)*a)'/size(FT,1);
       invcovgt_no = pinv2(covgt_no,15);
       jtstat_no = (gt'*invcovgt_no*gt);
       pval_no = 100*(1-cdf('chi2',jtstat_no,15));

       covgt_st = (eye(24)-d*inv(a*d)*a)*S_st*(eye(24)-d*inv(a*d)*a)'/size(FT,1);
       invcovgt_st = pinv2(covgt_st,15);
       jtstat_st = (gt'*invcovgt_st*gt);
       pval_st = 100*(1-cdf('chi2',jtstat_st,15));

       
       % wald tests
       
       du = kron(eye(4),Eff); 
       covbu = kron(eye(4),inv(Eff)); 
       covbu = covbu*S*covbu'/size(FT,1);   % cov matrix of unr pars
       covbu = (covbu+covbu')/2; 
       
       covbu_no = kron(eye(4),inv(Eff)); 
       covbu_no = covbu_no*S_no*covbu_no'/size(FT,1);   % cov matrix of unr pars
       covbu_no = (covbu_no+covbu_no')/2; 

       covbu_st = kron(eye(4),inv(Eff)); 
       covbu_st = covbu_st*S_st*covbu_st'/size(FT,1);   % cov matrix of unr pars
       covbu_st = (covbu_st+covbu_st')/2; 
       
       upar = [betamat(1,:) betamat(2,:) betamat(3,:) betamat(4,:)]';
       rpar = bets*gammas';
       rpar = [rpar(1,:) rpar(2,:) rpar(3,:) rpar(4,:)]'; 
       disp('unrestricted, restricted parameters, diff, and se and t'); 
       disp('this is the t referred to in the text for individual parameter equality between constrained and unconstrained models.') ; 
       disp([upar rpar upar-rpar diag(covbu).^0.5 (upar-rpar)./(diag(covbu).^0.5)]);
       
       teststat = (upar-rpar)'*inv(covbu)*(upar-rpar); 
       teststat_no = (upar-rpar)'*inv(covbu_no)*(upar-rpar); 
       teststat_st = (upar-rpar)'*inv(covbu_st)*(upar-rpar); 
       
       pval_wald = ([100*(1-cdf('chi2',teststat,15))]); 
       pval_wald_no = ([100*(1-cdf('chi2',teststat_no,15))]); 
       pval_wald_st = ([100*(1-cdf('chi2',teststat_st,15))]); 
       
       disp('JT stats for joint restrictions test'); 
       fprintf('               %8s %8s %8s %8s \n',  'NW, 18','No ovlp','Strct','5% cv'); 
       fprintf(' JT statistic  %8.2f %8.2f %8.2f %8.2f \n', jtstat, jtstat_no, jtstat_st, cutoff); 
       fprintf('  pct p value  %8.2f %8.2f %8.2f %8.2f \n', pval, pval_no, pval_st, 5); 
       fprintf('Wald test chi2 %8.2f %8.2f %8.2f  \n', teststat, teststat_no,teststat_st); 
       fprintf('  pct  p value  %8.2f %8.2f %8.2f  \n', pval_wald, pval_wald_no,pval_wald_st); 
              
       
   end;  % ends weights loop (if any) 
     
   % check -- standard errors. Should be the same as above. 
   
   disp('standard errors for b, gamma from unconstrained moments -- check same as above'); 
   covunc = diag(inv(a*d)*a*S*a'*inv(a*d)'/size(FT,1)).^0.5; 
   disp(covunc');  

   % *********************************************************
   % Now, the same thing but with lagged right hand variables. 
   % *********************************************************
   
   disp('--------------------------------'); 
   disp('Addidional GMM tests of the 1 factor model using lagged right hand variables.'); 
   
   MAXL=[1:5];
   
   for i=1:length(MAXL);
       
       maxL=MAXL(i);
       disp('Lagged by this many months:'); disp(maxL);
       
       AHPRXL=AHPRX(1+maxL:end,:);
       HPRXL=HPRX(1+maxL:end,:);
       FTL=FT(1:end-maxL,:);
       EffL=FTL'*FTL/size(FT,1);
       
       olsseL = zeros(10,1);
       [gammasL,olsseL(5:end),R2humpL,R2humpadjL,vL] = olsgmm(AHPRXL,FTL,12,0);
       humpL = FTL*gammasL; 
       erravgL=AHPRXL-humpL;
       
       % Estimation of b's (without constant)
       
       [betsL,olsseL(1:4),R2hprxL,R2hprxadjL,vL] = olsgmm(HPRXL,humpL,12,0);
       betsL = betsL'; % usual convention of column vectors
       errL = HPRXL-humpL*betsL';
       
       % calculate wald and JT test of inefficient two step model
       
       uL = [(errL(:,1)*ones(1,6)).*FTL ...
               (errL(:,2)*ones(1,6)).*FTL ...
               (errL(:,3)*ones(1,6)).*FTL ...
               (errL(:,4)*ones(1,6)).*FTL];
       
       % note this uses the restricted errors. Mean u will be gt. S may be affected
       gtL = mean(uL)'; 
       
       [betamatL,betaseL,R2uncL,R2uncadjL,vL] = olsgmm(HPRXL,FTL,12,0);
       erruncL=HPRXL-FTL*betamatL;
       betamatL = betamatL'; % now is the same size and shape as b*gamma 
       
       uuL = [(erruncL(:,1)*ones(1,6)).*FTL ...
               (erruncL(:,2)*ones(1,6)).*FTL ...
               (erruncL(:,3)*ones(1,6)).*FTL ...
               (erruncL(:,4)*ones(1,6)).*FTL];
       
       aL = [ kron(ones(1,4),eye(6)) ; 
           kron(eye(3),gammasL') zeros(3,6) ]; 
       
       dL = [ kron([-eye(3);ones(1,3)],EffL*gammasL)  -kron(betsL,EffL)];
       
       wtval = [18]';  % report the usual 18 lags newey west as other tests 
       
       for indx = 1:size(wtval,1); % loop allows you to try variations
           disp(wtval(indx)); 
           SL = spectralmatrix(uuL,wtval(indx),1);   
           SL_no = spectralmatrix_nonoverlap(uuL); % spectral matrix formed from nonoverlapping data
           SL_st = spectralmatrix_structure(erruncL,FTL); % spectral matrix formed from no chs and overlap for serial 
           
           % JT test 
           
           covgtL = (eye(24)-dL*inv(aL*dL)*aL)*SL*(eye(24)-dL*inv(aL*dL)*aL)'/size(FTL,1);
           invcovgtL = pinv2(covgtL,15);
           jtstatL = (gtL'*invcovgtL*gtL);
           pvalL = 100*(1-cdf('chi2',jtstatL,15));
           
           covgtL_no = (eye(24)-d*inv(a*d)*a)*SL_no*(eye(24)-d*inv(a*d)*a)'/size(FTL,1);
           invcovgtL_no = pinv2(covgtL_no,15);
           jtstatL_no = (gtL'*invcovgtL_no*gtL);
           pvalL_no = 100*(1-cdf('chi2',jtstatL_no,15));
           
           covgtL_st = (eye(24)-d*inv(a*d)*a)*SL_st*(eye(24)-d*inv(a*d)*a)'/size(FTL,1);
           invcovgtL_st = pinv2(covgtL_st,15);
           jtstatL_st = (gtL'*invcovgtL_st*gtL);
           pvalL_st = 100*(1-cdf('chi2',jtstatL_st,15));
           
           % wald tests
           
           duL = kron(eye(4),EffL); 
           covbuL = kron(eye(4),inv(EffL)); 
           covbuL = covbuL*SL*covbuL'/size(FTL,1);   % cov matrix of unr pars. MP: I changed size(FT to size(FTL here 5/31/04
           covbuL = (covbuL+covbuL')/2; 
           
           covbuL_no = kron(eye(4),inv(EffL)); 
           covbuL_no = covbuL_no*SL_no*covbuL_no'/size(FTL,1);   % cov matrix of unr pars
           covbuL_no = (covbuL_no+covbuL_no')/2; 
           
           covbuL_st = kron(eye(4),inv(EffL)); 
           covbuL_st = covbuL_st*SL_st*covbuL_st'/size(FTL,1);   % cov matrix of unr pars
           covbuL_st = (covbuL_st+covbuL_st')/2; 
           
           
           uparL = [betamatL(1,:) betamatL(2,:) betamatL(3,:) betamatL(4,:)]';
           rparL = betsL*gammasL';
           rparL = [rparL(1,:) rparL(2,:) rparL(3,:) rparL(4,:)]'; 
           %disp('unrestricted, restricted parameters, diff, and se and t'); 
           %disp('this is the t referred to in the text for individual parameter equality between constrained and unconstrained models.') ; 
           %disp([uparL rparL uparL-rparL diag(covbuL).^0.5 (uparL-rparL)./(diag(covbuL).^0.5)]);
           
           teststatL = (uparL-rparL)'*inv(covbuL)*(uparL-rparL); 
           teststatL_no = (uparL-rparL)'*inv(covbuL_no)*(uparL-rparL); 
           teststatL_st = (uparL-rparL)'*inv(covbuL_st)*(uparL-rparL); 
           
           pval_waldL = ([100*(1-cdf('chi2',teststatL,15))]); 
           pval_waldL_no = ([100*(1-cdf('chi2',teststatL_no,15))]); 
           pval_waldL_st = ([100*(1-cdf('chi2',teststatL_st,15))]); 
           
           fprintf('JT stats for joint restrictions test using lagged data -- lags = %8f \n',i);
           fprintf('               %8s %8s %8s %8s \n',  'NW, 18','No ovlp','Strct','5% cv'); 
           fprintf(' JT statistic  %8.2f %8.2f %8.2f %8.2f \n', jtstatL, jtstatL_no, jtstatL_st, cutoff); 
           fprintf('  pct p value  %8.2f %8.2f %8.2f %8.2f \n', pvalL, pvalL_no, pvalL_st, 100*0.05); 
           fprintf('Wald test chi2 %8.2f %8.2f %8.2f  \n', teststatL, teststatL_no,teststatL_st); 
           fprintf('  pct  p value  %8.2f %8.2f %8.2f  \n', pval_waldL, pval_waldL_no,pval_waldL_st); 
           
           
       end;
       
   end;
   
   
   % ***********************************************************
   % Exploration of standard errors and joint tests for Table 2 
   % ***********************************************************
   
   % 1. Hansen Hodrick
   olsse = zeros(10,1);
   [gammas,olsse(5:end),R2hump,R2humpadj,vHH,FHH] = olsgmm(AHPRX,FT,12,0);
   teststat = gammas(2:end)'*inv(vHH(2:end,2:end))*gammas(2:end); 
   dof = size(gammas(2:end),1); 
   pval = 100*(1-cdf('chi2',teststat,dof));
     
   disp('\begin{tabular}{rcccccccc}'); 
   fprintf('          Method          &%8s &%8s &%8s &%8s &%8s &%8s &%8s &%8s \\\\ \n', 'const', 'y1','f2','f3','f4','f5','chi2','\%prob'); 
   fprintf('          OLS estimates   &%8.2f &%8.2f &%8.2f &%8.2f &%8.2f &%8.2f \\\\ \n', gammas'); 
   fprintf(' Hansen-Hodrick, 12 lags  &%8.2f &%8.2f &%8.2f &%8.2f &%8.2f &%8.2f &%8.2f &%8.2f \\\\ \n', olsse(5:end), teststat, pval); 

   teststat = gammas(2:end)'*inv(diag(diag(vHH(2:end,2:end))))*gammas(2:end); 
   pval = 100*(1-cdf('chi2',teststat,dof));
   fprintf('   HH t and diagonal chi2 &%8.2f &%8.2f &%8.2f &%8.2f &%8.2f &%8.2f &%8.2f &%8.2f \\\\ \n', gammas./olsse(5:end), teststat, pval); 
   
   % 2. Newey West
   [gammas_NW,se_NW,R2_trash,R2adj_trash,vNW,FNW] = olsgmm(AHPRX,FT,18,1);
   teststat = gammas_NW(2:end)'*inv(vNW(2:end,2:end))*gammas_NW(2:end); 
   pval = 100*(1-cdf('chi2',teststat,dof));
   fprintf('     Newey-West, 18 lags &%8.2f &%8.2f &%8.2f &%8.2f &%8.2f &%8.2f &%8.2f &%8.2f  \\\\ \n', se_NW, teststat, pval); 
   % Note that this is numerically identical to the JT test
      
   % 3. structured HH
   % S = E(xx')^-1 sum (E(x_t e_t x_t-j' e_t-j)) E(xx')^-1
   % suppose no CHS
   % S = E(xx')^-1 sum [ (E(x_t x_t-j)' E(e_t e_t-j)]  E(xx')^-1
   % Suppose e_t = sum(u_t-k u_t-k+1...u_t)
   % then E(e_t e_t-j) = (# months overlap) / 12 * E(e_t e_t) 

   err = AHPRX - FT*gammas;
   sigerr = err'*err/Ts;  
   Exx = FT'*FT/Ts ;
   S = FT'*FT/Ts * sigerr; 
   for i = 1:11; 
       S = S+( FT(1+i:end,:)'*FT(1:end-i,:)/Ts*sigerr*(12-i)/12+ ...
               FT(1:end-i,:)'*FT(1+i:end,:)/Ts*sigerr*(12-i)/12); 
   end; 
   v_st = inv(Exx)*S*inv(Exx)/Ts;   
   se_st = diag(v_st);
   se_st = sign(se_st).*(abs(se_st).^0.5);
   teststat = gammas(2:end)'*inv(v_st(2:end,2:end))*gammas(2:end); 
   pval = 100*(1-cdf('chi2',teststat,dof));
   fprintf('           Simplified HH &%8.2f &%8.2f &%8.2f &%8.2f &%8.2f &%8.2f &%8.2f &%8.2f  \\\\ \n', se_st, teststat,  pval); 
  

   % nonoverlapping data
   % since the results depend on starting month, report average result over all starting months
   ann = 1:12:Ts; 
   v2avg = zeros(6,6);  
   for smonth = 1:12; 
       [gammas_mo,se_mo(:,smonth),R2_mo(smonth),R2adj_mo,v2_mo] = olsgmm(AHPRX(ann+smonth-1),FT(ann+smonth-1,:),0,0); 
       % the following version uses the full sample estimate and the monthly covariance matrix. 
       v2avg = v2avg + v2_mo; 
   end; 
   v2avg = v2avg/12; 
   teststat = gammas(2:end)'*inv(v2avg(2:end,2:end))*gammas(2:end); 
   pval = 100*(1-cdf('chi2',teststat,dof));
   fprintf('   Non-overlap, average v&%8.2f &%8.2f &%8.2f &%8.2f &%8.2f &%8.2f &%8.2f &%8.2f  \\\\ \n', mean(se_mo'), teststat,  pval); 
   disp('\end{tabular}'); 
   
   disp('note: Non-overlap presents OLS standard errors and test statistic using non-overlapping annual returns; standard error is'); 
   disp('the average standard error across starting months; chi2 is formed using the average parameter covariance matrix across strating months. '); 
   disp('and the full sample estimates'); 

   disp('note: 10%, 5% and 1% critical values for chi2(5) are '); 
   disp(chi2inv([0.90 0.95 0.99],dof)); 
   
   % ********************************************
   % Comparison to Fama Bliss and short rate forecast
   % ********************************************
   
   disp('----------------------------------------------------------');
   disp('TABLE 3: Fama Bliss Regression Coefficients, 1964-2003');

   R2un=zeros(4,1); 
   
   indx = 1;  
   while indx <= 4;
      % regression of holding period returns on forward-spot spread
	  
      [bi,HHi,R2(indx),R2adj(indx),v,Fi] = olsgmm(HPRX(:,indx),FS(:,[1 indx+1]),18,1);
      % though not necessary since it's scalar, for compatibility with others, report chi2 using NW, 18 lags
      [bi,HHi,R2(indx),R2adj(indx),v,Fi_trash] = olsgmm(HPRX(:,indx),FS(:,[1 indx+1]),12,0);
      fitfb(:,indx) = FS(:,[1 indx+1])*bi; % full sample fit 
         
      bFB(indx,:)=bi';
      HH(indx,1:2) = HHi';
      F(indx,1:3) = Fi; 
                
      % regression of changes in short rate on forward-spot spread
      rhv = [ones(length((beg:T-12*indx)),1) 100*fs(beg:T-12*indx,indx)];
      lhv = 100*yields(beg+12*indx:T,1)-100*yields(beg:T-12*indx,1);
      [bi,HHi,R2r(indx,1),R2radj(indx),v,Fi] = olsgmm(lhv,rhv,18*indx,1);
      [bi,HHi,R2r(indx,1),R2radj(indx),v,Fi_trash] = olsgmm(lhv,rhv,12*indx,0);

      br(indx,:)=bi';
      HHr(indx,1:2) = HHi';
      Fr(indx,1:3) = Fi;  
	  indx = indx+1;
  end;
    
   disp('hprx(n) (t) = a + b [ f (n-1 -> n) (t) - y(1) (t)]')
   disp(' maturity  constant    coefficient   H-Hconst H-Hcoef   R2  F %p ')
   disp([(2:5)' (bFB) (HH) (R2) F(:,1) F(:,3)*100]);
   disp('Following is not used as we do not look past 1 year horizon'); 
   disp('y(1)(t+n) -y(1)(t) = a + b [ f (n-1 -> n) (t) - y(1) (t)]')
   disp(' maturity  constant   coefficient   H-Hconst HHcoef     R2   F % p');
   disp([(1:4)' (br) (HHr) (R2r) Fr(:,1) Fr(:,3)*100 ]);
   disp('note: standard errors use Hansen-Hodrick GMM correction for overlapping data');

   disp('dof for FB return regression'); 
   disp(F(1,2));
   disp('5% , 1% critical value'); 
   disp(chi2inv([0.95 0.99],F(1,2))); 
        
       
   disp('----------------------------------------------------');
   disp('TABLE 4: Horse race between gammaf  and fama bliss');

   bhr  = zeros(3,4);
   sthr = zeros(3,4);
   R2hr = zeros(4,1);
   for indx =1 :4 
       [bhr(:,indx),sthr(:,indx),R2hr(indx),R2hradj(indx),vi] = olsgmm(HPRX(:,indx),[FS(:,[1 indx+1]) hump],12,0);
   end
    
   disp(' maturity  constant  fama bliss  gammaf  R2')
   disp([(2:5)' bhr' R2hr]);
   disp(' maturity  se    R2')
   disp([(2:5)' sthr' R2hr]);
   disp('note: standard errors use Hansen-Hodrick GMM correction for overlapping data');

   
   disp('-----------------------------------------------------');
   disp('TABLE 5: Short Rate Changes');

   [by,stby,R2y,R2ya,vy,F] = olsgmm(100*(yields(beg+12:T,1)-yields(beg:T-12,1)),FS(:,1:2),18,1);
   [by,stby,R2y,R2ya,vy] = olsgmm(100*(yields(beg+12:T,1)-yields(beg:T-12,1)),FS(:,1:2),12,0);
             
   disp(' RHS = forward-spot spread ');
   disp(' constant  coefficients   R2 chi2 dof % p');
   disp([ (by') (R2y) F(1:2) F(3)*100]);
   disp(' se constant  se coefficients  ');
   disp([ (stby')]);
   
   [by,stby,R2y,R2ya,vy,F] = olsgmm(100*(yields(beg+12:T,1)-yields(beg:T-12,1)),FT,18,1);
   [by,stby,R2y,R2ya,vy] = olsgmm(100*(yields(beg+12:T,1)-yields(beg:T-12,1)),FT,12,0);
             
   disp(' RHS = all forwards ');
   disp(' constant  coefficients   R2 chi2 dof %p');
   disp([ (by') (R2y) F(1:2) F(3)*100]);
   disp(' se constant  se coefficients  ');
   disp([ (stby')  (R2y)]);
   
   bspot = by; % saved

   disp('-------------------------------------------------------');
   disp('TABLE 6 -- need to run macro.m');

   % ***********************
   % Yield factor models 
   % ***********************
   
   disp('-------------------------------------------------------');  
   disp('TABLE 7: Yield Factors');
  
   [gammasy,astb,aR2,aR2adj,va] = olsgmm(AHPRX,YT,12,0);

   % Eigenvalue decomposition for yields
   
   Vy = cov((yields(beg:T,:))); 
   [Q,L] = eig(Vy);
   % this produces Q*L*Q' = Vy. The eigenvalues are not necessarily ordered.  
   % reorder with largest eigenvalue on top 
   [D,Indx] = sort(-diag(L));
   D = -D; 
   Q2 = Q(:,Indx); 
   L2 = diag(D,0); 
   Q = Q2; 
   L = L2; 

   PC=YT(:,2:end)*Q;  % principal components for forecasting
   % note that yields(beg:T,:)*Q are orthogonal, but YT*Q are a little bit shorter
   % so not exactly orthogonal in sample. Where possible, use 
   % PC = yields(beg:T,:)*Q
   level = PC(:,1);   
   slope = PC(:,2);
   curve = PC(:,3); 
      
   % for matrices x = Q'*yields
   % for time series: X = Y*Q
  
   figure; % not in paper, but pattern of last factors fyi
    plot(mats,Q(:,4),'-or',mats,Q(:,5),'-vg');
    legend('4th','5th');
    title('extra yield factors, not in paper'); 

  % ----------------------------------------------------------------
  % Table 7 -- performance of yield factor models and gamma prime f
  % ----------------------------------------------------------------
  
   disp('Line 1 of Table 7 -- explaining the variance of yields');
   % eigenvalue decomposition of yield changes -- factor variances. 
   % eigenvalues / sum(eigenvalues) * 100
   disp(D'./sum(D)*100); 
   
   % now we run yields and yield changes on gamma prime f. This completes the right columns of the table discussing eigenvalue yield factors. 
   disp('last column of Line 2, Table 7');

   %disp('Regression of contemporaneous yields on gamma prime f'); 
   rhv = [ones(T-beg+1,1) humpall(beg:T)]; 
   [bx,bxse,R2x,R2xadj,vx] = olsgmm(yields(beg:T,:),rhv,12,0);
   % Equivalent of eigenvalues for this factor 
   covpred = cov(rhv*bx);  % covariance matrix of yield ch fitted from gammaprime f
   disp('percent  of y variance accounted for by gamma prime f'); 
   disp(100*trace(covpred)/sum(D)); 
      
   %disp('Regression of contemporaneous yield changes on change in gamma prime f'); 
   rhv = [ones(T-beg,1) diff(humpall(beg:T))]; 
   [bx,bxse,R2x,R2xadj,vx] = olsgmm(diff(yields(beg:T,:)),rhv,12,0);
   % Equivalent of eigenvalues for this factor 
   covpred = cov(rhv*bx);  % covariance matrix of yield ch fitted from gammaprime f
   disp('percent  of dy variance accounted for by change in gamma prime f'); 
   disp(100*trace(covpred)/sum(D)); 
    
   disp('Line 2 of Table 7');
   % decomposition of variance of gamma prime f due to various factors
   
   for indx = 1:5; 
      decompovar(indx) = 100*gammasy(2:end)'*Q(:,indx)*D(indx)*Q(:,indx)'*gammasy(2:end)/...
                         (gammasy(2:end)'*Q*diag(D)*Q'*gammasy(2:end));
   end; 
   disp(decompovar);
    
      
   disp('line 2a of table 7. Correlation of gamma prime f with the factors -- decided it wasn''t very interesting'); 
   trash = corrcoef([hump,PC]); 
   disp(trash(1,2:end)); 
   
   
   disp('Line 3 and 4 of Table 7 -- forecasting returns with yield curve factors');
   
   R2f = zeros(5,1); R2f2=R2f;
     for indx = 1:5;
    
       rhv = [YT(:,1) PC(:,1:indx)];
       rhv2 = [YT(:,1) PC(:,indx)];
      
       [gamf,gamfse,R2f(indx),R2fadj,vf] = olsgmm(AHPRX,rhv,12,0);
       [gamf2,gamfse2,R2f2(indx),R2fadj2,vf2] = olsgmm(AHPRX,rhv2,12,0);
       %disp(indx);
       %disp(gamf'); 
       %disp(gamf2'); 
       % I put these in to verify that single = multiple regressions are exactly equal 
       %  when we start with cov(YT) rather than cov(yields)
     end; 
   
   
   disp('      n          R2      joint R2');  
   disp(round([(1:5)' R2f2 R2f]*1000)/10);

   disp(' as a reminder, gamma prime f as right hand variable. R2 and coefficient'); 
   rhv3 = [ones(size(hump,1),1) hump];
   [gamgf,gamgfse,R2gf,R2gfadj,vgf] = olsgmm(AHPRX,rhv3,12,0);
   disp([R2gf gamgf']);
  
   % ***************************************************************
   % Joint tests -- does gamma prime f really beat slope? Is there more than a
   % few yield spreads can do? 
   % ***************************************************************
   
   
   disp('-------------------------------------------------------');  
   disp('Table 8: Wald tests for parameter restrictions in average return = a + b*f');

   [b_un,olsy,R2hump,R2humpadj,v] = olsgmm(AHPRX, YT,18,1);
   
   fprintf('  right hand variables &%8s &%8s &%8s &%8s &%8s \\\\ \n', '     chi2', '    dof' ,'   prob'); 

   dorest(AHPRX,YT(:,1),YT(:,2:end),'const');
    
   %1
   b_rest = dorest(AHPRX,[YT(:,1) slope],YT(:,3:end),'slope'); 
   b_r = [b_rest(1); b_rest(2)*Q(:,2)]; % implied coefficients on yields. 
  
   %2
   b_rest = dorest(AHPRX,[YT(:,1) level slope],YT(:,4:end),'level, slope'); 
   b_r = [b_r [b_rest(1); b_rest(2)*Q(:,1)+b_rest(3)*Q(:,2)]]; 

  %3
   b_rest = dorest(AHPRX,[YT(:,1) level slope curve],YT(:,5:end),'level, slope, curve'); 
   b_r = [b_r [b_rest(1); b_rest(2)*Q(:,1)+b_rest(3)*Q(:,2)+b_rest(4)*Q(:,3)]]; 

   %4
   b_rest = dorest(AHPRX,[YT(:,1) YT(:,6)-YT(:,2)],YT(:,3:end),'y5-y1'); 
   b_r = [b_r [b_rest(1); -b_rest(2); 0 ; 0; 0; b_rest(2)]]; %
 
   %5
   b_rest = dorest(AHPRX,[YT(:,1) YT(:,2) YT(:,6)],YT(:,3:5),'y1, y5'); 
   b_r = [b_r  [b_rest(1); b_rest(2); 0 ; 0; 0; b_rest(3)]];

   %6
   b_rest = dorest(AHPRX,[YT(:,1) YT(:,5)-YT(:,6)],YT(:,2:5),'y4-y5'); 
   b_r = [b_r [b_rest(1); 0; 0 ; 0; b_rest(2);-b_rest(2)]]; % implied coefficients on yields.

   %7
   b_rest = dorest(AHPRX,[YT(:,1) YT(:,5) YT(:,6)],YT(:,2:4),'y4, y5'); 
   b_r = [b_r [b_rest(1); 0; 0 ; 0; b_rest(2);b_rest(3)]]; % implied coefficients on yields.
   
   %8
   b_rest = dorest(AHPRX,[YT(:,1) YT(:,2) YT(:,5) YT(:,6)],YT(:,3:4),'y1, y4, y5'); 
   b_r = [b_r [b_rest(1); b_rest(2); 0 ; 0; b_rest(3);b_rest(4)]]; % implied coefficients on yields.
 
   %9
   b_rest = dorest(AHPRX,[YT(:,1) YT(:,2) YT(:,5)-YT(:,6)],YT(:,3:5),'y1, y4-y5'); 
   b_r = [b_r [b_rest(1); b_rest(2); 0 ; 0; b_rest(3);-b_rest(3)]]; % implied coefficients on yields.
 
   %10
   b_rest = dorest(AHPRX,YT(:,1:2),YT(:,3:6),'y1 or f1'); 
  % find and plot restricted coefficients separately below; not interesting
  % for yields anyway. 

   %11
   b_rest = dorest(AHPRX,YT(:,1:3),YT(:,4:6),'y1,y2'); 

    %12
   b_rest = dorest(AHPRX,YT(:,1:4),YT(:,5:6),'y1,y2,y3'); 
 
   %13
   b_rest = dorest(AHPRX,YT(:,1:5),YT(:,6),'y1,y2,y3,y4'); 
   
      %10
   b_rest = dorest(HPRX(:,1),YT(:,1:2),YT(:,3:6),'R1 with y1 or f1'); 
  % find and plot restricted coefficients separately below; not interesting
  % for yields anyway. 

   %11
   b_rest = dorest(HPRX(:,1),YT(:,1:3),YT(:,4:6),'R1 with  y1,y2'); 

    %12
   b_rest = dorest(HPRX(:,1),YT(:,1:4),YT(:,5:6),'R1 with y1,y2,y3'); 
 
   %13
   b_rest = dorest(HPRX(:,1),YT(:,1:5),YT(:,6),'R1 with y1,y2,y3,y4'); 

   % *********************************************************************
   % regressions using f1, f1&f2, f1&f2&f3, etc. to see if pattern is
   % stable on including partial right hand variables. 
   % *********************************************************************   
   
   gamma_part = zeros(6,5); 
   gammay_part = zeros(6,5); 
   for i = 1:5; 
       [gamma_part(1:i+1,i),se_part_f,R2_part(i)] = olsgmm(AHPRX,FT(:,1:i+1),18,1);
       [gammay_part(1:i+1,i),se_part_y,R2y_part(i)] = olsgmm(AHPRX,YT(:,1:i+1),18,1);       
   end;
   ols_f = se_part_f; % save last se. Yes it was done up there somewhere, but this way you know it's right. 
   
        
   disp(' '); 
   disp('Regressions on partial right hand variables'); 
   disp('forwards, coeff and R2'); 
   disp([gamma_part' R2_part']); 
   disp('yields, coeff and R2'); 
   disp([gammay_part' R2_part']);      
   disp(' '); 
     
    
   % *************************************************************
   % 4 panel plot of yield factor and its approximations 
   % *************************************************************
   
  figure; 
  
 bigfont = 17; 
 smallfont = 16; % this figure will not reduce as much as the others, so smaller letters
 textfont = 15; 
  subplot(2,2,1);
   plot((1:5),gammasy(2:6),'-ob', ...
        (0:6),[0 0 0 0 0 0 0],'-k','Linewidth',1.5);
   set(gca,'xtick',[1 2 3 4 5],'Fontsize',smallfont);
   axis([0.5 5.5 -12 15]);
   title('A. Expected return factor \gamma*','Fontsize',bigfont);
   xlabel('Yield maturity'); 
   
  subplot(2,2,2);   
    plot(mats,Q(:,1),'-or',mats,Q(:,2),'-vg',mats,Q(:,3),'--sm','Linewidth',1.5);
    axis([0.5 5.5 -0.8 0.8]);

    text(0.8,0.35,'level','Fontsize',textfont);
    text(2.2,-0.25,'slope','Fontsize',textfont);
    text(3.5,-0.6,'curvature','Fontsize',textfont);
    set(gca,'xtick',[1 2 3 4 5],'Fontsize',textfont);
    title('B. Yield factors','Fontsize',bigfont); 
    xlabel('Yield maturity'); 
  
  subplot(2,2,3); 
    plot((1:5)', b_r(2:end,2),'-.vg',...
         (1:5)', b_r(2:end,3),'--sr', ...
         (1:5)', b_un(2:end),'-ob', 'Linewidth',1.5); 
    hold on;
    errorbar((1:5)',b_un(2:end),2*olsy(2:end),'-ob');
    set(gca,'Xtick',[1 2 3 4 5],'Fontsize',textfont); 
    text(1,7.5,'level, slope,','Fontsize',textfont);
    text(1,5,'& curve','Fontsize',textfont);
    text(2.5,-2,'level & slope','Fontsize',textfont);
    title('C. Return Predictions','Fontsize',bigfont);
    xlabel('Yield maturity'); 
    axis([0.5 5.5 -12 15]); 

    subplot(2,2,4); 
     plot((1:1)', gamma_part(2:2,1), '--sc',...
         (1:2)',gamma_part(2:3,2),'--vg',...
        (1:3)',gamma_part(2:4,3),'--^m',...
        (1:4)',gamma_part(2:5,4),'--*r',...
        (1:5)',gamma_part(2:6,5),'-ob','Linewidth',1.5); 
    hold on; 
    errorbar((1:5)',gamma_part(2:6,5),2*ols_f(2:end),'-ob'); 
    title('D. Forward rate forecasts','Fontsize',bigfont); 
    set(gca,'Xtick',[1 2 3 4 5],'Fontsize',smallfont); 
    xlabel('Forward rate maturity'); 
    axis([0.5 5.5 -inf inf]); 
      
   
   if printgraph;
        print -depsc2 fb7.eps;
   end;

      
  % *********************************************************
  % Subsamples 
  % *********************************************************
   
   disp('----------------------------------------------');
   disp('Table 9: Subsample Analysis');

   % many subsamples for regression of average return on lags. 

   obs = [139 T;
          139 T-48;
          139  327;                         % inflation to 79:8
          326  326+39;                      % 79:8 to 82:10
          326+38  T;                        % post inflation
          139 139+72 ;                      % 60s 
          139+72 139+72+120;                % 70s
          139+72+120 139+72+120+120;        % 80s
          139+72+120+120 139+72+120+120+120;% 90s
          139+72+120+120+120 T;             % 2000s
          139+72+120+120+120 T-24;
          139+72+120+120+120+12+12 T];            % 00-03

     obsind = 1; 
     while obsind <= size(obs,1); 
         
         begobs = obs(obsind,1);
         endobs = obs(obsind,2);
         Tl = endobs-begobs;
         
         % on all y and f
         lhv = 100*hprx(begobs+13:endobs,:);
         rhv = 100*[yields(:,1) forwards]; 
         rhv = [ones(Tl-12,1) rhv(begobs+1:endobs-12,:)];
         alhv = mean(lhv')';
         [ab,astb,aR2,aR2adj,v] = olsgmm(alhv,rhv,0,0);
         
         rhv=[ones(length(rhv),1) humpall(begobs+1:endobs-12)]; %
         [abb,astbb,R2sub,R2subadj,vb] = olsgmm(mean(lhv')',rhv,0,0);
         
         disp('data subsample');
         disp([famablisyld(begobs+1,1) famablisyld(endobs,1)]);
         disp('   gamma0    gamma1     gamma2   gamma3    gamma 4    gamma5     R2        bavg        R2' );
         disp([ (ab') (aR2) (abb(2)) (R2sub)]);
         
         obsind = obsind+1;
         
     end;
     
     % **************************************************
     % Comparison with MuCullogh - Kwon data
     % *************************************************
        
    disp('---------------------------------------');   
      
    disp('Table 10: Comparison with McCullogh data');

    dlmread zeroyld1.txt;
    
    z=ans;
    y1=z(3:8:end,4);
    y2=z(4:8:end,[3 5 6 7]);
    mcdate=z(1:8:end,[1 2]);

    dlmread zeroyld2.txt;

    z=ans;
    y1b=z(3:8:end,4);
    y2b=z(4:8:end,[3 5 6 7]);
    mcdateb=z(1:8:end,[1 2]);

    y1=[y1;y1b];
    y2=[y2;y2b];
    mcyields=[y1 y2]/100;
          
    mcprices=-(ones(length(mcyields),1)*mats').*mcyields;
    mcforwards = mcprices(:,1:4)-mcprices(:,2:5);
    mcfs = mcforwards-mcyields(:,1)*ones(1,4);
 
    mchpr = mcprices(13:end,1:4)-mcprices(1:end-12,2:5);
    mchprx = mchpr - mcyields(1:end-12,1)*ones(1,4);
    mchpr = [zeros(12,1)*ones(1,4); mchpr];
    mchprx = [zeros(12,1)*ones(1,4); mchprx];

    
    mcyields=mcyields(140:end-12,:);
    mcforwards=mcforwards(140:end-12,:);
    mcfs = mcfs(140:end-12,:);
    
    mcHPRX = 100*mchprx(140+12:end,:);      
    mcAHPRX = mean(mcHPRX')';
    mcdate=[mcdate;mcdateb];
    mcdate=mcdate(140:end,:);
    Tmc=size(mcyields,1);
    
    mcFS    = [ones(Tmc,1) 100*mcfs];
    mcFT    = [ones(Tmc,1) 100*mcyields(:,1) 100*mcforwards];

    mcyields=mcyields*100;
    
    % Fama-Bliss data over same MK-sample
    
    fbyields=YT(1:Tmc,2:end);
    fbHPRX  =HPRX(1:Tmc,:);
    fbAHPRX =mean(fbHPRX')';
    fbFT    =FT(1:Tmc,:);
    fbFS    =FS(1:Tmc,:);
    
    
    d=zeros(5,1);
    dd=d;
    for i=1:5
        c=corrcoef([fbyields(:,i),mcyields(:,i)]);
        d(i)=c(2,1);
    end
    
    for i=1:5
        c=corrcoef([diff(fbyields(:,i)),diff(mcyields(:,i))]);
        dd(i)=c(2,1);
    end
    disp('Correlation between FB-yields and MC-yields: Levels and Differences');
    disp([d';dd']);

    % not in paper -- plots of fb vs. mc yields to eyeball data and make
    % sure it looks ok. 
    
    figure;

    subplot(5,1,1);
    title('1-year yield');
    plot(dates(beg+1:beg+Tmc), mcyields(:,1)-fbyields(:,1));
    axis([dates(beg+1) dates(beg+Tmc) -1  1]); 
    title('Difference between McCullogh and Fama-Bliss yields');
    
    subplot(5,1,2);
    title('2-year yield');
    plot(dates(beg+1:beg+Tmc), mcyields(:,2)-fbyields(:,2));
    axis([dates(beg+1) dates(beg+Tmc) -1 1]); 

    subplot(5,1,3);
    title('3-year yield');
    plot(dates(beg+1:beg+Tmc), mcyields(:,3)-fbyields(:,3));
    axis([dates(beg+1) dates(beg+Tmc) -1 1]); 

    subplot(5,1,4);
    title('4-year yield');
    plot(dates(beg+1:beg+Tmc), [mcyields(:,4)-fbyields(:,4)]);
    axis([dates(beg+1) dates(beg+Tmc) -1 1]); 

    subplot(5,1,5);
    title('5-year yield');
    plot(dates(beg+1:beg+Tmc), [mcyields(:,5)-fbyields(:,5)]);
    axis([dates(beg+1) dates(beg+Tmc) -1 1]); 

    % run regressions
    
    indx = 1; 
    while indx <= 4;
      
      [bi,HHi,tes,fbR2(indx),v]   = olsgmm(fbHPRX(:,indx),fbFS(:,[1 indx+1]),12,0);
      fbb(indx,:)=bi';
      fbHH(indx,1:2) = HHi';
    
      [bi,HHi,tes,mcR2(indx),v]   = olsgmm(mcHPRX(:,indx),mcFS(:,[1 indx+1]),12,0);
      mcb(indx,:)=bi';
      mcHH(indx,1:2) = HHi';
  
      indx = indx+1;
      
   end;
    
   % make tables 
   
   disp('Fama Bliss Regression Coefficients');
   disp('with Fama Bliss data, 1964-1991');
   disp('hprx(n) (t) = a + b [ f (n-1 -> n) (t) - y(1) (t)]')
   disp(' maturity  coefficients   H-H   R2')
   disp([(2:5)' fbb(:,1:2) fbHH fbR2']);    
    
   disp('with McCullogh data, 1964-1991');
   disp('hprx(n) (t) = a + b [ f (n-1 -> n) (t) - y(1) (t)]')
   disp(' maturity  coefficient   H-H   R2')
   disp([(2:5)' mcb(:,1:2) mcHH mcR2']);
  
   disp('CP - regressions: all forwards');
   
   [fbbetas,fbstb,tes,fbR2,v] = olsgmm(fbHPRX,fbFT,12,0);
   [mcbetas,mcstb,tes,mcR2,v] = olsgmm(mcHPRX,mcFT,12,0);
   
   disp('coefficients R2, se R2 for FB yields');
   disp([fbbetas' fbR2]);
   disp([fbstb' fbR2]);
   
   disp('coefficients R2, se R2 for McCullogh yields');
   disp([mcbetas' mcR2]);
   disp([mcstb' mcR2]);
   
   fbolsse = zeros(10,1);
   [fbgammas,fbolsse(5:end),tes,fbR2hump,v] = olsgmm(fbAHPRX,fbFT,12,0);
   [fbbets,fbolsse(1:4),R2,fbR2hprx,v] = olsgmm(fbHPRX,fbFT*fbgammas,12,0);
   disp('CP Regression coefficients, 1964-1991');
   disp(' gammas, ols se, R2');
   disp([ fbgammas fbolsse(5:end) fbR2hump*ones(6,1)])
   disp('bn, ols se, gmm se, R2');;
   disp([fbbets' fbolsse(1:4) fbR2hprx]);
   
   mcolsse = zeros(10,1);
   [mcgammas,mcolsse(5:end),tes,mcR2hump,v] = olsgmm(mcAHPRX,mcFT,12,0);
   [mcbets,mcolsse(1:4),R2,mcR2hprx,v] = olsgmm(fbHPRX,mcFT*mcgammas,12,0);
   disp('CP Regression coefficients, McCullogh data, 1964-1991');
   disp(' gammas, ols se, R2');
   disp([ mcgammas mcolsse(5:end) mcR2hump*ones(6,1)])
   disp('bn, ols se, R2');;
   disp([mcbets' mcolsse(1:4) mcR2hprx]);
  
   
% **********************************************
% MORE LAGS
% **********************************************


disp('---------------------------------------------------');
   disp('TABLE 11: more lags');

   disp(' "single" regressions with one lag at a time -- not in paper, but useful to see there really is a single factor model with lags'); 
   lags = [ 0 1 2 3 4 5 6 9 12]'; 
   disp('individual bond excess returns on all forward rates, lagged'); 
   disp(' lag (usual = 0)  maturtity   R2 all f   chi2      pval     R2 gammaf    bs');
   for i=1:size(lags,1);     
        disp(' '); 
        [b,stb2,R22,R22adj,vb,F] = olsgmm(HPRX(1+lags(i):end,:),FT(1:end-lags(i),:),18,1); 
        [bg,stb2,R22g,R22adj,vb] = olsgmm(HPRX(1+lags(i):end,:),[ones(Ts-lags(i),1) hump(1:end-lags(i),:)],18,1); 
        disp([lags(i)*ones(4,1) (1:4)' R22 F(:,[1 3]) R22g bg(2,:)']);
    end; 
  
   % same for average (across maturity) returns
   
   disp('Average (across maturity) returns rx bar on lagged forward rates'); 
   disp('     lag      R2 all f   chi2      pval     R2 gammaf    bs');

   lags = [ 0 1 2 3 4 5 6 7 13]'; 
   
   for i = 1:size(lags,1);
       [b,stb2,R22,R22adj,vb,F] = olsgmm(AHPRX(1+lags(i):end,:),FT(1:end-lags(i),:),18,1); 
       [bg,stb2,R22g,R22adj,vb] = olsgmm(AHPRX(1+lags(i):end,:),[ones(Ts-lags(i),1) hump(1:end-lags(i),:)],18,1); 
       bv(:,i) = b; 
       disp([lags(i) R22 F(:,[1 3]) R22g bg(2,:)']);
   end; 

   % plot of single regression lag coefficients
   
   figure; 
   bigfont = 20; 
   smallfont = 20; 
   plot((1:5)',bv(2:end,1),'-bv',...
        (1:5)',bv(2:end,2),'-ro',...
        (1:5)',bv(2:end,3),'-g*',...
        (1:5)',bv(2:end,4),'-k^','Linewidth',2,'MarkerSize',12); 
   %title('Coefficients avg returns on lagged forward rates'); 
   set(gca,'xtick',[1 2 3 4 5],'FontSize',smallfont); 
   legend('i=0','i=1','i=2','i=3');
   xlabel('Maturity'); 
   ylabel('Coefficient'); 
   print -depsc2 single_lag.eps; 
   

   % **********************************************************************************
   % multiple lags. 
   % This section estimates the alpha model. 
   % r(n) = b(n) [gamma'f(t) + gamma' (alpha1 f(t-1) ) + gamma'(alpha2 f(t-2)) + ...]
   % Individual b's are not reported in the paper, but here they are, and they are all the same for different lags  
   % **********************************************************************************

   disp('Regressions with multiple lags at the same time'); 
   disp('Estimates of restricted model'); 
   
   maxlag = 7;
   alpha_v = zeros(maxlag,maxlag); 
   se_alpha_v = ones(maxlag,maxlag); % 1 avoids divide by zero error message when you make t stats. 
   for lags = 1:maxlag;  % number of lags in total to include 1 means just ft
        alphas = zeros(lags,1); 
        alphas(1) = 1; % start with just time t value
  
        %disp('iterating to estimate of a restricted model. watching gamma and alpha in iterations.' ); 
        for iter = 1:10;  % in experiments, 10 was plenty -- no need for fancy iteration schemes. 
            FTfilt = filter(alphas,1,FT); 
            FTfilt = FTfilt(lags:Ts,:); 
   
            [gammas_lags,se_trash,R2_lags] = olsgmm(AHPRX(lags:Ts),FTfilt,18,1); 
            %to debug: disp(gammas_lags'); 
            hump_lags = FT*gammas_lags; % note NOT Ftfilt -- this is not the fitted value; it's the individual terms so you can estimate alpha

            % step 2: estimate alphas
            rhv = [ ones(Ts-lags+1,1) hump_lags(lags:Ts)]; 
            for i = 1:lags-1; 
                rhv = [rhv hump_lags(lags-i:Ts-i)];
            end; 

            [alphas, se_alphas] = olsgmm(AHPRX(lags:Ts),rhv,18,1); 
            % to debug: disp(alphas'); 
            alphas = alphas(2:end); % don't want to use the constant to filter!
            se_alphas = se_alphas(2:end)/sum(alphas);
            alphas = alphas/sum(alphas); % normalize to sum of alphas = 1  

        % GMM standard errors for alpha    
        % moments: OLS for restricted model, 
        %        rxbar = a0*gamma'f_t + a1*gamma'f_t-1+...+ e_t+1
        % moments set to zero: the two regressions. Thus, first the alpha regression moments
        % 0 = E(gamma'f_t * e_t+1) 
        % 0 = E(gamma'f_t-1/12 * e_t+1)
        % 0 = E(gamma'f_t-2/12 * e_t+1) 
        % ...
        % then the gamma regression moments
        % 0 = E((a0*f(1)t + a1*f(1)_t-1/12 + a2*f(1)_t-2/12) * e_t+1) 
        % 0 = E((a0*f(2)t + a1*f(2)_t-1/12 + a2*f(2)_t-2/12) * e_t+1) 
        % ...
        % 
        % Standard error formula: 1/T(d'S-1d)^-1
        % d = d(gT)/d(alpha' gamma'); 
        %     a0              a1                ..     g0         g1
        % d = 0               0                       1*e_t+1    f(1)_t*e_t+1
        %     0               0                       1*e_t+1    f(1)_t-1/12*e_t+1
        %     0               0                 ...   1*e_t+1    f(1)_t-2/12*e_t+1
        %    ...
        %     f(1)_t*e_t+1  f(1)_t-1/12*e_t+1         0           0
        %     f(2)_t+e_t+1  f(2)_t-1/12*e_t+1         0           0 
        %    ...
        % Halelujiah! the d matrix is block diagonal so that alpha standard errors are not affected by gamma estimation
        
            
        end;
        
        gamma_v(:,lags) = gammas_lags; % keep track for printing
        alpha_v(1:lags,lags) = alphas; 
        se_alpha_v(1:lags,lags) = se_alphas; 
        R2_v(lags) = R2_lags; 
        hump_lags_fit = FTfilt*gammas_lags; % this one IS the fitted value, used next 
        
        % now, bs for the individual maturities
        
        [betas_lags,se_betas_lags,R2_betas_lags] = olsgmm(HPRX(lags:Ts,:),hump_lags_fit,12,0); 
        
        betas_lags_v(:,lags) = betas_lags'; 
        se_betas_lags_v(:,lags) = se_betas_lags';
        R2_betas_lags_v(:,lags) = R2_betas_lags; % maturities up and down, lags sideways
        
        % now, totally unrestricted regressions for comparison
        
        rhv = FT(lags:Ts,:); 
        for i = 1:lags-1; 
            rhv = [rhv FT(lags-i:Ts-i,2:end)];
        end; 
        
        [bet_unrest_lags,se_trash,R2_betas_unrest_lags] = olsgmm(HPRX(lags:Ts,:),rhv,12,0); 
        R2_betas_unrest_lags_v(:,lags) = R2_betas_unrest_lags;
        
        [gammas_unrest_lags,se_trash,R2_gammas_unrest_lags] = olsgmm(AHPRX(lags:Ts,:),rhv,12,0); 
        R2_gammas_unrest_lags_v(lags) = R2_gammas_unrest_lags;
        
        
    end; 
    
    disp('gammas'); 
    disp('\begin{tabular}{ccccccc}'); 
    fprintf('  %8s & %8s & %8s & %8s & %8s & %8s & %8s \\\\ \n','lags','const','y1','f2','f3','f4','f5'); 
    fprintf('  %8.2f & %8.2f & %8.2f & %8.2f & %8.2f & %8.2f & %8.2f \\\\ \n',[(1:maxlag); gamma_v]); 
    disp('\end{tabular}'); 
    
    disp('alphas '); 
    disp('\begin{tabular}{cccccccc}');    
    fprintf('  %8s & %8s & %8s & %8s & %8s & %8s & %8s & %8s \\\\ \n','lags','t','t-1/12','t-2/12','t-3/12','t-4/12','t-5/12','t-6/12'); 
    fprintf('  %8.2f & %8.2f & %8.2f & %8.2f & %8.2f & %8.2f & %8.2f & %8.2f  \\\\ \n',[(1:maxlag); alpha_v]); 
    disp('\end{tabular}'); 
    
    disp('alpha estimates and t statistics '); 
    disp('\begin{tabular}{cccccccc}');    
    fprintf('  %8s & %8s & %8s & %8s & %8s & %8s & %8s & %8s \\\\ \n','lags','t','t-1/12','t-2/12','t-3/12','t-4/12','t-5/12','t-6/12'); 
    for i = 1:7'; 
        fprintf('  %8.2f & %8.2f & %8.2f & %8.2f & %8.2f & %8.2f & %8.2f & %8.2f  \\\\ \n',[i alpha_v(:,i)']);
        fprintf('  %8.2f & (%8.2f) & (%8.2f) & (%8.2f) & (%8.2f) & (%8.2f) & (%8.2f) & (%8.2f)  \\\\ \n',[i alpha_v(:,i)'./se_alpha_v(:,i)']);
    end; 
    disp('\end{tabular}'); 
    
    
    disp('b loadings'); 
    disp('\begin{tabular}{cccccccc}');    
    fprintf('      %8s & %8s & %8s & %8s & %8s & %8s & %8s \\\\ \n','0','1','2','3','4','5','6');
    fprintf(' b(2) & %8.2f & %8.2f & %8.2f & %8.2f & %8.2f & %8.2f & %8.2f   \\\\ \n',betas_lags_v(1,:));
    fprintf(' b(3) & %8.2f & %8.2f & %8.2f & %8.2f & %8.2f & %8.2f & %8.2f   \\\\ \n',betas_lags_v(2,:));
    fprintf(' b(4) & %8.2f & %8.2f & %8.2f & %8.2f & %8.2f & %8.2f & %8.2f   \\\\ \n',betas_lags_v(3,:));
    fprintf(' b(5) & %8.2f & %8.2f & %8.2f & %8.2f & %8.2f & %8.2f & %8.2f   \\\\ \n',betas_lags_v(4,:));
    disp('\end{tabular}');
    
    disp('b standard errors'); 
    disp('\begin{tabular}{cccccccc}');    
    fprintf('      %8s & %8s & %8s & %8s & %8s & %8s & %8s \\\\ \n','0','1','2','3','4','5','6');
    fprintf(' b(2) & %8.2f & %8.2f & %8.2f & %8.2f & %8.2f & %8.2f & %8.2f   \\\\ \n',se_betas_lags_v(1,:));
    fprintf(' b(3) & %8.2f & %8.2f & %8.2f & %8.2f & %8.2f & %8.2f & %8.2f   \\\\ \n',se_betas_lags_v(2,:));
    fprintf(' b(4) & %8.2f & %8.2f & %8.2f & %8.2f & %8.2f & %8.2f & %8.2f   \\\\ \n',se_betas_lags_v(3,:));
    fprintf(' b(5) & %8.2f & %8.2f & %8.2f & %8.2f & %8.2f & %8.2f & %8.2f   \\\\ \n',se_betas_lags_v(4,:));
    disp('\end{tabular}');
    
    
    disp('R2 -- restricted model');
    disp('\begin{tabular}{cccccc}');    
    fprintf('      %8s & %8s & %8s & %8s & %8s & %8s \\\\ \n','lags','rx2','rx3','rx4','rx5','rxbar');
    fprintf('      %8.2f & %8.2f & %8.2f & %8.2f & %8.2f & %8.2f   \\\\ \n',[(1:maxlag)' R2_betas_lags_v' R2_v']'); 
    disp('\end{tabular}');

    disp('\begin{tabular}{cccccc}');    
    disp('R2 -- unrestricted model'); 
    fprintf('      %8s & %8s & %8s & %8s & %8s & %8s \\\\ \n','lags','rx2','rx3','rx4','rx5','rxbar');
    fprintf('      %8.2f & %8.2f & %8.2f & %8.2f & %8.2f & %8.2f  \\\\ \n',[(1:maxlag)'  R2_betas_unrest_lags_v' R2_gammas_unrest_lags_v']'); 
    disp('\end{tabular}');
   
    
    % **********************************************************
    % Create generic "hump" factor using three lags for use in 
    % other exercises
    % **********************************************************
    
    % New: use filter with 3 lags for real time forecasts, history, etc. 
     
     forwards_filt = filter(alpha_v(1:3,3),1,100*[yields(:,1) forwards]);  % full sample
     humpall_filt = [ones(size(forwards_filt,1),1) forwards_filt]*gamma_v(:,3); % this one IS the fitted value, used next 
                  % did not use FTall as filter would give 1/3 2/3 1 1 1 for constant
     humpall_filt = humpall_filt(beg:end);                                                                             
     hump_filt = humpall_filt(1:end-12); % matches hump and FT                                                                             
                                                                           
     
      % The following regressions do not appear in the paper. HOwever they document
      % the statements that extra months appear with tent shapes in multiple regressions
      % so leave them in the program so critics can see them. 
   
   % an extra month? 
   rhv = [ones(Ts-1,1) yields(beg+1:T-12,1) forwards(beg+1:T-12,:)...
                       yields(beg:T-13,1) forwards(beg:T-13,:)];
   rhv = [rhv(:,1) rhv(:,2:end)*100];
   lhv = 100*hprx(beg+13:T,:);
   [b2,stb2,R22,R22adj,v] = olsgmm(lhv,rhv,12,0); 
     
   % sum of first and second month?
   rhv = [ones(Ts-1,1) (yields(beg+1:T-12,1)+yields(beg:T-13,1))/2 ...
                           (forwards(beg+1:T-12,:)+forwards(beg:T-13,:))/2 ];
   rhv = [rhv(:,1) 100*rhv(:,2:end)];
   [b3,stb3,R23,R23adj,v] = olsgmm(lhv,rhv,12,0);
  

   % make plot of first two lags
  
   figure;
   subplot(3,1,1)
   plot((1:5),b2(2:6,1),'-ok',...
		   (1:5),b2(2:6,2),'-vb',...
		   (1:5),b2(2:6,3),'-sg',...
         (1:5),b2(2:6,4),'-dm');
   legend('2','3','4','5');
   title('Current');
   set(gca,'xtick',[1 2 3 4 5]);
   axis([1 5 -3 3]);
  
   subplot(3,1,2)
   plot((1:5),b2(7:11,:));
   plot((1:5),b2(7:11,1),'-ok',...
		    (1:5),b2(7:11,2),'-vb',...
		   (1:5),b2(7:11,3),'-sg',...
         (1:5),b2(7:11,4),'-dm');
   title('One month lag');
   set(gca,'xtick',[1 2 3 4 5]);
   axis([1 5 -3 3]);
   legend('2','3','4','5');
  
   subplot(3,1,3)
   plot((1:5),b3(2:6,:));
   plot((1:5),b3(2:6,1),'-ok',...
		    (1:5),b3(2:6,2),'-vb',...
		   (1:5),b3(2:6,3),'-sg',...
         (1:5),b3(2:6,4),'-dm');
   title('Average of current and one month lag');
   set(gca,'xtick',[1 2 3 4 5]);
   axis([1 5 -4.6 5]);
   legend('2','3','4','5');
  
 
         
  % two extra months?
  
   rhv = [ones(Ts-2,1) yields(beg+2:T-12,1) forwards(beg+2:T-12,:)...
                           yields(beg+1:T-13,1) forwards(beg+1:T-13,:)...
                           yields(beg  :T-14,1) forwards(beg  :T-14,:)];
   rhv = [rhv(:,1) rhv(:,2:end)*100];
   lhv = 100*hprx(beg+14:T,:);
   [b2,stb2,R24,R24adj,v] = olsgmm(lhv,rhv,12,0);
   
   disp('maturity, R2 from [ft, ft-1/12], [(ft+ft-1/12)/2], [ft, ft-1/12, ft-2/12]');
   disp([(1:4)' (R22) (R23) (R24)]);
  
   
   figure;
   subplot(3,1,1)
   plot((1:5),b2(2:6,:));
   title('Returns on all yields and forwards and an extra month lag');
   set(gca,'xtick',[1 2 3 4 5]);
   axis([1 5 -3 3]);
  
   subplot(3,1,2)
   plot((1:5),b2(7:11,:));
   title('One month lag');
   set(gca,'xtick',[1 2 3 4 5]);
   axis([1 5 -3 3]);
  
   subplot(3,1,3);
   plot((1:5),b2(12:16,:));
   title('two month lag');
   set(gca,'xtick',[1 2 3 4 5]);
   axis([1 5 -3 3]);

   
% ***************************************
% Measurement error simulation. 
% **************************************************
   
% sim_meas_error. Shows what patterns of coefficients can be induced by measurement error in prices. 
% these do not go in paper -- they document that the patterns hard wired
% below are correct. 

Tsim = 50000;
Tsim = 10; 
if T < 50000; 
    disp('SIMULATION LENGTH FOR MEASUREMENT ERROR IS SET WAY SHORT SO PROGRAM WILL RUN QUICKLY.') 
end; 
cormx = rand(5,5); % induces random correlation between variables at time t. Note this typically induces weird behavior in small s
% samples because of collinearity in right hand variables. Increase simulation size a lot to see the right pattern. 
cormx = eye(5);  % this version has iid across assets too. 
disp('simulating measurment error'); 
disp('covariance matrix of measurement error across maturity'); 
disp(cormx*cormx'); 

p = randn(Tsim,5)*cormx; % i.i.d. across time measurement error
f = [-p(:,1) p(:,1)-p(:,2) p(:,2)-p(:,3) p(:,3)-p(:,4) p(:,4)-p(:,5)]; 
y = [-p(:,1) -p(:,2)/2 -p(:,3)/3 -p(:,4)/4 -p(:,5)/5]; 
ft = [ones(Tsim-1,1) f(1:end-1,:)];
yt = [ones(Tsim-1,1) y(1:end-1,:)];
rx = p(2:end,1:end-1) - p(1:end-1,2:end) + p(1:end-1,1)*ones(1,4); 

[b,stb,R2,R2adj,v,F] = olsgmm(rx,ft,18,1); 

figure; 
subplot(2,2,1); 
plot((1:5)',b(2:end,:),'-v'); 
set(gca,'xtick',[1 2 3 4 5]); 
%set(gca,'ytick',[0 1]); 
%axis([ 0.5 5.5 -0.2 1.2]); 
title('forward coeffs, error in prices'); 

[b,stb,R2,R2adj,v,F] = olsgmm(rx,yt,18,1); 

subplot(2,2,2);
plot((1:5)',b(2:end,:),'-v'); 
set(gca,'xtick',[1 2 3 4 5]); 
%set(gca,'ytick',[0 1]); 
%axis([ 0.5 5.5 -0.2 1.2]); 
title('yield coeffs, error in prices'); 


% check same thing with noise in forward rates not prices 
f = randn(Tsim,5)*cormx; % i.i.d. one year yield and 5 forward rates. 
p = -cumsum(f')';  
 %    p(1) = -f(1)
 %    p(2) = p(1) + p(2) - p(1) = - f(1) - f(2) etc. 
 
%f = [-p(:,1) p(:,1)-p(:,2) p(:,2)-p(:,3) p(:,3)-p(:,4) p(:,4)-p(:,5)]; reproduces the original f, a good check!
ft = [ones(Tsim-1,1) f(1:end-1,:)];
rx = p(2:end,1:end-1) - p(1:end-1,2:end) + p(1:end-1,1)*ones(1,4); 
y = [-p(:,1) -p(:,2)/2 -p(:,3)/3 -p(:,4)/4 -p(:,5)/5]; 
yt = [ones(Tsim-1,1) y(1:end-1,:)];

[b,stb,R2,R2adj,v,F] = olsgmm(rx,ft,18,1); 

subplot(2,2,3); 
plot((1:5)',b(2:end,:),'-v'); 
set(gca,'xtick',[1 2 3 4 5]); 
%set(gca,'ytick',[0 1]); 
%axis([ 0.5 5.5 -0.2 1.2]); 
title('forward coeffs, error in forwards'); 

[b,stb,R2,R2adj,v,F] = olsgmm(rx,yt,18,1);

subplot(2,2,4);
plot((1:5)',b(2:end,:),'-v'); 
set(gca,'xtick',[1 2 3 4 5]); 
%set(gca,'ytick',[0 1]); 
%axis([ 0.5 5.5 -0.2 1.2]); 
title('yield coeffs, error in forwards'); 


% Now figures for paper that hard wire the patterns seen above. 

b = [ 0 0 0 0 ; 
      1 1 1 1 ; 
      0 1 1 1 ; 
      0 0 1 1 ; 
      0 0 0 1 ];
       
% Figure for paper
  
figure; 
bigfont = 20; 
smallfont = 18; 
subplot(2,1,1); 
plot((1:5)',b(:,4)+0.6,'-vb',...
     (1:5)',b(:,3)+0.4,'-^g',... 
     (1:5)',b(:,2)+0.2,'-or',... 
     (1:5)',b(:,1)+0.0,'-*m','Linewidth',2,'MarkerSize',12); 
 set(gca,'FontSize',smallfont); 
 %legend('n=5','n=4','n=3','n=2',2); 
 hold on; 

 plot([0.5 5.5]',zeros(2,1),'--k',...
     [0.5 5.5]',zeros(2,1)+0.2,'--k',... 
     [0.5 5.5]',zeros(2,1)+0.4,'--k',... 
     [0.5 5.5]',zeros(2,1)+0.6,'--k'); 
set(gca,'xtick',[1 2 3 4 5]); 
set(gca,'ytick',[0 0.2 0.4 0.6 1 1.2 1.4 1.6]); 
set(gca,'yticklabel',[ 0 0 0 0 1 1 1 1]); 
axis([ 0.5 5.5 -0.2 1.8]); 
ylabel('Coefficient'); 
title('Forward rates','Fontsize',bigfont); 

b = [ -1 -1 -1 -1 ; 
       2  0  0  0 ; 
       0  3  0  0 ; 
       0  0  4  0 ; 
       0  0  0  5 ];

subplot(2,1,2); 
plot((1:5)',b(:,4)+0.0,'-vb',...
     (1:5)',b(:,3)+0.0,'-^g',... 
     (1:5)',b(:,2)+0.0,'-or',... 
     (1:5)',b(:,1)+0.0,'-*m','Linewidth',2,'MarkerSize',12); 
set(gca,'FontSize',smallfont); 
 legend('n=5','n=4','n=3','n=2','Location','NorthWest'); 
 hold on; 

 plot([0.5 5.5]',zeros(2,1),'--k');
     %,[0.5 5.5]',zeros(2,1)+0.2,'--k',... 
     %[0.5 5.5]',zeros(2,1)+0.4,'--k',... 
     %[0.5 5.5]',zeros(2,1)+0.6,'--k'...

set(gca,'xtick',[1 2 3 4 5]); 
%set(gca,'ytick',[0 0.2 0.4 0.6 1 1.2 1.4 1.6]); 
%set(gca,'yticklabel',[ 0 0 0 0 1 1 1 1]); 
set(gca,'ytick',[-1 0 1 2 3 4 5 ]); 

axis([ 0.5 5.5 -1.2 5.2]); 
xlabel('Maturity'); 
ylabel('Coefficient');
title('Yields','Fontsize',bigfont); 

print -depsc2 sim_meas_err.eps; 


% ***********************************************************
% Implied annual var from monthly var. Compare predictability
% ***********************************************************


  disp('-------------------------------------------------------'); 
  disp('Compare annual yield var and monthly var to 12th power');  
    
  % direct regression on yields to check direct annual var
   [by,stby,R2y,R2yadj,v] = olsgmm(HPRX,YT,12,0);
   disp('b and r2 from regression of excess returns on yields, annual horizon');
   disp([by' R2y]); 
  
   % direct annual var
    % reminder:   YT   = [ones(Ts,1) 100*yields(beg:T-12,:)];  % all yields
   rhv   = YT; 
   lhv =  100*yields(beg+12:T,:); 
   [by,stby,R2y,R2yadj,vy] = olsgmm(lhv,rhv,12,0);
   erry = lhv-rhv*by;
   Aa = by(2:end,:)'; 
   
   % from yields to excess returns. 
   % rx(n)_t+1 = p(n-1)_t+1-p(n)_t-y(1)_t
   %           = -(n-1)y(n-1)+t+1 + (n)y(n)_t - y(1)_t
   % rx(2)       [ -1  0  0  0  0 ]  [y(1)]       [-1 2 0 0 0]  y_t   
   % rx(3)  =    [  0 -2  0  0  0 ]  [y(2)]       [-1 0 3 0 0]
   % rx(4)       [       -3  0  0 ]  [y(3)]     + [-1 0 0 4 0]
   % rx(5)       [          -4  0 ]  [y(4)]       [-1 0 0 0 5]
%                                    [y(5)]_t+1   

% or, rx_t+1 =  M y_t+1 + N y_t 

% Now if y_t+1 = Aay_t + d_t+1, 
% then rx_t+1 = ( M Aa + N ) y_t - Md_t+1

% we can transform the right hand variable to forward rates too
% f(n-1 -> n) = p(n-1)-p(n) = ny(n) - (n-1)y(n-1)
% y(1)          [  1  0  0  0  0 ] [y(1)]
% f(1->2)       [ -1  2  0  0  0 ] [y(2)]
% f(2->3)  =    [  0 -2  3  0  0 ] [y(3)]
% f(3->4)       [  0  0 -3  4  0 ] [y(4)]
% f(4->5)       [  0  0  0 -4  5 ] [y(5)]

% f = P y
% then rx_t+1 = (M Aa + N) inv(P) f

  M = [ -1  0  0  0  0 ; 
         0 -2  0  0  0 ;
         0  0 -3  0  0 ; 
         0  0  0 -4  0 ]; 
  N = [  -1 2 0 0 0;
         -1 0 3 0 0; 
         -1 0 0 4 0;
         -1 0 0 0 5]; 
  P = [  1  0  0  0  0 ;
        -1  2  0  0  0 ;
         0 -2  3  0  0 ;
         0  0 -3  4  0 ; 
         0  0  0 -4  5 ]; 
  implb = M*Aa + N; 
  implbf = implb*inv(P); 
  disp('return forecast coefficients on Y, then F and r2 implied by direct annual var'); 
  r2 = 1-(std(HPRX-rhv(:,2:end)*implb').^2)./(std(HPRX).^2);
  disp([implb r2']); 
  disp([implbf r2']);   

  
  % monthly var, find implied annual. 
   rhv   = YT; 
   lhv =  100*yields(beg+1:T-11,:); 
   [bym,stbym,R2ym,R2ymadj,vym] = olsgmm(lhv,rhv,12,0);
   errym = lhv-rhv*bym;
   A = bym(2:end,:)'; 
   Ay = A^12; 
   implb = M*Ay + N; 
   disp('return forecast coefficients and r2 implied by implied annual var from monthly'); 
   r2 = 1-(std(HPRX-rhv(:,2:end)*implb').^2)./(std(HPRX).^2);
   disp([implb r2']); 
   implbf = implb*inv(P); 
   disp([implbf r2']); 
   
   % regression of average returns is average of individual regressions rx(n) = g(n)*f+e(n) -> rxbar = gbar*f+ebar, ebar is orthogonal to
   % f, so this is the regression.
   implbavg = mean(implb); 
   implbfavg = mean(implbf); 
   
   figure; 
   plot(implbavg'); 
   title('implied average returns on yields'); 
   figure; 
   plot(implbfavg'); 
   title('implied average returns on forwards');    
   
   r2implavg = 1-(std(AHPRX-rhv(:,2:end)*implbavg').^2)./(std(AHPRX).^2);
   disp('R2 of annual average returns rxbar on forwards implied by monthly var(1)'); 
   disp(r2implavg);

% figure for paper

figure; 
bigfont = 20; 
smallfont = 18; 
textfont = 18; 
subplot(2,1,1); 
plot((1:5),implbf(1,:)+0,'-vb',...
     (1:5),implbf(2,:)+0,'--^g',... 
     (1:5),implbf(3,:)+0,'-or',... 
     (1:5),implbf(4,:)+0.0,'--*m','Linewidth',2,'MarkerSize',12); 
% legend('n=2','n=3','n=4','n=5',2); 
text(1.5,implbf(1,2),'n=2','color','b','FontSize',textfont);
text(2.5,implbf(2,3),'n=3','color','g','FontSize',textfont); 
text(4.2,implbf(3,4),'n=4','color','r','FontSize',textfont);
text(5.1,implbf(4,5),'n=5','color','m','FontSize',textfont); 
 hold on; 

 plot([0 6]',zeros(2,1)+0,'--k',...
     [0 6]',zeros(2,1)+0,'--k',... 
     [0 6]',zeros(2,1)+0,'--k',... 
     [0 6]',zeros(2,1)+0,'--k');
set(gca,'FontSize',smallfont); 
set(gca,'xtick',[1 2 3 4 5]); 
axis([ 0.5 5.5 -1.5 1.2]); 
%xlabel('Maturity'); 
ylabel('Coefficient');
title('Forward rates','FontSize',bigfont); 

subplot(2,1,2); 
plot((1:5),implb(1,:)+0,'-vb',...
     (1:5),implb(2,:)+0,'--^g',... 
     (1:5),implb(3,:)+0,'-or',... 
     (1:5),implb(4,:)+0.0,'--*m','Linewidth',2,'MarkerSize',12); 
% legend('n=2','n=3','n=4','n=5',2); 
text(1.5,implb(1,2),'n=2','color','b','FontSize',textfont);
text(2.5,implb(2,3),'n=3','color','g','FontSize',textfont); 
text(4.2,implb(3,4),'n=4','color','r','FontSize',textfont);
text(5.1,implb(4,5),'n=5','color','m','FontSize',textfont); 
 hold on; 

plot([0 6]',zeros(2,1)+0,'--k',...
     [0 6]',zeros(2,1)+0,'--k',... 
     [0 6]',zeros(2,1)+0,'--k',... 
     [0 6]',zeros(2,1)+0,'--k');
set(gca,'FontSize',smallfont); 
set(gca,'xtick',[1 2 3 4 5]); 
axis([ 0.5 5.5 -2.2 5.2]); 
xlabel('Maturity'); 
ylabel('Coefficient');
title('Yields','Fontsize',bigfont); 

print -depsc2 ar1yields.eps; 


% ******************************************
% Real time vs. full sample calculations
% ********************************************
   
disp('--------------------------------------------------------');
disp(' Out of sample exercise');

% forecasts of average hprx using entire sample 

 cpavin = hump;   
 cpavin_filt = hump_filt; % version with filtered rhs
 fbavin = 0 ; 
 for j=1:4; 
     fbavin = fbavin + FS(:,[1 j+1])*bFB(j,:)'/4;
 end;

 
 for t = 5*12:length(AHPRX); 
     
        % CP with restriction 
        % hprx from 65:1 to 65:1+t-12 using forwards from 64:1 to 64:1+t-12
        % at time t you only can use forward data up to time t-12; as you have just seen return t
        % AHPRX at t is observation t-12 -- AHPRX and FT are shifted so forecast and actual have same index
        [b,stb,R2b,R2bb,v] = olsgmm(AHPRX(1:t-12,:), FT(1:t-12,:), 12,0);
        
        % then first forecast uses forwards one year later (same as the last AHPRX observation_ at 64:1+t-12+12 to forecast ahead   
        humpout(t,:) = FT(t,:)*b;  
        cpavout(t,:) = FT(t,:)*b;
        % this is the forecast for AHPRX in position t  
  
        % Filtered CP with restriction -- code copied from above with lags = 3. 
        
        lags = 3; 
        alphas = zeros(lags,1); 
        alphas(1) = 1; % start with just time t value
  
        for iter = 1:10; 
            FTfilt = filter(alphas,1,FT); 
            %FTfilt = FTfilt(lags:Ts,:); Didn't bother with this to keep code simpler. 
   
            [gammas_lags,se_trash,R2_lags] = olsgmm(AHPRX(1:t-12),FTfilt(1:t-12,:),18,1); 
            %to debug: disp(gammas_lags'); 
            hump_lags = FT*gammas_lags; 
            % note NOT Ftfilt -- this is not the fitted value; it's the individual terms so you can estimate alpha

            
            % step 2: estimate alphas
            rhv = [ ones(t-lags+1,1) hump_lags(lags:t)]; 
            for i = 1:lags-1; 
                rhv = [rhv hump_lags(lags-i:t-i)];
            end; 

            
            [alphas, se_alphas] = olsgmm(AHPRX(lags:t),rhv,18,1); 
            % to debug: disp(alphas'); 
            alphas = alphas(2:end); % don't want to use the constant to filter!
            alphas = alphas/sum(alphas); % normalize to sum of alphas = 1  
        end;         
        
        humpout_filt(t,:) = FTfilt(t,:)*gammas_lags;  
        cpavout_filt(t,:) = FTfilt(t,:)*gammas_lags;
        
                
        % FB
        %[b,stb,R2b,R2bb,v] = olsgmm(HPRX(1:t-12,pick),FS(1:t-12,[1 pick+1]),12,0);
        %fbout(t,:) = FS(t,[1 pick+1])*b;

        % FB for all maturities to forecast equally weighted portfolio
        fbavout(t,:) = 0; 
        for j=1:4; 
            [b,stb,R2b,R2bb,v] = olsgmm(HPRX(1:t-12,j),FS(1:t-12,[1 j+1]),12,0);
            fbavout(t,:) =  fbavout(t,:) + FS(t,[1 j+1])*b;
        end; 
        fbavout(t,:) = fbavout(t,:)/4; 
        
        % CP unrestricted for the picked maturity
        %[b,stb,R2b,R2bb,v] = olsgmm(HPRX(1:t-12,pick), FT(1:t-12,:), 12,0);
        %cpurout(t,:) = FT(t,:)*b; 

 end; 
 
 st = 120; % start after 10 years
 
 % characterize forecasts by mean squared error, etc. 
 % decided numbers aren't that interesting, average over long time of
 % learning 
 
 sdrx = std(AHPRX(st:end)); % std dev of ex post returns 
 
 sdcpin =  std(cpavin_filt(st:end));     % sigma rhs full sample
 sdcpout =  std(cpavout_filt(st:end));  % sigma rhs full sample  
 
 sdfbin =  std(fbavin(st:end));     
 sdfbout =  std(fbavout(st:end));  

 mserx = mean(AHPRX(st:end).^2)^0.5; 
 
 msecpout = sqrt(mean(AHPRX(st:end)-cpavout_filt(st:end)).^2); %mse of CP  real time 
 msecpin =  sqrt(mean(AHPRX(st:end)-cpavin_filt(st:end)).^2);  % mse of CP full sample

 msefbout = sqrt(mean(AHPRX(st:end)-fbavout(st:end)).^2); %mse of fb  real time 
 msefbin =  sqrt(mean(AHPRX(st:end)-fbavin(st:end)).^2);  % mse of fb full sample
 
 disp('root mean squared errors') ; 
 disp('CP real time and full sample'); 
 disp([msecpout msecpin]); 
 disp('FB real time and full sample'); 
 disp([msefbout msefbin]); 
 disp('sd of actual returns, cp full, cp realtime, fb full, fb realtime'); 
 disp([sdrx sdcpin sdcpout sdfbin sdfbout]); 
 
   
 % plot real time vs. end of sample forecasts for average across maturities
 % this is a plot of just these; no longer used in paper -- real time is subsumed in history graph. 
 % if you delete history graph or otherwise modify it, you can bring this
 % back. 
 
figure;
plot(dates(beg+12:end), cpavin_filt+10,'r', dates(beg+12:end), cpavout_filt, 'g','Linewidth',1.5); 
axis([1970,2004,-inf,inf]);
text(1994,15,'Full sample','color','r');
text(1994,5,'Real time','color','g');
hold on; 
plot(dates(beg+12:end), ones(length(cpavin_filt),1)*[0 10],'k'); 
set(gca,'Ytick',[-5:5:20]); 
set(gca,'Yticklabel','-5|0|5,-5|0|5|10');

if printgraph;
print -depsc2 fb5.eps;
end



% same for fama bliss -- they look pretty good too. Not in paper. 

figure;
plot(dates(beg+12:end), [fbavin+7 fbavout]); 
legend('full sample','real time'); 
title('FB forecasts, full sample vs. real time');
axis([1970,2004,-inf,inf]);

% ************************************************
% trading rules for average across maturities
% invest 1 x Et(hprx) long and the same short 
% return is hprx_t+1 times Et(hprx) (ignoring logs vs. levels)
% plot returns

% MONIKA: Im not totally sure we shouldn't show this one -- non cumulated returns --  instead. Opinions? 
% *************************************************

% trading rule profits

figure; 
plot(dates(beg+12:end), (cpavin_filt.*AHPRX)+50,'r',dates(beg+12:end),(cpavout_filt.*AHPRX),'g','Linewidth',1.5);
hold on; 
%legend('Full sample','Real time'); 
text(1972, 25, 'Real Time'); 
text(1972, 100, 'Full Sample'); 
plot([1970; 2004],[0 50; 0 50],'k'); 
axis([1970,2004,-inf,inf]);
print -depsc2 tradrule1.eps; 


% cumulated trading rule profits. 



% a blowup of the early part of the sample. TOo bad we can't show 1000 pictures
figure; 
series =  [cpavin_filt.*AHPRX cpavout_filt.*AHPRX fbavin.*AHPRX fbavout.*AHPRX];
series = cumsum(series(121:end,:));
plot(dates(beg+12+120:beg+12+120+119), series(1:120,1),'-r','Linewidth',1.5);
hold on; 
plot(dates(beg+12+120:beg+12+120+119), series(1:120,2),'-g','Linewidth',1.5);
plot(dates(beg+12+120:12:beg+12+120+119), series(1:12:120,2),'vg','Linewidth',1.5);
plot(dates(beg+12+120:beg+12+120+119), series(1:120,3),':r','Linewidth',1);
plot(dates(beg+12+120:beg+12+120+119), series(1:120,4),':g','Linewidth',1);
plot(dates(beg+12+120:12:beg+12+120+119), series(1:12:120,4),'vg','Linewidth',1.5);
%title('Cumulative trading rule profits');
text(1988,2300,'CP full sample');
text(1996,1650,'CP real time'); 
text(1996,870,'FB full sample');
text(1996,480,'FB real time'); 
%legend('CP full sample','CP real time', 'FB full sample', 'FB real time');
axis([1975,1985,-inf,600]);

figure; 
series =  [cpavin_filt.*AHPRX cpavout_filt.*AHPRX fbavin.*AHPRX fbavout.*AHPRX];
series = cumsum(series(121:end,:));
plot(dates(beg+12+120:end), series(:,1),'-r','Linewidth',1.5);
hold on; 
plot(dates(beg+12+120:end), series(:,2),'-r','Linewidth',1.5);
plot(dates(beg+12+120:12:end), series(1:12:end,2),'vr','Linewidth',1.5);
plot(dates(beg+12+120:end), series(:,3),':g','Linewidth',1);
plot(dates(beg+12+120:end), series(:,4),':g','Linewidth',1);
plot(dates(beg+12+120:12:end), series(1:12:end,4),'vg','Linewidth',1.5);
%title('Cumulative trading rule profits');
text(1988,2300,'CP full sample','color','r');
text(1996,1650,'CP real time','color','r'); 
text(1996,870,'FB full sample','color','g');
text(1996,480,'FB real time','color','g'); 
%legend('CP full sample','CP real time', 'FB full sample', 'FB real time');
axis([1975,2004,-50,3300]);
set(gca,'XTick',[1965 1970 1975 1980 1985 1990 1995 2000]); 
    
print -depsc2 fb6.eps; 


    % ******************************
    % History. Compare FB, hump, with ex post returns. 
    % *******************************
   
    % Old: 
   % use MA(2)  on right hand side for other exercises
  %     FTMA = filter(ones(4,1)/4,1,FT); % create moving average of FT on rhs
  %     humpma = filter(ones(4,1)/4,1,hump); 
  %     [b2,stb2,R22] = olsgmm(AHPRX(4:end,:),FTMA(4:end,:),18,1); 
  %     hump_filt = FTMA(4:end,:)*b2; 
       
      % average of FB for comparison 
     FTall = [ones(size(yields,1),1) 100*yields(:,1) 100*forwards] ; 
     FTall = FTall(beg:end,:); % FT stops one year short, this one goes ot end of sample

      clear fitfb2;
       for i = 1:4; 
        [b2,stb2,R22] = olsgmm(HPRX(:,i),[FT(:,1) FT(:,i+2)-FT(:,2)],18,1); 
        fitfb2(:,i) = [FTall(:,1) FTall(:,i+2)-FTall(:,2)]*b2; 
       end; 
       fitfb2 = mean(fitfb2')'; 
        
      spdates = [1982.00;... %special dates to investigate
                 1983.33;...
                 1985.67;...
                 1992.33;...
                 1993.67;...
                 2002.17;...
                 2004.00];
      for i = 1:size(spdates,1);
          [C, spdatindx(i)] = min(abs(spdates(i)-dates)); % finds index of special dates
      end; 
      % check: [ dates(spdatindx)      spdates]
       
       % plot of history over time
        
      figure;
      smallfont = 18; 
      ds=dates(beg:T);
      z = zeros(length(ds),1);
      plot(ds,fitfb2+15,'b',...  %dates(beg:end-12), cpavout_filt+10,'m',... deleted real time graph
           ds,humpall_filt,'r',...
           ds(1:end-12),AHPRX-15,'g','Linewidth',1.5);
      hold on; 
      plot(ds,100*yields(beg:end,:)-40,'m');
      plot(ds,100*yields(beg:end,1)-40,'b','Linewidth',1.5); % distinguishes one year rate
   
      plot(ds,[z-40 z-15 z z+15],'k'); 
      hold on; 
      hold on; 
      for i = 1:size(spdates,1); 
          plot(dates(spdatindx(i))*[1;1],[-40 25],'-k'); 
          hold on; 
      end; 
      axis([1964 2004+2/12 -40 20]);
      set(gca,'FontSize',smallfont); 
      set(gca,'YTick',-40:5:20); 
      set(gca,'Yticklabel','0|5|10||-5|0|5|-5|0|5||0|5'); 
      set(gca,'XTick',1964:2004); 
      set(gca,'XTicklabel','|1965|||||1970|||||1975|||||1980|||||1985|||||1990|||||1995|||||2000||||'); 
    

      text(1964.2,12,'Fama-Bliss','FontSize',smallfont);
       %  text(1964.2,12,'\gamma \prime f 1964-t'); % JC removed this line in final draft -- to robustnes paper
       %  text(1964.2,2,'\gamma \prime f 1964-2003');
        text(1964.2,3,'\gamma \prime f','FontSize',smallfont);
        text(1964.2,-23,'Ex-post returns','FontSize',smallfont);
        text(1964.2,-32,'Yields','FontSize',smallfont);
        
      if printgraph;
    	   print -depsc2 fb3.eps;
      end;
      
       
      % plot of yield curves at special moments in history. 
      % NOte this has to adapt manually to changes in the size and dates of the special dates
      

      if 0; % hard to see much with yields
          figure; 
          plot((1:5)',100*yields(spdatindx(1),:),'-vg');
          hold on; 
          plot((1:5)',100*yields(spdatindx(2),:),'-vr');
          plot((1:5)',100*yields(spdatindx(3),:),'-vg');
          plot((1:5)',100*yields(spdatindx(4),:),'-vg');
          plot((1:5)',100*yields(spdatindx(5),:),'-vr');
          plot((1:5)',100*yields(spdatindx(6),:),'-vg');
          plot((1:5)',100*yields(spdatindx(7),:),'-vr');
      end;    
      
      figure; 
      smallfont = 18; 
      plot((1:5)',100*[yields(spdatindx(1),1) forwards(spdatindx(1),:)],'-^g','Linewidth',2,'MarkerSize',12);
      text(5.1,100*forwards(spdatindx(1),4),'Dec 1981','FontSize',smallfont); 
      hold on; 
      plot((1:5)',100*[yields(spdatindx(2),1) forwards(spdatindx(2),:)],'--vr','Linewidth',2,'MarkerSize',12);
      text(5.1,100*forwards(spdatindx(2),4),'Apr 1983','FontSize',smallfont);
      plot((1:5)',100*[yields(spdatindx(3),1) forwards(spdatindx(3),:)],'-^g','Linewidth',2,'MarkerSize',12);
      text(5.1,100*forwards(spdatindx(3),4),'Aug 1985','FontSize',smallfont);
      plot((1:5)',100*[yields(spdatindx(4),1) forwards(spdatindx(4),:)],'-^g','Linewidth',2,'MarkerSize',12);
      text(5.1,100*forwards(spdatindx(4),4),'Apr 1992','FontSize',smallfont);
      plot((1:5)',100*[yields(spdatindx(5),1) forwards(spdatindx(5),:)],'--vr','Linewidth',2,'MarkerSize',12);
      text(5.1,100*forwards(spdatindx(5),4)+.2,'Aug 1993','FontSize',smallfont);
      plot((1:5)',100*[yields(spdatindx(6),1) forwards(spdatindx(6),:)],'-^g','Linewidth',2,'MarkerSize',12);
      text(5.1,100*forwards(spdatindx(6),4),'Feb 2002','FontSize',smallfont);
      plot((1:5)',100*[yields(spdatindx(7),1) forwards(spdatindx(7),:)],'--vr','Linewidth',2,'MarkerSize',12);
      text(5.1,100*forwards(spdatindx(7),4)-.2,'Dec 2003','FontSize',smallfont);
      set(gca,'FontSize',smallfont); 
      set(gca,'Xtick',[1 2 3 4 5]); 
      axis([0.8 6 1 14.5]); 
      xlabel('Maturity'); 
      ylabel('Forward rate (%)'); 
      
      print -depsc2 specialdates.eps; 
           

% autocorrelation, non-markovian structure of gamma prime f deleted, no
% longer in paper

% ***********************************************
% Regressions of special portfolios to diagnose GMM failures
% **********************************************

   disp('-------------------------------------------------------'); 
   disp('Table 12: Failures of the single-factor model');
   
   % using yields -- prettier since rhv translates directly to price
   olssey = olsse; % initialize
   [gammasy,olssey(5:end),R2humpy,R2humpadjy,vy] = olsgmm(AHPRX,YT,12,0);
   [betsy,olssey(1:4),R2hprxy,R2hprxadjy,vy] = olsgmm(HPRX,YT*gammasy,12,0);
   betsy = betsy'; 
  
   % actually run the regressions with portfolios so you can do R2
   [gammaytil,gtyse,gtyR2,gtyR2adj,gtyv] = olsgmm(HPRX-AHPRX*betsy',YT,12,0);
   [gammaytil,gtyse2,gtyR2,gtyR2adj,gtyv2] = olsgmm(HPRX-AHPRX*betsy',YT,18,1);  % 0 lags had a singular covariance matrix. 
   
   disp('regressions of special portfolios on yields'); 
   disp('gamma tilde -- regression coefficients'); 
   disp(gammaytil'); 
   disp('t statistics HH and NW'); 
   disp(gammaytil'./gtyse'); 
   disp(gammaytil'./gtyse2'); 
   disp('R2'); 
   disp(gtyR2');
   
         
   disp('joint tests; row, stat, %prob -- using NW lags (HH was singular)'); 
   for i = 1:4; 
       stat = gammaytil(2:end,i)'*inv(gtyv2((i-1)*6+2:i*6,2:end))*gammaytil(2:end,i); 
       chi2pv = (1-chi2cdf(stat,5))*100; 
       disp([i stat chi2pv]); 
   end; 
   disp('critical value'); 
   disp(chi2inv(0.95,5)); 
   disp('sigma(Er)'); 
   fit = YT*gammaytil; 
   sd = diag(cov(fit)).^0.5; 
   disp(sd'); 
   disp('sigma(rx) of portfolio'); 
   disp((diag(cov(HPRX-AHPRX*betsy')).^0.5)');
   disp('sigma(Erx) of xs return'); 
   disp((diag(cov((YT*gammasy)*betsy')).^0.5)');
   disp('sigma(rx) of xs returns'); 
   disp((diag(cov(HPRX)).^0.5)');

   % repeat with lagged right hand variables to check iid measurement error interpretation
   % some of these are "significant" but se are suspicious, ts are much
   % lower and there is no pattern, so we conclude nothing really here. 
   
   MAXL=[1:5];
   
   for i=1:maxL
   
       maxL=MAXL(i);
       
       disp('yields on RHS are lagged by this many months:'); disp(maxL);
       
       % lag yields on RHS to check whether it's measurement error
       olsseyL = olsse; % initialize
       AHPRXL  = AHPRX(1+maxL:end,:);
       HPRXL   = HPRX(1+maxL:end,:);
       YTL     = YT(1:end-maxL,:);
       
       [gammasyL,olsseyL(5:end),R2humpyL,R2humpadjyL,vyL] = olsgmm(AHPRXL,YTL,12,0);
       [betsyL,olsseyL(1:4),R2hprxyL,R2hprxadjyL,vyL] = olsgmm(HPRXL,YTL*gammasyL,12,0);
       betsyL = betsyL'; 
       
       [gammaytilL,gtyseL,gtyR2L,gtyR2adjL,gtyvL] = olsgmm(HPRXL-AHPRXL*betsyL',YTL,12,0);
       [gammaytilL,gtyse2L,gtyR2L,gtyR2adjL,gtyv2L] = olsgmm(HPRXL-AHPRXL*betsyL',YTL,18,1);  % 0 lags had a singular covariance matrix. 
       
       disp('regressions of special portfolios on lagged yields'); 
       disp('gamma tilde -- regression coefficients'); 
       disp(gammaytilL'); 
       disp('t statistics HH and NW'); 
       disp(gammaytilL'./gtyseL'); 
       disp(gammaytilL'./gtyse2L'); 
       disp('R2'); 
       disp(gtyR2L');
       
       
       disp('lagged yields - joint tests; row, stat, %prob -- using NW lags (HH was singular)'); 
       for i = 1:4; 
           statL = gammaytilL(2:end,i)'*inv(gtyv2L((i-1)*6+2:i*6,2:end))*gammaytilL(2:end,i); 
           chi2pvL = (1-chi2cdf(statL,5))*100; 
           disp([i statL chi2pvL]); 
       end; 
       disp('critical value'); 
       disp(chi2inv(0.95,5)); 

   end
  
   % *****************************************
   % eigenvalue approach to "extra factors". Does eigenvalue decomposition of Et(rx_t+1) 
   % ****************************************

   disp('eigenvalue decomposition of expected return covariance matrix'); 
   
   covff=cov(FT(:,2:end));
   [Q L] = eig(betamat(:,2:end)*covff*betamat(:,2:end)'); 
   gammat = (Q'*betamat(:,2:end))';  % now betamat = bmat*gammat'

   disp('Q matrix, loadings') ; 
   disp(Q); 
   disp('Gamma matrix, factors'); 
   disp(gammat');   
   
   [betaymat,betyase,R2yunc,R2yuncadj,vy] = olsgmm(HPRX,YT,12,0);
   betaymat = betaymat'; % now is the same size and shape as b*gamma 
 
   covyy=cov(YT(:,2:end));
   [Qy Ly] = eig(betaymat(:,2:end)*covyy*betaymat(:,2:end)'); 
   gamymat = (Q'*betaymat(:,2:end))';  % now betamat = bmat*gammat'
   
   disp('Q matrix with yields on RHS'); 
   disp(Qy); 
   disp('Gamma matrix with yields on RHS'); 
   disp(gamymat'); 
   
   disp('std dev and fractions of variance of the factors'); 
   disp(diag(L)'.^0.5); 
   disp(diag(L)'./sum(diag(L))*100); 
 
   
   % Much stuff in fbregs was deleted 6/6/04 past here. I don't think any
   % of it made it to the paper. 
   
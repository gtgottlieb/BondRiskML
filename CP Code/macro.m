  % macro.m 
  % this is the "new" macro.m macro_old.m contains all sorts of extra
  % exercises, and attempt to connect the factor to macro. This one keeps only the items -- I think -- that made it to
  % the paper, in particular the stock return forecast regressions. If anything is missing, try the old macro.m
  % 
  %does estimates of predictability and risk premia with macro data 

  clear all;  close all;
  printgraph = 1; 
  lotsplots  = 0;
  
  % load monthly yields 
  load bondprice.dat;
  T=length(bondprice);
  y=-log(bondprice(:,2:end)/100).*(ones(T,1)*[1/1 1/2 1/3 1/4 1/5]);
  famablisyld=[bondprice(:,1) y];
  
  famablisyld=famablisyld(1:end,:);
  T=length(famablisyld);
  
  % make nice dates series to get nice graphs
  dates = famablisyld(:,1); 
  yr = floor(dates/10000);
  mo = dates-10000*yr;
  mo = floor(mo/100);
  day = dates-10000*yr-100*mo;
  dates = yr+mo/12;

  
  % collect the annual yields, form prices, forwards, hpr
  T = size(famablisyld,1);

  beg=140;

  yields=famablisyld(beg:end,2:end);
  dates=dates(beg:end);
  T=length(dates);


  % construct annual yields, form prices, forwards, hpr
  mats=[1 2 3 4 5]'; 
  prices=-(ones(T,1)*mats').*yields;
  forwards = prices(:,1:4)-prices(:,2:5);
  fs = forwards-yields(:,1)*ones(1,4);

  hpr = prices(13:T,1:4)-prices(1:T-12,2:5);
  hprx = hpr - yields(1:T-12,1)*ones(1,4);

  HPRX = hprx;
  Ts   = length(HPRX);
  YT = [ones(Ts,1) yields(1:T-12,:)*100];
  FT = [ones(Ts,1) yields(1:T-12,1)*100 forwards(1:T-12,:)*100]; 

  AHPRX = 100*mean(HPRX')'; 
  h=AHPRX/100;
  
  % construct gamma'f 
  [gammas,stgamma,tes,r2]=olsgmm(AHPRX,FT,0,0);     
  gpf=[ones(length(yields),1) yields(:,1)*100 forwards*100]*gammas;  % time t as in yields 


  %   ([gammas(1); gammas(2:end)-mean(gammas(2:end))]);  % time t as in yields
  % robustness check: was it level of interest rates? A: no, gpf2 works just as well and has no level effect.  


  load indices.txt;
  msia=indices(:,1:3);
  mvwret = 1+msia(2:end,2);
  mvwretx = 1+msia(2:end,3);

  avwret = mvwret(12:end).*mvwret(11:end-1).*mvwret(10:end-2).*mvwret(9:end-3).* ...
    mvwret(8:end-4).*mvwret(7:end-5).*mvwret(6:end-6).*mvwret(5:end-7).* ...
    mvwret(4:end-8).*mvwret(3:end-9).*mvwret(2:end-10).*mvwret(1:end-11);

  avwretx = mvwretx(12:end).*mvwretx(11:end-1).*mvwretx(10:end-2).*mvwretx(9:end-3).* ...
    mvwretx(8:end-4).*mvwretx(7:end-5).*mvwretx(6:end-6).*mvwretx(5:end-7).* ...
    mvwretx(4:end-8).*mvwretx(3:end-9).*mvwretx(2:end-10).*mvwretx(1:end-11);

  vwdp = avwret./avwretx-1;
  avwret = avwret(458-12:end);

  avwretx = avwretx(458-12:end); 
  vwdp = vwdp(458-12:end);


disp('-----------------Forecasting excess stock returns, Table 6-----------------------');


Tl2 = size(avwret,1); % use real tl when you update data

ers=avwret(13:Tl2,1)-yields(13:Tl2,1)-1;
[bv,sebv,R2v,v] = olsgmm(ers,[ones(Tl2-12,1) vwdp(1:Tl2-12,1)],12,0);
disp('vw excess returns on d/p'); disp(([bv(2) bv(2)/sebv(2) R2v]));

[bv,sebv,R2v,v] = olsgmm(ers(1:end-(120+24+12)), ...
                         [ones(Tl2-12-(120+24+12),1) vwdp(1:Tl2-12-(120+24+12),1)],12,0);
disp('vw excess returns on d/p, not last 10 years'); disp(([bv(2) bv(2)/sebv(2) R2v]));

[bv,sebv,R2v,v] = olsgmm(ers, ...
                         [ones(Tl2-12,1) yields(1:Tl2-12,5)-yields(1:Tl2-12,1)],12,0);
disp('vw excess returns on y5 - y1'); disp(([bv(2) bv(2)/sebv(2) R2v]));

[bv,sebv,R2v,v] = olsgmm(ers, ...
                         [ones(Tl2-12,1) vwdp(1:Tl2-12,1) yields(1:Tl2-12,5)-yields(1:Tl2-12,1)],12,0);
disp('vw excess returns on d/p and y5 - y1'); disp(([bv(2:3)' bv(2:3)'./sebv(2:3)' R2v]));

[bv,sebv,R2v,v] = olsgmm(ers, ...
                         [ones(Tl2-12,1) gpf(1:Tl2-12)/100],12,0);
disp('vw excess returns on gpf'); disp(([bv(2) bv(2)/sebv(2) R2v]));

[bv,sebv,R2v,v] = olsgmm(ers, ...
                         [ones(Tl2-12,1) gpf(1:Tl2-12)/100 yields(1:Tl2-12,5)-yields(1:Tl2-12,1)],12,0);
disp('vw excess returns on gpf and y5 - y1'); disp(([bv(2:3)' bv(2:3)'./sebv(2:3)' R2v]));

[bv,sebv,R2v,v] = olsgmm(ers, ...
                         [ones(Tl2-12,1) gpf(1:Tl2-12)/100 vwdp(1:Tl2-12,1)],12,0);
disp('vw excess returns on gpf and d/p'); disp(([bv(2:3)' bv(2:3)'./sebv(2:3)' R2v]));

[bv,sebv,R2v,v] = olsgmm(ers, ...
                         [ones(Tl2-12,1) gpf(1:Tl2-12)/100 vwdp(1:Tl2-12,1) yields(1:Tl2-12,5)-yields(1:Tl2-12,1)],12,0);
disp('vw excess returns on gpf d/p and term'); disp(([bv(2:4)' bv(2:4)'./sebv(2:4)' R2v]));

[bv,sebv,R2v,v] = olsgmm(ers, ...
                         [ones(Tl2-12,1) yields(1:Tl2-12,1) forwards(1:Tl2-12,:)],12,0);
disp('vw excess returns on all forwards');
disp(([bv' R2v]));
disp((bv'./sebv'));

maxlag = 12;
   alpha_v = zeros(maxlag,maxlag); 
   se_alpha_v = ones(maxlag,maxlag); % avoids error message when you make t stats. 
   for lags = 1:maxlag;  % number of lags in total to include 1 means just ft
        alphas = zeros(lags,1); 
        alphas(1) = 1; % start with just time t value
  
        %disp('iterating to estimate of a restricted model. watching gamma and alpha in iterations.' ); 
        for iter = 1:10; 
            FTfilt = filter(alphas,1,FT); 
            FTfilt = FTfilt(lags:Ts,:); 
   
            [gammas_lags,se_trash,R2_lags] = olsgmm(AHPRX(lags:Ts),FTfilt,18,1); 
            %to debug: disp(gammas_lags'); 
            gpf_lags = FT*gammas_lags; % note NOT Ftfilt -- this is not the fitted value; it's the individual terms so you can estimate alpha

            % step 2: estimate alphas
            rhv = [ ones(Ts-lags+1,1) gpf_lags(lags:Ts)]; 
            for i = 1:lags-1; 
                rhv = [rhv gpf_lags(lags-i:Ts-i)];
            end; 

            [alphas, se_alphas] = olsgmm(AHPRX(lags:Ts),rhv,18,1); 
            % to debug: disp(alphas'); 
            alphas = alphas(2:end); % don't want to use the constant to filter!
            se_alphas = se_alphas(2:end)/sum(alphas);
            alphas = alphas/sum(alphas); % normalize to sum of alphas = 1  
            
        end;
        gamma_v(:,lags) = gammas_lags; % keep track for printing
        alpha_v(1:lags,lags) = alphas; 
        se_alpha_v(1:lags,lags) = se_alphas; 
        R2_v(lags) = R2_lags; 
        gpf_lags_fit = FTfilt*gammas_lags; % this one IS the fitted value, used next 


        [bvf,sebvf,R2vf,vf] = olsgmm(ers(lags:end),[ones(length(ers(lags:end)),1) gpf_lags_fit/100],12,0);
        disp('vw excess returns on filtered gpf');
        disp(([lags-1 bvf(2) bvf(2)/sebvf(2) R2vf]));

        [bvf2,sebvf2,R2vf2,vf2] = olsgmm(ers(lags:end), [ones(Tl2-12-lags+1,1) gpf_lags_fit/100 vwdp(1:Tl2-12-lags+1,1) yields(1:Tl2-12-lags+1,5)-yields(1:Tl2-12-lags+1,1)],12,0);
        disp('vw excess returns on filtered gpf d/p and term'); disp(([lags-1 bvf2(2:4)' bvf2(2:4)'./sebvf2(2:4)' R2vf2]));

        
end



[bv2,sebv2,R2v2,v2] = olsgmm(h(1:Tl2-12,:), ...
                         [ones(Tl2-12,1) yields(1:Tl2-12,1) forwards(1:Tl2-12,:)],12,0);
disp('bond average excess return on all forwards');
disp(([bv2' R2v2]));
disp((bv2'./sebv2'));

figure; 
errorbar((1:5)-0.02,bv(2:end),2*sebv(2:end),'--vr');
hold on;
errorbar((1:5)+0.02,bv2(2:end),2*sebv2(2:end),'-og');
xlabel('Maturity');
ylabel('Coefficient');
if printgraph; 
   print -depsc2 stock.eps;
end; 

% try it in logs, like bonds. About the same, so no point. 
[bv,sebv,R2v,v] = olsgmm(log(avwret(13:Tl2,1))-yields(13:Tl2,1)-1, ...
                         [ones(Tl2-12,1) yields(1:Tl2-12,1) forwards(1:Tl2-12,:)],12,0);
disp('log vw excess returns on all forwards');
disp([bv' R2v]);
disp(bv'./sebv');


figure; 
errorbar((1:5)-0.02,bv(2:end),2*sebv(2:end),'--v');
hold on;
errorbar((1:5)+0.02,bv2(2:end),2*sebv2(2:end),'-o');


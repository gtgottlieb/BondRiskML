% **********************************
% Monte Carlos for Bond Risk Premia
% **********************************

clear all; close all;

diary fbmc_out.txt;

%----------------------------------------------------------------------------%
% Choose data-generating process for simulations
% mc = 
%       1  : VAR(12) for yields
%       2  : Expectations hypothesis, AR(12) for short rate
%       3  : Cointegrated VAR(12)
%----------------------------------------------------------------------------%

mc=1;

%----------------------------------------------------------------------------%
% More choices : 
% S     = number of simulations (will be displayed in output)
% nlags = lags for cov matrix
% nw    = 1 for newey west
% L     = number of lags in estimated VAR for simulations. Should be 12; too few and you miss lag structure and don't see tent. 
%----------------------------------------------------------------------------%

S=50000;
nlags=18;  % newey west lags to use when doing NW rather than HH to get nonsingular cov matrices 
nw=1;      % use newey west rather than HH
L=12;      


%----------------------------------------------------------------------------%
% Load data 
%----------------------------------------------------------------------------%

        load bondprice.dat;

        T=length(bondprice);
        y=-log(bondprice(:,2:end)/100).*(ones(T,1)*[1/1 1/2 1/3 1/4 1/5]);
        famablisyld=[bondprice(:,1) y];

        yields=famablisyld(:,2:end);

        mats=[1 2 3 4 5]'; 
        N=size(mats,1);
        prices=-(ones(T,1)*mats').*yields;
        forwards = prices(:,1:4)-prices(:,2:5);
        fs = forwards-yields(:,1)*ones(1,4);

        hpr = prices(13:T,1:4)-prices(1:T-12,2:5);
        hprx = hpr - yields(1:T-12,1)*ones(1,4);
        hpr = [zeros(12,1)*ones(1,4); hpr];
        hprx = [zeros(12,1)*ones(1,4); hprx];

    % sample restriction to FB sample
        yields=yields(140:end,:);
        forwards=forwards(140:end,:);
        fs=fs(140:end,:);
        hprx=hprx(140:end,:);
        T=length(yields);
        
        HPRX = 100*hprx(13:end,:);
        AHPRX= mean(HPRX')';
        FT   = [ones(T-12,1) 100*yields(1:end-12,1) 100*forwards(1:end-12,:)];
        FS   = [ones(T-12,1) 100*fs(1:T-12,:)];     % forward-spot spread
        YT   = [ones(T-12,1) 100*yields(1:end-12,:)];

        % 1 lag
        HPRXL   = HPRX(2:end,:);
        AHPRXL  = AHPRX(2:end,:);
        FTL     = FT(1:end-1,:);
        
        % 2 lags
        HPRXLL  = HPRX(3:end,:);
        AHPRXLL = AHPRX(3:end,:);
        FTLL    = FT(1:end-2,:);
        
        % construct principal components
        Vy = cov(YT(:,2:end));
        [Q,Ldata] = eig(Vy);   
        [D,Indx] = sort(-diag(Ldata));
        Q = Q(:,Indx); 
        PC=YT(:,2:end)*Q;  
        curve = PC(:,3); 
        slope = PC(:,2);
        level = PC(:,1);   


%----------------------------------------------------------------------------%
% Predictability regressions with data
%----------------------------------------------------------------------------%

% Fama-Bliss regressions
        indx = 1; 
        
        while indx <= 4;
             [bFB(:,indx),se,R2FB(indx,1),Radj,v,WFB(indx,:)] = ...
                     olsgmm(HPRX(:,indx),FS(:,[1 1+indx]),nlags,nw);
             indx=indx+1;
         end
             
             [bFBy,se,R2FBy,Radj,v,WFBy] = ...
                     olsgmm(100*(yields(1+12:T,1)-yields(1:T-12,1)), ...
                                         FS(1:T-12,[1 2]),nlags,nw);
            

%  Cochrane-Piazzesi regressions         
         [betasCP,se,R2CP,Radj,v,WCP]    = olsgmm(HPRX,FT,nlags,nw);
         bun = betasCP';  % 4x6
        
         %1 lag
         [betasCPL,seL,R2CPL,RadjL,vL,WCPL]    = olsgmm(HPRXL,FTL,nlags,nw);
         bunL = betasCPL';  % 4x6

         %2 lags
         [betasCPLL,seLL,R2CPLL,RadjLL,vLL,WCPLL]    = olsgmm(HPRXLL,FTLL,nlags,nw);
         bunLL = betasCPLL';  % 4x6

         [betasCPy,se,R2CPy,Radj,v,WCPy] = olsgmm(100*(yields(1+12:T,1)-yields(1:T-12,1)),FT,nlags,nw);
            
             
%  Restricted CP regressions
             
         [gammas,se,R2CPav,Radj,v,WCPrest]=olsgmm(AHPRX,FT,nlags,nw);
         [bCP,se,R2CPres]=olsgmm(HPRX,FT*gammas,nlags,nw);
         bCP = bCP';      % 4x1
         br  = bCP*gammas';  % 4x6
         
         %1 lag
         [gammasL,seL,R2CPavL,RadjL,vL,WCPrestL]=olsgmm(AHPRXL,FTL,nlags,nw);
         [bCPL,seL,R2CPresL]=olsgmm(HPRXL,FTL*gammasL,nlags,nw);
         bCPL = bCPL';      % 4x1
         brL  = bCPL*gammasL';  % 4x6

         %2 lags
         [gammasLL,seLL,R2CPavLL,RadjLL,vLL,WCPrestLL]=olsgmm(AHPRXLL,FTLL,nlags,nw);
         [bCPLL,seLL,R2CPresLL]=olsgmm(HPRXLL,FTLL*gammasLL,nlags,nw);
         bCPLL = bCPLL';      % 4x1
         brLL  = bCPLL*gammasLL';  % 4x6

                  
% Testing gamma'f

         % Jtest
         err = HPRX-FT*br';
         u = [(err(:,1)*ones(1,6)).*FT ...
              (err(:,2)*ones(1,6)).*FT ...
              (err(:,3)*ones(1,6)).*FT ...
              (err(:,4)*ones(1,6)).*FT];
         % note this uses the restricted errors. Mean u will be gt. S may be affected
         gt = mean(u)'; 

         % 1 lag
         errL = HPRXL-FTL*brL';
         uL = [(errL(:,1)*ones(1,6)).*FTL ...
               (errL(:,2)*ones(1,6)).*FTL ...
               (errL(:,3)*ones(1,6)).*FTL ...
               (errL(:,4)*ones(1,6)).*FTL];
         % note this uses the restricted errors. Mean u will be gt. S may be affected
         gtL = mean(uL)'; 

         % 2 lags
         errLL = HPRXLL-FTLL*brLL';
         uLL = [(errLL(:,1)*ones(1,6)).*FTLL ...
                (errLL(:,2)*ones(1,6)).*FTLL ...
                (errLL(:,3)*ones(1,6)).*FTLL ...
                (errLL(:,4)*ones(1,6)).*FTLL];
         % note this uses the restricted errors. Mean u will be gt. S may be affected
         gtLL = mean(uLL)'; 
        
         
         errunc=HPRX-FT*bun';
         uu = [(errunc(:,1)*ones(1,6)).*FT ...
               (errunc(:,2)*ones(1,6)).*FT ...
               (errunc(:,3)*ones(1,6)).*FT ...
               (errunc(:,4)*ones(1,6)).*FT];
   
         a = [ kron(ones(1,4),eye(6)) ; 
                kron(eye(3),gammas') zeros(3,6) ]; 
                
         Eff=FT'*FT/size(FT,1);
         d = [ kron([-eye(3);ones(1,3)],Eff*gammas)  -kron(bCP,Eff)];
     
         SM = spectralmatrix(uu,nlags,nw);   

         covgt = (eye(24)-d*inv(a*d)*a)*SM*(eye(24)-d*inv(a*d)*a)'/size(FT,1);
         invcovgt = pinv2(covgt,15);
         jtstat=gt'*invcovgt*gt;
         JT = [jtstat chi2inv(0.95,15) 100*(1-cdf('chi2',jtstat,15))];
  
         
         % 1 lag
         erruncL=HPRXL-FTL*bunL';
         uuL = [(erruncL(:,1)*ones(1,6)).*FTL ...
                (erruncL(:,2)*ones(1,6)).*FTL ...
                (erruncL(:,3)*ones(1,6)).*FTL ...
                (erruncL(:,4)*ones(1,6)).*FTL];
   
         aL = [ kron(ones(1,4),eye(6)) ; 
                kron(eye(3),gammasL') zeros(3,6) ]; 
                
         EffL=FTL'*FTL/size(FTL,1);
         dL = [ kron([-eye(3);ones(1,3)],EffL*gammasL)  -kron(bCPL,EffL)];
     
         SML = spectralmatrix(uuL,nlags,nw);   

         covgtL = (eye(24)-dL*inv(aL*dL)*aL)*SML*(eye(24)-dL*inv(aL*dL)*aL)'/size(FTL,1);
         invcovgtL = pinv2(covgtL,15);
         jtstatL=gtL'*invcovgtL*gtL;
         JTL = [jtstatL chi2inv(0.95,15) 100*(1-cdf('chi2',jtstatL,15))];
        
         
         % 2 lag
         erruncLL=HPRXLL-FTLL*bunLL';
         uuLL = [(erruncLL(:,1)*ones(1,6)).*FTLL ...
                 (erruncLL(:,2)*ones(1,6)).*FTLL ...
                 (erruncLL(:,3)*ones(1,6)).*FTLL ...
                 (erruncLL(:,4)*ones(1,6)).*FTLL];
   
         aLL = [ kron(ones(1,4),eye(6)) ; 
                 kron(eye(3),gammasLL') zeros(3,6) ]; 
                
         EffLL=FTLL'*FTLL/size(FTLL,1);
         dLL = [ kron([-eye(3);ones(1,3)],EffLL*gammasLL)  -kron(bCPLL,EffLL)];
     
         SMLL = spectralmatrix(uuLL,nlags,nw);   

         covgtLL = (eye(24)-dLL*inv(aLL*dLL)*aLL)*SMLL*(eye(24)-dLL*inv(aLL*dLL)*aLL)'/size(FTLL,1);
         invcovgtLL = pinv2(covgtLL,15);
         jtstatLL=gtLL'*invcovgtLL*gtLL;
         JTLL = [jtstatLL chi2inv(0.95,15) 100*(1-cdf('chi2',jtstatLL,15))];
         
         % Wald test
         du = kron(eye(4),Eff); 
         covbu = kron(eye(4),inv(Eff)); 
         covbu = covbu*SM*covbu'/size(FT,1);   % cov matrix of unr pars
         covbu = (covbu+covbu')/2; 
       
         duL = kron(eye(4),EffL); 
         covbuL = kron(eye(4),inv(EffL)); 
         covbuL = covbuL*SML*covbuL'/size(FTL,1);   % cov matrix of unr pars
         covbuL = (covbuL+covbuL')/2; 

         duLL = kron(eye(4),EffLL); 
         covbuLL = kron(eye(4),inv(EffLL)); 
         covbuLL = covbuLL*SMLL*covbuLL'/size(FTLL,1);   % cov matrix of unr pars
         covbuLL = (covbuLL+covbuLL')/2; 
             
         bun  =vec(bun');
         bunL =vec(bunL');
         bunLL=vec(bunLL');
         
         br   = vec(br');
         brL  = vec(brL');
         brLL = vec(brLL');
         
         wstat=(bun-br)'*inv(covbu)*(bun-br);
         WR = [wstat chi2inv(0.95,15) 100*(1-cdf('chi2',wstat,15))];
         
         wstatL=(bunL-brL)'*inv(covbuL)*(bunL-brL);
         WRL = [wstatL chi2inv(0.95,15) 100*(1-cdf('chi2',wstatL,15))];
         
         wstatLL=(bunLL-brLL)'*inv(covbuLL)*(bunLL-brLL);
         WRLL = [wstatLL chi2inv(0.95,15) 100*(1-cdf('chi2',wstatLL,15))];

         
% Testing other restrictions

        gammastar=olsgmm(AHPRX,YT,nlags,nw);
        
        A=zeros(6,6,6);
        blarge=zeros(6,6);
        N = [3;4;5;3;4;5];

        % x(t) = [1 slope y2 y3 y4 y5]'
        % y(t) = [1 y1    y2 y3 y4 y5]'
        % x(t) = a*y(t)
        
        % slope - y2, y3, y4, y5
                          
        A(:,:,1)=[1 zeros(1,5); 
                  0 Q(:,2)';   % slope
                  zeros(4,2) eye(4)];
                     
        % slope, level - y3, y4, y5
        A(:,:,2)=[1 zeros(1,5);
                  0 Q(:,2)'; % slope
                  0 Q(:,1)'; % level
                  zeros(3,3) eye(3)];
        
        % slope,level,curve -- y4 y5
        A(:,:,3)=[1 zeros(1,5);
                  0 Q(:,2)'; % slope
                  0 Q(:,1)'; % level
                  0 Q(:,3)'; % curve
                  zeros(2,4) eye(2)];
               
        % y5-y1 - y2, y3, y4, y5
        A(:,:,4)=[1 zeros(1,5);
                  0 -1 0 0 0 1; % y5-y1
                  zeros(4,2) eye(4)];
        
        % y1, y5 - y2, y3, y4
        A(:,:,5)=[1 zeros(1,5);
                  0 1 0 0 0 0; % y1 
                  0 0 0 0 0 1; % y5
                  zeros(3,2) eye(3) zeros(3,1)];
        
        % y(1), y(4), y(5) - y2 y3
        A(:,:,6)=[1 zeros(1,5);
                  0 1 0 0 0 0; % y1
                  0 0 0 0 1 0; % y4 
                  0 0 0 0 0 1; % y5
                  0 0 1 0 0 0;
                  0 0 0 1 0 0];
 
       % x(t) = a y(t)       
       %   XT = YT*a'    
       for i=1:6
           a=A(:,:,i);
           blarge(:,i) = olsgmm(AHPRX,YT*a',nlags,nw);
       end
 
%--------------------------------------------------------------------------%
% Estimate data-generating process for yields
%--------------------------------------------------------------------------%
        y=yields*100;

        yL  = ones(T-L,1);
        dyL = ones(T-L,1);
        yc  = ones(T-L,1);                          

if mc == 1
    
        % right-hand side for VAR(12)
        for i=L:-1:1;
            yL=[yL y(i:T-L+i-1,:)];
        end

        % arcoef is constant, 
        % then 2-6 is AR(1), 7-11 is AR(2), ... 
        arcoef = yL\y(L+1:T,:);   
        err = y(L+1:T,:)-yL*arcoef;  % iid shocks
        yL =yL(:,2:end);             % rhs without constant
        
elseif mc==2

        % left-hand and ride-hand side for VAR(1) 
        % which is extended Markov system for AR(12) short rate
        for i=L:-1:1;                      %  Markov state y(t) and its lag y(t-1)
            yL =[yL y(i:T-L+i-1,1)];       %  y(t-1)=(1,y1(t-1),y1(t-2),....,y(t-12))
            yc =[yc y(i+1:T-L+i,1)];       %  y(t)  =(1,y1(t),y1(t-1),....,y1(t-11)) 
        end

        % resulting autoregressive coefficient is a constant,   
        % then 2-6 is AR(1), 7-11 is AR(2), ... 
        a = yL\yc(:,2);   

        A0=[a(1);zeros(L-1,1)]; 
        A1=[a(2:end)';eye(L-1) zeros(L-1,1)]; 
        SIGMA=[1 zeros(1,L-1); zeros(L-1,L)];

        % map from state to yields: 1-year = Y*one,  
        one = [1;zeros(L-1,1)];
        [a0_12,a1_12]=ycoeff(12,A0,A1);
        [a0_24,a1_24]=ycoeff(24,A0,A1);
        [a0_36,a1_36]=ycoeff(36,A0,A1);
        [a0_48,a1_48]=ycoeff(48,A0,A1);

        err = yc(:,2)-yL*a;      % iid shocks
        yL  = yL(:,2:end);       % Markov t-1 without constant
        yc  = yc(:,2:end);       % Markov t without constant

else

        % right-hand side for cointegrated VAR(12)
        for i=L:-1:1;
            yL=[yL y(i:T-L+i-1,:)];  % lagged yield levels
        end
        
        for i=L:-1:2;
            dyL=[dyL y(i:T-L+i-1,:)-y(i-1:T-L+i-2,:)]; % lagged yield differences
            spr=[yL(:,2)-yL(:,6) yL(:,3)-yL(:,6) yL(:,4)-yL(:,6) yL(:,5)-yL(:,6)]; % lagged spreads
        end

       
        dyL = dyL(:,2:end);
       
        % left-hand side for cointegrated VAR(12)
        dy=y(L+1:T,:)-y(L:T-1,:);
        RHS=[ones(T-L,1) spr dyL];
        arcoef = RHS\dy;

        err=dy-RHS*arcoef;       % iid shocks
        
end

%----------------------------------------------------------------------------%
% Run bootstrap
%----------------------------------------------------------------------------%

seed(100); disp('# simulations'); disp(S);
    


for s=1:S

       if s/1000==floor(s/1000);
           disp(s);
       end

    if mc==1
        
       yt=y(L,:)';                        % start in t=L
       ytL= [yL(1,6:end) zeros(1,5)]; 
       yh(1:L,:)=y(1:L,:);                % presample values
       
        for t = L+1:T
            ytL = [yt' ytL(:,1:end-5)];    
            ut    = err(ceil((T-L).*rand(1))',:)';
            %test code 
            %ut = err(t-L,:)';
            yt   = ([1 ytL]*arcoef)' + ut;
            yh(t,:)=yt';
        end
   
    elseif mc==2
        
        % simulate the 1-year rate   
        yt=yc(L,:)';           % Markov simulated state t                
        ych(1:L,:)=yc(1:L,:);  % Markov simulated state L initial values              
       
        for t = L+1:T
            ytL = yt;          % update Markov state t-1
            ut    = err(ceil((T-L).*rand(1))');
            %ut =err(t-L,:);   % check code here
            yt   = A0+A1*ytL + SIGMA*[ut;zeros(L-1,1)]; % Markov state t
            ych(t,:)=yt';      % save Markov state t
        end
    
        % compute yields yh using expectations hypothesis
        yh(:,1)=ych*one;
        yh(:,2)=(1/2)*ych*one+(1/2)*(a0_12'*one+ych*a1_12'*one);
        yh(:,3)=(2/3)*yh(:,2)+(1/3)*(a0_24'*one+ych*a1_24'*one);
        yh(:,4)=(3/4)*yh(:,3)+(1/4)*(a0_36'*one+ych*a1_36'*one);
        yh(:,5)=(4/5)*yh(:,4)+(1/5)*(a0_48'*one+ych*a1_48'*one); 
    
             
    else

               
        yt   = y(L,:)';
        ytL  = y(L-1,:)';
        dyt  = yt-ytL;                                % start in t=L
        dytL = [dyL(1,6:end) zeros(1,5)]; 
        yh(1:L,:)=y(1:L,:);                           % presample values
       
        for t = L+1:T
            ytL  = yt;
            dytL = [dyt' dytL(:,1:end-5)];
            sprt = (yt(1:4)-yt(5))';
            ut    = err(ceil((T-L).*rand(1))',:)';
            dyt   = ([1 sprt dytL]*arcoef)' + ut;
            yt    = dyt + ytL;
            yh(t,:)=yt';
        end
        
    end

   
%-------------------------------------------------------------------------------%
% Compute holding period returns for simulated yields
%-------------------------------------------------------------------------------%

        yieldsh=yh/100;
        
        pricesh=-(ones(T,1)*mats').*yieldsh;
        forwardsh = pricesh(:,1:4)-pricesh(:,2:5);
        fsh   = forwardsh-yieldsh(:,1)*ones(1,4);

        hprh  = pricesh(13:T,1:4)-pricesh(1:T-12,2:5);
        hprxh = hprh - yieldsh(1:T-12,1)*ones(1,4);
      
        hprh  = [zeros(12,1)*ones(1,4); hprh];
        hprxh = [zeros(12,1)*ones(1,4); hprxh];

        HPRXh  = 100*hprxh(13:end,:);
        AHPRXh = mean(HPRXh')';     
        FTh    = [ones(T-12,1) 100*yieldsh(1:end-12,1) 100*forwardsh(1:end-12,:)];
        FSh    = [ones(T-12,1) 100*fsh(1:T-12,:)];     
        YTh    = [ones(T-12,1) 100*yieldsh(1:end-12,:)];
  
        HPRXhL   = HPRXh(2:end,:);
        AHPRXhL  = AHPRXh(2:end,:);
        FThL     = FTh(1:end-1,:);
        
        HPRXhLL  = HPRXh(3:end,:);
        AHPRXhLL = AHPRXh(3:end,:);
        FThLL    = FTh(1:end-2,:);
        

 %----------------------------------------------------------------------------------%               
 % Predictability regressions with simulated data
 %----------------------------------------------------------------------------------%

 % Fama-Bliss regression
        indx = 1; 
        while indx <= 4;
           
             [bhFB(s,:,indx),se,R2hFB(indx,s),R2adj,v,WFBh(s,indx,:)] ...
                      = olsgmm(HPRXh(:,indx),FSh(:,[1 indx+1]),nlags,nw); 
             indx=indx+1;
        end
            
             lhv = 100*yieldsh(1+12:T,1)-100*yieldsh(1:T-12,1);
             [bhFBys,se,R2hFBy(s),R2adj,v,WFByh(s,:)] ...
                      = olsgmm(lhv, FSh(1:T-12,[1 2]),nlags,nw);
             bhFBy(s,:)=bhFBys';
        
% Cochrane-Piazzesi regression
     	    
        [bunh,se,R2hCP(:,s),R2adj,v,WCPh(s,:,:)] = olsgmm(HPRXh,FTh,nlags,nw);
        betashCP(s,:,:)=bunh; % 6x4 
        bunCP(s,:)=vec(bunh)'; % 24 x1 
        bunh=bunh'; % 4x6 
       
        [bunhL,seL,R2hCPL(:,s),R2adjL,vL,WCPhL(s,:,:)] = olsgmm(HPRXhL,FThL,nlags,nw);
        betashCPL(s,:,:)=bunhL; % 6x4 
        bunCPL(s,:)=vec(bunhL)'; % 24 x1 
        bunhL=bunhL'; % 4x6 

     
        [bunhLL,seLL,R2hCPLL(:,s),R2adjLL,vLL,WCPhLL(s,:,:)] = olsgmm(HPRXhLL,FThLL,nlags,nw);
        betashCPLL(s,:,:)=bunhLL; % 6x4 
        bunCPLL(s,:)=vec(bunhLL)'; % 24 x1 
        bunhLL=bunhLL'; % 4x6 
       
        
        lhv = 100*yieldsh(1+12:T,1)-100*yieldsh(1:T-12,1);            
        [betasCPy,se,R2hCPy(:,s),Rsadj,v,WCPyh(s,:)] = olsgmm(lhv,FTh(1:T-12,:),nlags,nw);
        betashCPy(s,:)=betasCPy';
     
% Restricted CP regression
               
        % 0 lags
        [gammash,se,R2havCP(:,s),Radj,v,WCPresth(s,:)] = olsgmm(AHPRXh,FTh,nlags,nw);
        ghCP(s,:)=gammash';
               
        % 1 lag
        [gammashL,seL,R2havCPL(:,s),RadjL,vL,WCPresthL(s,:)] = olsgmm(AHPRXhL,FThL,nlags,nw);
        ghCPL(s,:)=gammashL';
        
        % 2 lags
        [gammashLL,seLL,R2havCPLL(:,s),RadjLL,vLL,WCPresthLL(s,:)] = olsgmm(AHPRXhLL,FThLL,nlags,nw);
        ghCPLL(s,:)=gammashLL';
 
        % 0 lags
        [bhCP(s,:),se,R2hCPres(:,s)]=olsgmm(HPRXh,FTh*gammash,nlags,nw);
        bCPh=bhCP(s,:); bCPh=bCPh';
        brh=bCPh*gammash'; % 4x6
         
        % 1 lag
        [bhCPL(s,:),seL,R2hCPresL(:,s)]=olsgmm(HPRXhL,FThL*gammashL,nlags,nw);
        bCPhL=bhCPL(s,:); bCPhL=bCPhL';
        brhL=bCPhL*gammashL'; % 4x6
        
        % 2 lags
        [bhCPLL(s,:),seLL,R2hCPresLL(:,s)]=olsgmm(HPRXhLL,FThLL*gammashLL,nlags,nw);
        bCPhLL=bhCPLL(s,:); bCPhLL=bCPhLL';
        brhLL=bCPhLL*gammashLL'; % 4x6

        % 0 lags 
        errh = HPRXh-FTh*brh';
        uh = [(errh(:,1)*ones(1,6)).*FTh ...
              (errh(:,2)*ones(1,6)).*FTh ...
              (errh(:,3)*ones(1,6)).*FTh ...
              (errh(:,4)*ones(1,6)).*FTh]';
      
        meangth(s,:) = mean(uh'); 
      
        % 1 lag
        errhL = HPRXhL-FThL*brhL';
        uhL   = [(errhL(:,1)*ones(1,6)).*FThL ...
                 (errhL(:,2)*ones(1,6)).*FThL ...
                 (errhL(:,3)*ones(1,6)).*FThL ...
                 (errhL(:,4)*ones(1,6)).*FThL]';
      
        meangthL(s,:) = mean(uhL'); 

        % 2 lags

        errhLL = HPRXhLL-FThLL*brhLL';
        uhLL   = [(errhLL(:,1)*ones(1,6)).*FThLL ...
                  (errhLL(:,2)*ones(1,6)).*FThLL ...
                  (errhLL(:,3)*ones(1,6)).*FThLL ...
                  (errhLL(:,4)*ones(1,6)).*FThLL]';
      
        meangthLL(s,:) = mean(uhLL'); 

        
        % Testing other restrictions

        gammastar=olsgmm(AHPRXh, YTh, nlags,nw);
        gammastarh(s,:)=gammastar';
   
                                                
   end

% Collect results

   for i=1:4
         bFBh(:,i)=mean(bhFB(:,:,i))';
         stdFBh(:,i)=std(bhFB(:,:,i))';
         vFBh(:,:,i)=cov(bhFB(:,:,i));
         
         betasCPh(:,i)=mean(betashCP(:,:,i))';
         stdbetasCPh(:,i)=std(betashCP(:,:,i))';
         vCPh(:,:,i) =cov(betashCP(:,:,i));
   end
        
        bCPh=mean(bhCP)';
        stdbCPh=std(bhCP)';
        varbCPh=cov(bhCP);
    
        bFByh=mean(bhFBy)';
        stdFByh=std(bhFBy)';
        varFByh=cov(bhFBy);
        
        betasCPhy=mean(betashCPy);
        stdbetasCPhy=std(betashCPy);
        varCPyh=cov(betashCPy);
        
        gCPh   = mean(ghCP)';
        gCPhL  = mean(ghCPL)';
        gCPhLL = mean(ghCPLL)';
        
        stdgCPh=std(ghCP)';
                
        vgCPh  =cov(ghCP);
        vgCPhL =cov(ghCPL);
        vgCPhLL=cov(ghCPLL);
        
      
        % sort all Wald-statistics, starting with smallest
        WFBsort=sort(WFBh(:,:,1));
        WFBysort=sort(WFByh(:,:,1));
        WCPsort=sort(WCPh(:,:,1));
        WCPysort=sort(WCPyh(:,1));
        WCPrestsort=sort(WCPresth(:,1));
              
        WCPyhpvalue=(length(find(WCPysort>WCPy(:,1))))/S;
        WCPresthpvalue=(length(find(WCPrestsort>WCPrest(:,1))))/S;   
        WFByhpvalue=(length(find(WFBysort>WFBy(:,1))))/S;

    for indx=1:4
        
        WFBhpvalue(indx)=(length(find(WFBsort(:,indx)>WFB(indx,1))))/S;
        WCPhpvalue(indx)=(length(find(WCPsort(:,indx)>WCP(indx,1))))/S;
               
    end
    
    
        R2FBconf   =percentile(R2hFB',   [0.025 0.975])';
        R2FByconf  =percentile(R2hFBy',  [0.025 0.975])';
        R2CPconf   =percentile(R2hCP',   [0.025 0.975])';
        R2CPyconf  =percentile(R2hCPy',  [0.025 0.975])';
        R2CPavconf =percentile(R2havCP', [0.025 0.975])'; 
        R2CPresconf=percentile(R2hCPres',[0.025 0.975])';
      
   
disp('====================================================');    
disp('Bootstrap results');

disp(' ');
disp('Data-generating process is:');
if mc==1 
    disp(' VAR(12) for yields');
elseif mc==2
    disp(' Expectations hypothesis, AR(12) for short rate');
else
    disp(' Cointegrated VAR for yields');
end

disp(' ');
disp('Number of simulations:');
disp(S);


disp('-----------------------------------------------------');
disp('Table 1: Regression of 1-year excess return on all forward rates');

    for i=1:4
        v=vCPh(:,:,i);
        b=betasCP(:,i);
        w=b(2:end)'*inv(v(2:end,2:end))*b(2:end);
        dof=length(v(2:end,2:end));
        pval = 1-cdf('chi2',w, dof); 
        R(i,:) = [w pval];
    end 
    
    
    disp('     R2   mean(simR2)  95% confidence interval');
    disp([R2CP mean(R2hCP')' R2CPconf]);
    disp('     Wald   largeT-p    smallT-p');
    disp([WCP(:,1) WCP(:,3) WCPhpvalue']);
    disp('     Simulated Wald, p-val');
    disp(R);
    disp('     CP parameter estimates for beta');
    disp('     betas 1 2: data,  mean,  std ');
    disp([betasCP(1,:)' betasCPh(1,:)' stdbetasCPh(1,:)' betasCP(2,:)' betasCPh(2,:)' stdbetasCPh(2,:)']);
    disp('     betas 3 4');
    disp([betasCP(3,:)' betasCPh(3,:)' stdbetasCPh(3,:)' betasCP(4,:)' betasCPh(4,:)' stdbetasCPh(4,:)']);
    disp('     betas 5 6');
    disp([betasCP(5,:)' betasCPh(5,:)' stdbetasCPh(5,:)' betasCP(6,:)' betasCPh(6,:)' stdbetasCPh(6,:)']);    

disp('----------------------------------------------------');      
    disp('Table 2: Estimates of the return-forecasting factor');

    disp('average hprx on f');
    disp('     R2   mean(simR2)  95% confidence interval');
    disp([R2CPav mean(R2havCP') R2CPavconf]);
    disp('     Wald   largeT-p    smallT-p');
    disp([WCPrest(:,1) WCPrest(:,3) WCPresthpvalue']);

    disp('     Lag, sim. Wald & its aympt. p-value');
    vg=vgCPh(2:end,2:end);
    vgL=vgCPhL(2:end,2:end);
    vgLL=vgCPhLL(2:end,2:end);
    
    F=gammas(2:end)'*inv(vg)*gammas(2:end);
    dof = size(gCPh(2:end),1); 
    pval = 1-cdf('chi2',F, dof); 

    FL=gammasL(2:end)'*inv(vgL)*gammasL(2:end);
    dofL = size(gCPhL(2:end),1); 
    pvalL = 1-cdf('chi2',FL, dof); 

    FLL    = gammasLL(2:end)'*inv(vgLL)*gammasLL(2:end);
    dofLL  = size(gCPhLL(2:end),1); 
    pvalLL = 1-cdf('chi2',FLL, dofLL); 

    disp([0 F pval; 1 FL pvalL; 2 FLL pvalLL]);
    

   
    disp('     gammas: data, mean, std');
    disp([gammas gCPh stdgCPh]);
    
    disp('     hprx(n) on gamma*f');
    disp('     R2   mean(simR2)  95% confidence interval');
    disp([R2CPres mean(R2hCPres')' R2CPresconf]);
 
    disp('     bs: data, mean, std');
    disp([bCP bCPh stdbCPh]);

disp('-----------------------------------------------------');    
disp('Table 3: Fama-Bliss excess return regressions');

    for i=1:4
        v=vFBh(:,:,i);
        b=bFB(:,i);
        w=b(2:end)'*inv(v(2:end,2:end))*b(2:end);
        dof=length(v(2:end,2:end));
        pval = 1-cdf('chi2',w, dof); 
        R(i,:) = [w pval];
    end 


    disp('     R2   mean(simR2)  95% confidence interval');
    disp([R2FB mean(R2hFB')' R2FBconf]);
    disp('    Wald   largeT-p    smallT-p');
    disp([WFB(:,1) WFB(:,3) WFBhpvalue']);
    disp('    Simulated Wald, p-val ');
    disp(R);
    disp('   betas, mean(betas), smallT-std(betas)');
    disp([bFB(1,:)' bFBh(1,:)' stdFBh(1,:)' bFB(2,:)' bFBh(2,:)' stdFBh(2,:)']);

disp('----------------------------------------------------');
disp('Table 5: Forecasting short rate changes');

    w=bFBy(2)*inv(varFByh(2:end,2:end))*bFBy(2);        
    dof=length(varFByh(2:end,2:end));
    pval = 1-cdf('chi2',w, dof); 
    R = [w pval];


disp('FB');
    disp('     R2   mean(simR2)  95% confidence interval');
    disp([R2FBy mean(R2hFBy')' R2FByconf]);
    disp('    Wald   largeT-p    smallT-p');
    disp([WFBy(:,1) WFBy(:,3) WFByhpvalue']);
    disp(R);
    disp('   betas, mean(betas), smallT-std(betas)');   
    disp([bFBy(1) bFByh(1) stdFByh(1) bFBy(2) bFByh(2) stdFByh(2)]);

    
    w=betasCPy(2:end)'*inv(varCPyh(2:end,2:end))*betasCPy(2:end);        
    dof=length(varCPyh(2:end,2:end));
    pval = 1-cdf('chi2',w, dof); 
    R = [w pval];

    
disp('CP');
    disp('     R2   mean(simR2)  95% confidence interval');
    disp([R2CPy mean(R2hCPy')' R2CPyconf]);
    disp('     Wald   largeT-p    smallT-p');
    disp([WCPy(:,1) WCPy(:,3) WCPyhpvalue']);
    disp(R);
    disp('     CP parameter estimates for beta');
    disp([betasCPy betasCPhy' stdbetasCPhy']);
      
disp('----------------------------------------------------');      

   disp('Table 12 - testing gammaf');

   WRh=(bun-br)'*inv(cov(bunCP))*(bun-br);  % MP I changed bun to bun-br here and below
   pWRh=1-cdf('chi2',WRh,length(bun));  
    
   JTh=gt'*pinv2(cov(meangth),15)*gt; % pinv because covariance matrix is supposed to be singular
   pJTh=1-cdf('chi2',JTh,15);  
  
   WRhL=(bunL-brL)'*inv(cov(bunCPL))*(bunL-brL);
   pWRhL=1-cdf('chi2',WRhL,length(bunL));  
    
   JThL=gtL'*pinv2(cov(meangthL),15)*gtL;
   pJThL=1-cdf('chi2',JThL,15);  

   WRhLL=(bunLL-bunL)'*inv(cov(bunCPLL))*(bunLL-bunL);
   pWRhLL=1-cdf('chi2',WRhLL,length(bunLL));  
    
   JThLL=gtLL'*pinv2(cov(meangthLL),15)*gtLL;
   pJThLL=1-cdf('chi2',JThLL,15);  
   
   fprintf('0, JT   data, p-val, JT-sim,   p-val  %8.2f %8.2f %8.2f  %8.2f \n', JT(:,1), JT(:,3), JTh, pJTh); 
   fprintf('0, Wald-data, p-val, Wald-sim, p-val  %8.2f %8.2f %8.2f  %8.2f \n', WR(:,1), WR(:,3), WRh, pWRh); 
   
   fprintf('1, JT   data, p-val, JT-sim,   p-val  %8.2f %8.2f %8.2f  %8.2f \n', JTL(:,1), JTL(:,3), JThL, pJThL); 
   fprintf('1, Wald-data, p-val, Wald-sim, p-val  %8.2f %8.2f %8.2f  %8.2f \n', WRL(:,1), WRL(:,3), WRhL, pWRhL); 
   
   fprintf('2, JT   data, p-val, JT-sim,   p-val  %8.2f %8.2f %8.2f  %8.2f \n', JTLL(:,1), JTLL(:,3), JThLL, pJThLL); 
   fprintf('2, Wald-data, p-val, Wald-sim, p-val  %8.2f %8.2f %8.2f  %8.2f \n', WRLL(:,1), WRLL(:,3), WRhLL, pWRhLL); 

   disp('diagnostics for Table 12'); 
   disp('using only diagonal elements of variance covariance matrices')
   
   WRh=(bun-br)'*inv(diag(diag(cov(bunCP))))*(bun-br);  
   pWRh=1-cdf('chi2',WRh,length(bun));  
    
   JTh=gt'*pinv2(diag(diag(cov(meangth))),15)*gt; 
   pJTh=1-cdf('chi2',JTh,15);  
  
   WRhL=(bunL-brL)'*inv(diag(diag(cov(bunCPL))))*(bunL-brL);
   pWRhL=1-cdf('chi2',WRhL,length(bunL));  
    
   JThL=gtL'*pinv2(diag(diag(cov(meangthL))),15)*gtL;
   pJThL=1-cdf('chi2',JThL,15);  

   WRhLL=(bunLL-bunL)'*inv(diag(diag(cov(bunCPLL))))*(bunLL-bunL);
   pWRhLL=1-cdf('chi2',WRhLL,length(bunLL));  
    
   JThLL=gtLL'*pinv2(diag(diag(cov(meangthLL))),15)*gtLL;
   pJThLL=1-cdf('chi2',JThLL,15);  
   
   fprintf('0, JT-sim,   p-val  %8.2f %8.2f  \n',  JTh, pJTh); 
   fprintf('0, Wald-sim, p-val  %8.2f %8.2f  \n',  WRh, pWRh); 
   
   fprintf('1, JT-sim,   p-val  %8.2f  %8.2f \n',  JThL, pJThL); 
   fprintf('1, Wald-sim, p-val  %8.2f  %8.2f \n',  WRhL, pWRhL); 
   
   fprintf('2, JT-sim,   p-val  %8.2f  %8.2f \n',  JThLL, pJThLL); 
   fprintf('2, Wald-sim, p-val  %8.2f  %8.2f \n',  WRhLL, pWRhLL); 
   
   disp('which linear combinations are causing the most trouble -- chi squared components'); 
   % here's the idea. Write a test as x'inv(V)*x = x'*inv(Q L Q')*x = x'Q *L^-1 * Q'x = sum(l_i^-1 * (q_i'x)^2). Thus, 
   % the chi2 test is this sum, and we can look at the biggest elements to see who is causing the most trouble. It will 
   % usually be the smallest eigenvalue, except if q_i'x = 0, that won't matter -- it is the product of the two that matters. 
   % so rank things by the product l_i^-1 * (q_i'x)^2; see if one or two is causing problems, look at the corresponding q_i. Those
   % are the linear combinations of parameters that are really causing the rejections. 
   
   [Q,L] = eig(cov(bunCP)); 
   qb = Q*(bun-br); 
   qbl = (qb.^2)./diag(L); 
   [sqbl,indx] = sort(qbl); 
   disp('first 4 elements of JT-sim, no lags. First, the contribution to JT and total JT.'); 
   disp([sqbl(end-4:end)' sum(sqbl)]); 
   disp('now the 4 eigenvectors q with bun, br, bun-br next to them'); 
   disp([Q(:,indx(end-4:end)) bun br (bun-br)]); 
   
disp('----------------------------------------------------');      

   
   disp('Table 8 - Testing other restrictions');
    
    % V = cov(gammastar)
    % unconstrained variance of gammastar=[gammastar0; gammastar1-5]
    % ahprx = gammastar0 + gammastar1-5'*yields + error 
    
    V=cov(gammastarh);
    
    % ahprx = delta0 + delta1-5'*(level slope y3 y4 y5), x(t) = (1 level slope y3 y4 y5)
    % [delta0; delta1-5]' x(t) = [gammastar0; gammastar1-5]'*y(t)
    % x(t) = A*y(t)
    % delta'*A  = gammstar'
    % A'*delta = gammastar
    % delta = inv(A')*gammastar
    
    for i=1:6
        
        a=A(:,:,i);
        v=inv(a')*V*inv(a')';
        v = v(N(i):end,N(i):end);
        b = blarge(N(i):end,i);
        
        F(i)   = b'*inv(v)*b;
        dof(i) = length(v);
        pr(i)  = 1-cdf('chi2',F(i),dof(i));
        
    end
    

    % get asymptotic p-values 
    disp('Rows in Table 8');
    disp([(1:6)' F' pr'])


diary off;
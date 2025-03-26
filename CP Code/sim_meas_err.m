% sim_meas_error. Shows what patterns of coefficients can be induced by measurement error in prices. 

close all; 
clear all; 
T = 20000; 
cormx = rand(5,5); % induces random correlation between variables at time t. Note this typically induces weird behavior in small s
% samples because of collinearity in right hand variables. Increase simulation size a lot to see the right pattern. 
cormx = eye(5); 
disp('simulating measurment error'); 
disp('covariance matrix of measurement error across maturity'); 
disp(cormx*cormx'); 

p = randn(T,5)*cormx; % i.i.d. across time measurement error
f = [-p(:,1) p(:,1)-p(:,2) p(:,2)-p(:,3) p(:,3)-p(:,4) p(:,4)-p(:,5)]; 
y = [-p(:,1) -p(:,2)/2 -p(:,3)/3 -p(:,4)/4 -p(:,5)/5]; 
ft = [ones(T-1,1) f(1:end-1,:)];
yt = [ones(T-1,1) y(1:end-1,:)];
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
f = randn(T,5)*cormx; % i.i.d. one year yield and 5 forward rates. 
p = -cumsum(f')';  
 %    p(1) = -f(1)
 %    p(2) = p(1) + p(2) - p(1) = - f(1) - f(2) etc. 
 
%f = [-p(:,1) p(:,1)-p(:,2) p(:,2)-p(:,3) p(:,3)-p(:,4) p(:,4)-p(:,5)]; reproduces the original f, a good check!
ft = [ones(T-1,1) f(1:end-1,:)];
rx = p(2:end,1:end-1) - p(1:end-1,2:end) + p(1:end-1,1)*ones(1,4); 
y = [-p(:,1) -p(:,2)/2 -p(:,3)/3 -p(:,4)/4 -p(:,5)/5]; 
yt = [ones(T-1,1) y(1:end-1,:)];

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


% these figures are here only to verify by simulation that the exact figures I make next are right. 

b = [ 0 0 0 0 ; 
      1 1 1 1 ; 
      0 1 1 1 ; 
      0 0 1 1 ; 
      0 0 0 1 ];
       
     
figure; 
subplot(2,1,1); 
plot((1:5)',b(:,4)+0.6,'-vb',...
     (1:5)',b(:,3)+0.4,'-^g',... 
     (1:5)',b(:,2)+0.2,'-or',... 
     (1:5)',b(:,1)+0.0,'-*m','Linewidth',1.5); 
 legend('n=5','n=4','n=3','n=2',2); 
 hold on; 

 plot([0.5 5.5]',zeros(2,1),'--k',...
     [0.5 5.5]',zeros(2,1)+0.2,'--k',... 
     [0.5 5.5]',zeros(2,1)+0.4,'--k',... 
     [0.5 5.5]',zeros(2,1)+0.6,'--k'); 
set(gca,'xtick',[1 2 3 4 5]); 
set(gca,'ytick',[0 0.2 0.4 0.6 1 1.2 1.4 1.6]); 
set(gca,'yticklabel',[ 0 0 0 0 1 1 1 1]); 
axis([ 0.5 5.5 -0.2 1.8]); 
xlabel('Maturity'); 
ylabel('Coefficient'); 
title('Forward rates'); 

b = [ -1 -1 -1 -1 ; 
       2  0  0  0 ; 
       0  3  0  0 ; 
       0  0  4  0 ; 
       0  0  0  5 ];

subplot(2,1,2); 
plot((1:5)',b(:,4)+0.0,'-vb',...
     (1:5)',b(:,3)+0.0,'-^g',... 
     (1:5)',b(:,2)+0.0,'-or',... 
     (1:5)',b(:,1)+0.0,'-*m','Linewidth',1.5); 
 legend('n=5','n=4','n=3','n=2',2); 
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
title('Yields'); 


print -depsc2 sim_meas_err.eps; 
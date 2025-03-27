function c=corr(y,x);

a=corrcoef([y x]);
c=a(2,1);
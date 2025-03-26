function [u0,u1] = ycoeff(h,A0,A1);

N=size(A1,1);

u0=A0;

for i=1:(h-1)
    u0=u0+(A1^i)*A0;
end

u1 = A1^h;
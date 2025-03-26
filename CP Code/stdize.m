function [x]=stdize(y);

x=(y-mean(y))/std(y);
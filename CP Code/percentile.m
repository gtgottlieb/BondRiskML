function res = percentile(data,pct)


% PERCENTILE(data,pct) 
% give the indicated percentile of the data
% modified code from resample website

if any(pct > 100.0 | pct < 0.0)
   error('Percentile: pct must be between 0.0 and 1.0');
end
if any( pct > 1 )
  pct = pct/100; % convert from percent to a fraction;
  disp('Percentile: converting from percent to a fraction.')
end

data = sort(data);

foo = pct*(length(data)-1) + 1;

res = data(round(foo),:);
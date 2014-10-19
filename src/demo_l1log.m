clear;
pdir = './';
fname = {'a9a'};
ext = '';

% verbose
param.v = 1;
% optimal tol
param.e = 1e-7;
% lambda
param.lambda = 1;
% max iterations
param.i = 3000;
param.loss = 'square';
param.cached = 0;
param.filename = [pwd, '/', fname{1}, ext];

[w, iter, fval, t] = LHACl1log(param);



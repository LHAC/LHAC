clear;
pdir = './';
fname = {'a9a'};
ext = '';

% verbose
param.v = 2;
% optimal tol
param.e = 1e-7;
% lambda
param.lmd = 1;
% max iterations
param.i = 3000;

datafile = [pwd, '/', fname{1}, ext];

[w, iter, fval, t] = LHACl1log(datafile, param.lmd, param);



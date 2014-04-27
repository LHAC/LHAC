clear;
pdir = './';
fname = {'scaledp1'};
ext = '.csv';

% verbose
param.v = 2;
% optimal tol
param.e = 1e-6;
% max iterations
param.i = 1000;

% lambda
param.lmd = 0.5;

% use load for other files, e.g., ER_692
fin = [pdir fname{1} ext];
% read the data
S = dlmread(fin);

W = LHAC(S, param.lmd, param);



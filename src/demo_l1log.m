clear;
pdir = './';
fname = {'a9a'};
ext = '';

% verbose
param.v = 0;
% optimal tol
param.e = 1e-7;
% lambda
param.lambda = 0.0001;
% max iterations
param.i = 3000;
% param.loss = 'square';
param.loss = 'log';
% pre-compute ATA in lasso
param.cached = 0;
param.filename = [pwd, '/', fname{1}, ext];
param.weight = 2;

% reading data directly from file using param.filename
[w] = lhac(param);
% passing data from memory in MATLAB 
load a9a
[w, iter, fval, t] = lhac(param, data.y, full(data.X));
fprintf('time = %.4e, iter = %d, optimal fval = %.4e\n', t(end), iter(end), fval(end));
fprintf('Verify that the optimal objective value is around 4.7096e-01!\n');



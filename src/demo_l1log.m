clear;
pdir = './';
fname = {'a9a'};
ext = '';

% verbose
param.v = 0;
% optimal tol
param.e = 1e-7;
% lambda
param.lambda = 1;
% max iterations
param.i = 3000;
% param.loss = 'square';
param.loss = 'log';
param.cached = 0;
param.filename = [pwd, '/', fname{1}, ext];

fprintf('test 1 ---- reading data directly from file ---- 0%%...\n');
[w] = lhac(param);
fprintf('test 1 ---- reading data directly from file ---- 100%%!\n');
fprintf('test 2 ---- passing data from memory in MATLAB ---- 0%%...\n');
load a9a
[w, iter, fval, t] = lhac(param, data.y, full(data.X));
fprintf('time = %.4e, iter = %d, optimal fval = %.4e\n', t(end), iter(end), fval(end));
fprintf('test 2 ---- passing data from memory in MATLAB ---- 100%%!\n');



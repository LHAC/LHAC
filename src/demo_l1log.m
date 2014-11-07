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
% param.loss = 'square';
param.loss = 'log';
param.cached = 0;
param.filename = [pwd, '/', fname{1}, ext];


[w] = lhac(param);
fprintf('test 1 ---- reading data directly from file ---- finished!\n');
fprintf('test 2 ---- passing data from memory in MATLAB ---- start!\n');
load a9a
param.v = 0;
[w, iter, fval, t] = lhac(param, data.y, full(data.X));
fprintf('time = %.4e, iter = %d, optimal fval = %.4e\n', t(end), iter(end), fval(end));



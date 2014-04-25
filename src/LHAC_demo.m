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
param.g = 6;
param.l = 1;

% lambda
param.lmd = 0.7;


for ii=1:length(fname)
	algs(1).name = 'LHAC';
	algs(1).exps = [];

	tmp = struct();
	tmp.name = fname{ii};
	% day-hour-minute
	fout = [pwd, '/outputs/', fname{ii}, '-lmd7-', datestr(now, 'dd-HH-MM')];
	% use load for other files, e.g., ER_692
	fin = [pdir fname{ii} ext];
	% read the data
	S = dlmread(fin);

	[tmp.W, tmp.iter tmp.fval tmp.t tmp.normgs tmp.numActive] = LHAC(S, param.lmd, param);
	tmp.W = sparse(tmp.W);
	tmp.param = param;
	algs(1).exps = [algs(1).exps tmp];

	save(fout, 'algs');
end


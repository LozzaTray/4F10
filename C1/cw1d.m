meanfunc = [];                    % empty: don't use a mean function
covfunc = {@covProd, {@covPeriodic, @covSEiso}};  % Periodic * SE
likfunc = @likGauss;              % Gaussian likelihood

% initial params
cov = [-0.5, 0, 0, 2, 0];
lik = 0;

% define hyperparams
hyp = struct('mean', [], 'cov', cov, 'lik', lik);

n = 200;
seed = 0.8;
x = gpml_randn(seed, n, 1);
%x = linspace(-5, 5, n)'; 
K = feval(covfunc{:}, hyp.cov, x) + 1e-6 * eye(n);
y = chol(K)'*x;

xs = linspace(-5, 5, n)';      % test input range
[nlZ,dnlZ,post] = gp(hyp, @infGaussLik, meanfunc, covfunc, likfunc, x, y);
[ymu,ys2,fmu,fs2,junk,post] = gp(hyp, @infGaussLik, meanfunc, covfunc, likfunc, x, post, xs);
%[ymu, ys2, fmu, fs2] = gp(hyp, @infGaussLik, meanfunc, covfunc, likfunc, x, y, xs);
f = [fmu+2*sqrt(fs2); flip(fmu-2*sqrt(fs2),1)];
%fill([xs; flip(xs,1)], f, [7 7 7]/8)
hold on;
grid on;
plot(xs, ymu);
plot(x, y, '+');
xlabel('Input x');
ylabel('Sampled Function f(x)');
%legend('Sampled Function', 'Training Datapoints');
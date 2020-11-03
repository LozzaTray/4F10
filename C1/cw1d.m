meanfunc = [];                    % empty: don't use a mean function
covfunc = {@covProd, {@covPeriodic, @covSEiso}};  % Periodic * SE
likfunc = @likGauss;              % Gaussian likelihood

% initial params
cov = [-0.5, 0, 0, 2, 0];
lik = 0;

% define hyperparams
hyp = struct('mean', [], 'cov', cov, 'lik', lik);

n = 20;
seed = 0.7;
x = gpml_randn(seed, n, 1);
K = feval(covfunc{:}, hyp.cov, x);
y = chol(K)'*x;

xs = linspace(-5, 5, 200)';      % test input range
[ymu, ys2, fmu, fs2] = gp(hyp, @infGaussLik, meanfunc, covfunc, likfunc, x, y, xs);
hold on;
grid on;
plot(xs, ymu);
plot(x, y, '+');
xlabel('Input x');
legend('Sampled Function', 'Training Datapoints');
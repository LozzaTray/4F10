meanfunc = [];                    % empty: don't use a mean function
covfunc = {@covProd, {@covPeriodic, @covSEiso}};  % Periodic * SE
likfunc = @likGauss;              % Gaussian likelihood

% initial params
cov = [-0.5, 0, 0, 2, 0];
lik = 0;

% minimise likelihood
hyp = struct('mean', [], 'cov', cov, 'lik', lik);

x = linspace(-5, 5, 200);      % test input range
[mu s2] = gp(hyp, @infGaussLik, meanfunc, covfunc, likfunc, x, []);
f = [mu+2*sqrt(s2); flip(mu-2*sqrt(s2),1)];
fill([xs; flip(xs,1)], f, [7 7 7]/8)
hold on;
grid on;
plot(xs, mu);
plot(x, y, '+')
xlabel('Input x');
legend('95% Error Bounds', 'Function Mean', 'Training Datapoints');
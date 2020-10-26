load("cw1a.mat")

meanfunc = [];                    % empty: don't use a mean function
covfunc = @covPeriodic;           % Periodic
likfunc = @likGauss;              % Gaussian likelihood

% initial params
cov = [0, 0, 0]; % [log(l), log(p), log(v)]
lik = 0;

% minimise likelihood
hyp = struct('mean', [], 'cov', cov, 'lik', lik);
hyp2 = minimize(hyp, @gp, -100, @infGaussLik, meanfunc, covfunc, likfunc, x, y);

disp("Trained hyperparameters:");
disp(hyp2);

xs = linspace(-3, 3, 1000)';      % test input range
[mu s2] = gp(hyp2, @infGaussLik, meanfunc, covfunc, likfunc, x, y, xs);
f = [mu+2*sqrt(s2); flip(mu-2*sqrt(s2),1)];
fill([xs; flip(xs,1)], f, [7 7 7]/8)
hold on;
grid on;
plot(xs, mu);
plot(x, y, '+')
xlabel('Input x');
legend('95% Error Bounds', 'Function Mean', 'Training Datapoints');
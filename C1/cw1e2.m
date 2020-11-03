load("cw1e.mat")

% covSEard
meanfunc = [];                    % empty: don't use a mean function
covfunc = {@covSum, {@covSEard, @covSEard}};           % Periodic
likfunc = @likGauss;              % Gaussian likelihood

% initial params
cov = 0.1 * randn(6, 1); % [log(ell_1), log(ell_2), log(sf)]
lik = 0;

% minimise likelihood
hyp = struct('mean', [], 'cov', cov, 'lik', lik);
hyp2 = minimize(hyp, @gp, -100, @infGaussLik, meanfunc, covfunc, likfunc, x, y);

disp("Trained hyperparameters:");
disp(hyp2);

%plotting
n = 100;
first_dim = linspace(-3, 3, n*n);      % test input range
second_dim = linspace(-3, 3, n);
second_dim = repmat(second_dim, 1, n);
xs = zeros(n*n, 2);
xs(: ,1) = first_dim;
xs(:, 2) = second_dim;

[mu s2] = gp(hyp2, @infGaussLik, meanfunc, covfunc, likfunc, x, y, xs);

% visualize
x1 = reshape(first_dim, n, n);
x2 = reshape(second_dim, n, n);
mugrid = reshape(mu, n, n);
sqrt_grid = reshape(sqrt(s2), n, n);

% visualize error
hold on;
%scatter3(x(:,1), x(:,2), y, '+');
%mesh(x1, x2, mugrid);
mesh(x1, x2, sqrt_grid);
xlabel("x1");
ylabel("x2");
%zlabel("y");
zlabel("std dev");
hold off;
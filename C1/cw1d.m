% initial params
covFunc = {@covProd, {@covPeriodic, @covSEiso}};
cov = [-0.5, 0, 0, 2, 0];

n = 2000;

seed = 0.9;
x = linspace(-5, 5, n)';
z = gpml_randn(seed, n, 1);
K = feval(covfunc{:}, cov, x);
K_pos_def = K + 1e-6 * eye(n);
y = chol(K_pos_def)'*z;

plot(x, y);
xlabel('Input x');
ylabel('Sampled Function f(x)');
title('Seed = 0.9');
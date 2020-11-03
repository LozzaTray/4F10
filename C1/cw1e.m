load("cw1e.mat")

% visualize
x1 = reshape(x(:,1), 11, 11);
x2 = reshape(x(:,2), 11, 11);
ygrid = reshape(y, 11, 11);
mesh(x1, x2, ygrid);
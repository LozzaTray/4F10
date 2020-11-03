load("cw1e.mat")

% visualize
x1 = reshape(x(:,1), 11, 11);
x2 = reshape(x(:,2), 11, 11);
ygrid = reshape(y, 11, 11);

hold on;
scatter3(x(:,1), x(:,2), y, '+');
mesh(x1, x2, ygrid);
xlabel("x1");
ylabel("x2");
zlabel("y");
hold off;
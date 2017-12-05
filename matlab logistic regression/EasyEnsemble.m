function [X, y] = EasyEnsemble(X_train, y_train, n_subsets, balance)
X = [];
y = [];

X0 = X_train(y_train==0, :);
X1 = X_train(y_train==1, :);
n_class0 = size(X0, 1);
n_class1 = size(X1, 1);

if nargin == 3
    assert(n_subsets == 1);
    X{n_subsets} = X_train;
    y{n_subsets} = y_train;
elseif strcmp(balance, 'true')
    for i = 1:n_subsets
        idx = randperm(n_class0, n_class1);
        X0_sampled = X0(idx, :);
        new_X = [X0_sampled; X1];
        new_y = [zeros(n_class1, 1); ones(n_class1, 1)];
    
        idx = randperm(n_class1 * 2);
        new_X = new_X(idx, :);
        new_y = new_y(idx, :);
        X{i} = new_X;
        y{i} = new_y;
    end
elseif strcmp(balance, 'false')
    for i = 1:n_subsets
        idx1 = randi(n_class1, n_class1, 1);
        idx0 = randi(n_class0, n_class0, 1);
        X1_sampled = X1(idx1, :);
        X0_sampled = X0(idx0, :);
        new_X = [X0_sampled; X1_sampled];
        new_y = [zeros(n_class0, 1); ones(n_class1, 1)];
    
        idx = randperm(n_class1 + n_class0);
        new_X = new_X(idx, :);
        new_y = new_y(idx, :);
        X{i} = new_X;
        y{i} = new_y;
    end
end
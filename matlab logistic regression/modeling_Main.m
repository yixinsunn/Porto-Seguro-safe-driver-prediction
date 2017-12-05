initPmtk3;
clear; clc; close all;

%% Read data, and drop 'ps_calc_' features
train = csvread('..\input\train.csv', 1, 0);
validation = csvread('..\input\validation.csv', 1, 0);
train(:, 28:47) = []; validation(:, 28:47) = [];

X_train = train(:, 2:end); y_train = train(:, 1);
X_validation = validation(:, 2:end); y_validation = validation(:, 1);

% %% PCA
% % Standardizatioin
% [X_train, mu, sigma] = standardizeCols(X_train);
% X_validation = standardizeCols(X_validation, mu, sigma);
% 
% K = 160;  % 140, 150, 160, 170
% [~, eigVec] = pca(X_train);
% X_train = X_train * eigVec(:, 1:K);
% X_validation = X_validation * eigVec(:, 1:K);

%% Ensemble method
n_subsets = 50;
[X, y] = EasyEnsemble(X_train, y_train, n_subsets, 'true');

%% Select optimal alpha and lambda
alphas = [11, 11.24190278, 11.48912529, 11.74178451, 12];
lambdas = [17, 17.27227683, 17.54891451, 17.82998291, 18.11555298, 18.40569682, 18.70048768, 19];
gini_trn = zeros(size(alphas, 2), size(lambdas, 2));
gini_val = zeros(size(alphas, 2), size(lambdas, 2));

for i = 1:size(alphas, 2)
    for j = 1:size(lambdas, 2)
        prob_trn = zeros(size(y_train, 1), 1);
        prob_val = zeros(size(y_validation, 1), 1);
        for k = 1:n_subsets
            model = logregFit(X{k}, y{k}, 'regType', 'both', ...
                'alpha', alphas(i), ...
                'lambda', lambdas(j));
            [~, prob1] = logregPredict(model, X_train);
            [~, prob2] = logregPredict(model, X_validation);
            prob_trn = prob_trn + prob1;
            prob_val = prob_val + prob2;
        end
        prob_trn = prob_trn / n_subsets;
        prob_val = prob_val / n_subsets;
        gini_trn(i, j) = gini_normalized(y_train, prob_trn);
        gini_val(i, j) = gini_normalized(y_validation, prob_val);
        fprintf('Gini coefficient on entire training set: %2.4f\n', ...
            gini_trn(i, j));
        fprintf('Gini coefficient on validation set: %2.4f\n', ...
            gini_val(i, j));
    end
end

[i, j] = find(gini_val == max(max(gini_val)));
optimal_alpha = alphas(i);
optimal_lambda = lambdas(j);

%% Use the optimal alpha and lambda to train on full training set
X_train = [X_train; X_validation];     % combine training set and
y_train = [y_train; y_validation];     % validation set to get the full set

test = csvread('..\input\test.csv', 1, 0); test(:, 28:47) = [];  % read test set
X_test = test(:, 2:end); y_test = test(:, 1);

n_subsets = 50;
[X, y] = EasyEnsemble(X_train, y_train, n_subsets, 'true');

prob_train = zeros(size(y_train, 1), 1);
prob_test = zeros(size(y_test, 1), 1);
for k = 1:n_subsets
    model = logregFit(X{k}, y{k}, 'regType', 'both', ...
        'alpha', optimal_alpha, ...
        'lambda', optimal_lambda);
    [~, prob1] = logregPredict(model, X_train);
    [~, prob2] = logregPredict(model, X_test);
    prob_train = prob_train + prob1;
    prob_test = prob_test + prob2;
end

prob_train = prob_train / n_subsets;
prob_test = prob_test / n_subsets;
fprintf('Gini coefficient on entire training set: %2.8f\n', ...
    gini_normalized(y_train, prob_train));
fprintf('Gini coefficient on test set: %2.8f\n', ...
    gini_normalized(y_test, prob_test));

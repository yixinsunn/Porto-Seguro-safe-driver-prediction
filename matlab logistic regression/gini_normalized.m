function gini_coefficient = gini_normalized(actual, pred)
gini_coefficient = gini(actual, pred) / gini(actual, actual);

function gini_coefficient = gini(actual, pred)
assert(size(actual, 1) ~= 1);
assert(size(actual, 2) == size(pred, 2));
n = size(actual, 1);

all = [actual, pred, [1:n]'];
all = sortrows(all, [-2, 3]);
totalLosses = sum(all(:, 1));
giniSum = sum(cumsum(all(:, 1))) / totalLosses;

giniSum = giniSum - (n + 1) / 2;
gini_coefficient = giniSum / n;
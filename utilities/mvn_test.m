function X_mvn = mvn_test(X, mu, stdev)

% X should be frames x feature_dimension
% i.e. examples are along rows of X

X_mvn = bsxfun(@minus, X, mu);
X_mvn = bsxfun(@rdivide, X_mvn, stdev + eps);
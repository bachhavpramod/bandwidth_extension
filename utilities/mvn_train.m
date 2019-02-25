function [X_mvn, mu, stdev] = mvn_train(X)

% X should be frames x feature_dimension
% i.e. examples are along rows of X

stdev = std(X);
mu = mean(X);
X_mvn = bsxfun(@minus, X,mu);
X_mvn = bsxfun(@rdivide, X_mvn, stdev + eps);
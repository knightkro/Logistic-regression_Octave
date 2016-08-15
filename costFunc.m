function [J, grad] = costFunc(theta, X, y)
%COSTFUNCTION Compute cost and gradient for logistic regression
% Initialize some useful values
m = length(y); % number of training examples
J =(-y'*log(sigmoid(X*theta))-(1-y)'*log(1-sigmoid(X*theta)))/m;
grad = (((sigmoid(X*theta) - y)'*X)')/m;
% =============================================================
end

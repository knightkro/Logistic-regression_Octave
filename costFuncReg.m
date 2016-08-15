function [J, grad] = costFuncReg(theta, X, y, lambda)
%COSTFUNCREG Compute cost and gradient for logistic regression with regularization
% Initialize some useful values
m = length(y); % number of training examples
% You need to return the following variables correctly 
[J, grad] = costFunc(theta, X, y);
%ignore "theta zero" ie don't regualrise it:
theta_zeroed_first = [0; theta(2:length(theta));];
J = J + lambda / (2 * m) * sum( theta_zeroed_first .^ 2 );
grad = grad .+ (lambda / m) * theta_zeroed_first;
% =============================================================

end

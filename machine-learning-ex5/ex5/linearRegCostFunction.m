function [J, grad] = linearRegCostFunction(X, y, theta, lambda)
%LINEARREGCOSTFUNCTION Compute cost and gradient for regularized linear 
%regression with multiple variables
%   [J, grad] = LINEARREGCOSTFUNCTION(X, y, theta, lambda) computes the 
%   cost of using theta as the parameter for linear regression to fit the 
%   data points in X and y. Returns the cost in J and the gradient in grad

% Initialize some useful values
m = length(y); % number of training examples

% You need to return the following variables correctly 
J = 0;
grad = zeros(size(theta));

% ====================== YOUR CODE HERE ======================
% Instructions: Compute the cost and gradient of regularized linear 
%               regression for a particular choice of theta.
%
%               You should set J to the cost and grad to the gradient.
%


% theta size 2x1
% X size 12 x 1 (now 12 x 2)
% y size 12 x 1

% add bias term to X
% X = [ones(m, 1) X] % size 12 x 2 % Dont need to do this

h = X * theta; % 12 x 1
J = (h-y)' * (h-y);
J = (1/(2*m)) * J;
J = J + ((lambda/(2*m)) * (theta(2:end)' * theta(2:end)));

grad = (1/m) * X' * (h - y); % 2 x 1
grad(2:end) = grad(2:end) + (lambda/m) * theta(2:end);






% =========================================================================

grad = grad(:);

end

function [J, grad] = costFunctionReg(theta, X, y, lambda)
% function [J, grad, A, X, m, theta] = costFunctionReg(theta, X, y, lambda)
%COSTFUNCTIONREG Compute cost and gradient for logistic regression with regularization
%   J = COSTFUNCTIONREG(theta, X, y, lambda) computes the cost of using
%   theta as the parameter for regularized logistic regression and the
%   gradient of the cost w.r.t. to the parameters. 

% Initialize some useful values
m = length(y); % number of training examples

% You need to return the following variables correctly 
J = 0;
grad = zeros(size(theta));

% ====================== YOUR CODE HERE ======================
% Instructions: Compute the cost of a particular choice of theta.
%               You should set J to the cost.
%               Compute the partial derivatives and set grad to the partial
%               derivatives of the cost w.r.t. each parameter in theta


% size(theta) % 3 x 1
% size(X) % 100 x 3
% size(y) % 100 x 1

h = sigmoid(X * theta); % 100 x 1

y_pos_term = -y' * log(h); % 1 x 1
y_neg_term = -(1 - y)' * log(1 - h);

J = ((y_pos_term + y_neg_term) / m);
J_lambda = theta(2:end)' * theta(2:end) * lambda / (2*m);

J = J + J_lambda;

A = h - y;

grad(1,:) = X'(1,:)*A/m;
grad(2:end, :) = X'(2:end, :)*A/m + (lambda/m)*theta(2:end, :);




% =============================================================

end

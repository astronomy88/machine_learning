function [theta, J_history] = gradientDescent(X, y, theta, alpha, num_iters)
%GRADIENTDESCENT Performs gradient descent to learn theta
%   theta = GRADIENTDESCENT(X, y, theta, alpha, num_iters) updates theta by 
%   taking num_iters gradient steps with learning rate alpha

% Initialize some useful values
m = length(y); % number of training examples
J_history = zeros(num_iters, 1);

for iter = 1:num_iters

    % ====================== YOUR CODE HERE ======================
    % Instructions: Perform a single gradient step on the parameter vector
    %               theta. 
    %
    % Hint: While debugging, it can be useful to print out the values
    %       of the cost function (computeCost) and gradient here.
    %

    % fprintf('Size of X = %f\n', size(X))
    % fprintf('Size of y = %f\n', size(y))
    % fprintf('Size of theta = %f\n', size(theta))
    % fprintf('alpha = %f\n', alpha)
    % fprintf('num_iters = %f\n', num_iters)

    h = X * theta;
    A = h - y;
    grad = X' * A; % this is now a 2 x 1 matrix of the grad portion
    theta = theta - (alpha / m) * grad;

    % fprintf('Gradient: \n%f\n', grad)
    % fprintf('Theta: \n%f\n', theta)
    % fprintf('Cost function: \n%f\n', computeCost(X, y, theta))


    % ============================================================

    % Save the cost J in every iteration    
    J_history(iter) = computeCost(X, y, theta);

end

end

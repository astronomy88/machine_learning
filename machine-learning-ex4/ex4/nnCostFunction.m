function [J grad] = nnCostFunction(nn_params, ...
                                   input_layer_size, ...
                                   hidden_layer_size, ...
                                   num_labels, ...
                                   X, y, lambda)
%NNCOSTFUNCTION Implements the neural network cost function for a two layer
%neural network which performs classification
%   [J grad] = NNCOSTFUNCTON(nn_params, hidden_layer_size, num_labels, ...
%   X, y, lambda) computes the cost and gradient of the neural network. The
%   parameters for the neural network are "unrolled" into the vector
%   nn_params and need to be converted back into the weight matrices. 
% 
%   The returned parameter grad should be a "unrolled" vector of the
%   partial derivatives of the neural network.
%

% Reshape nn_params back into the parameters Theta1 and Theta2, the weight matrices
% for our 2 layer neural network
Theta1 = reshape(nn_params(1:hidden_layer_size * (input_layer_size + 1)), ...
                 hidden_layer_size, (input_layer_size + 1));

Theta2 = reshape(nn_params((1 + (hidden_layer_size * (input_layer_size + 1))):end), ...
                 num_labels, (hidden_layer_size + 1));

% Setup some useful variables
m = size(X, 1); % 5000
         
% You need to return the following variables correctly 
J = 0;
Theta1_grad = zeros(size(Theta1));
Theta2_grad = zeros(size(Theta2));

% ====================== YOUR CODE HERE ======================
% Instructions: You should complete the code by working through the
%               following parts.
%
% Part 1: Feedforward the neural network and return the cost in the
%         variable J. After implementing Part 1, you can verify that your
%         cost function computation is correct by verifying the cost
%         computed in ex4.m
%

% X size 5000 x 400
% y size 5000 x 1
% Theta1 size 25 x 401
% Theta2 size 10 x 26
% y_new size is 5000 x 10
% a1 size is 5000 x 401

%-- We'll need to convert y to a 5000 x 10 matrix, since each row is a 1 x 10 vector
y_new = zeros(m, num_labels);

for i = 1:m
	y_new(i,y(i)) = 1;
end

%-- Add ones to our X training data
a1 = [ones(m,1) X]; % size 5000 x 401
z2 = a1 * Theta1'; % size 5000 x 25
a2 = sigmoid(z2); % size 5000 x 25
a2 = [ones(m,1) a2]; % size 5000 x 26
z3 = a2 * Theta2'; % size 5000 x 10
h = sigmoid(z3); % size 5000 x 10

J = -(y_new .* log(h)) - ((1-y_new) .* log(1 - h)); % 5000 x 10, and just sum them up
J = sum(sum(J)) / m;

%-- Now adding regularization - ignore first column in Theta1 and Theta2 (bias terms)
sum_Theta1 = sum(sum(Theta1(:,2:end) .* Theta1(:,2:end)));
sum_Theta2 = sum(sum(Theta2(:,2:end) .* Theta2(:,2:end)));
J = J + (lambda / (2*m)) * (sum_Theta1 + sum_Theta2);

%
% Part 2: Implement the backpropagation algorithm to compute the gradients
%         Theta1_grad and Theta2_grad. You should return the partial derivatives of
%         the cost function with respect to Theta1 and Theta2 in Theta1_grad and
%         Theta2_grad, respectively. After implementing Part 2, you can check
%         that your implementation is correct by running checkNNGradients
%
%         Note: The vector y passed into the function is a vector of labels
%               containing values from 1..K. You need to map this vector into a 
%               binary vector of 1's and 0's to be used with the neural network
%               cost function.
%
%         Hint: We recommend implementing backpropagation using a for-loop
%               over the training examples if you are implementing it for the 
%               first time.
%

%size(Theta1_grad); % 25 x 401
%size(Theta2_grad); % 10 x 26

% -- This is the correct way, but is way too slow
% for i = 1:m
% 	% delta3 = zeros(num_labels, 1);
% 	delta3 = h(i,:)' - y_new(i,:)';

% 	grad_z2 = sigmoidGradient([ones(m, 1) z2])(i,:);
% 	delta2 = (Theta2' * delta3) .* grad_z2';
% 	delta2 = delta2(2:end);

% 	Theta2_grad = Theta2_grad + (delta3 * a2(i,:)); % 10 x 26 
% 	Theta1_grad = Theta1_grad + (delta2 * a1(i,:)); % 25 x 401

% 	% Regularization - don't include first column
% 	Theta2_grad(:,2:end) = Theta2_grad(:,2:end) + ((lambda/m) * Theta2(:,2:end));
% 	Theta1_grad(:,2:end) = Theta1_grad(:,2:end) + ((lambda/m) * Theta1(:,2:end));

% end

%-- Doing the vectorization way, which is claimed to be 30x-50x faster
delta3 = h - y_new;
grad_z2 = sigmoidGradient([ones(m, 1) z2]);
delta2 = (delta3 * Theta2) .* grad_z2;
delta2 = delta2(:, 2:end);
Theta2_grad = Theta2_grad + (delta3' * a2);
Theta1_grad = Theta1_grad + (delta2' * a1);

Theta2_grad = Theta2_grad / m;
Theta1_grad = Theta1_grad / m;

% Regularization - don't include first column
Theta2_grad(:,2:end) = Theta2_grad(:,2:end) + ((lambda/m) * Theta2(:,2:end));
Theta1_grad(:,2:end) = Theta1_grad(:,2:end) + ((lambda/m) * Theta1(:,2:end));


% Part 3: Implement regularization with the cost function and gradients.
%
%         Hint: You can implement this around the code for
%               backpropagation. That is, you can compute the gradients for
%               the regularization separately and then add them to Theta1_grad
%               and Theta2_grad from Part 2.
%


















% -------------------------------------------------------------

% =========================================================================

% Unroll gradients
grad = [Theta1_grad(:) ; Theta2_grad(:)];


end

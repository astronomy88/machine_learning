%% Initialization
clear ; close all; clc

%% Setup the parameters you will use for this exercise
input_layer_size  = 400;  % 20x20 Input Images of Digits
hidden_layer_size = 25;   % 25 hidden units
num_labels = 10;          % 10 labels, from 1 to 10   
                          % (note that we have mapped "0" to label 10)

% Load Training Data
fprintf('Loading Data ...\n')

load('ex4data1.mat');
m = size(X, 1);

fprintf('\nLoading Saved Neural Network Parameters ...\n')

% Load the weights into variables Theta1 and Theta2
load('ex4weights.mat');

% X size 5000 x 400
% y size 5000 x 1
% Theta1 size 25 x 401
% Theta2 size 10 x 26
% y_new 5000 x 10

y_new = zeros(m, num_labels);

for i = 1:m
	y_new(i,y(i)) = 1;
end

a1 = [ones(m,1) X]; % size 5000 x 401
z2 = a1 * Theta1'; % size 5000 x 25
a2 = sigmoid(z2); % size 5000 x 25
a2 = [ones(m,1) a2]; % size 5000 x 26
z3 = a2 * Theta2'; % size 5000 x 10
h = sigmoid(z3); % size 5000 x 10

% Unroll parameters 
nn_params = [Theta1(:) ; Theta2(:)];

lambda = 0;

J = nnCostFunction(nn_params, input_layer_size, hidden_layer_size, ...
                   num_labels, X, y, lambda)

lambda = 1;

J = nnCostFunction(nn_params, input_layer_size, hidden_layer_size, ...
                   num_labels, X, y, lambda)

initial_Theta1 = randInitializeWeights(input_layer_size, hidden_layer_size);
initial_Theta2 = randInitializeWeights(hidden_layer_size, num_labels);

% Unroll parameters
initial_nn_params = [initial_Theta1(:) ; initial_Theta2(:)];


% delta3 = h(1,:)' - y_new(1,:)';
% grad_z2 = sigmoidGradient([ones(m, 1) z2])(1,:);
% delta2 = (Theta2' * delta3) .* grad_z2';
% delta2 = delta2(2:end);

% Theta2_grad = delta3 * a2(1,:);
Theta1_grad = zeros(size(Theta1));
Theta2_grad = zeros(size(Theta2));

delta3 = h - y_new;
grad_z2 = sigmoidGradient([ones(m, 1) z2]);
delta2 = (delta3 * Theta2) .* grad_z2;
delta2 = delta2(:, 2:end);
Theta2_grad = Theta2_grad + (delta3' * a2);
Theta1_grad = Theta1_grad + (delta2' * a1);

function [error_train, error_val] = ...
    randLearningCurve(X, y, Xval, yval, lambda)
%LEARNINGCURVE Generates the train and cross validation set errors needed 
%to plot a learning curve
%   [error_train, error_val] = ...
%       LEARNINGCURVE(X, y, Xval, yval, lambda) returns the train and
%       cross validation set errors for a learning curve. In particular, 
%       it returns two vectors of the same length - error_train and 
%       error_val. Then, error_train(i) contains the training error for
%       i examples (and similarly for error_val(i)).
%
%   In this function, you will compute the train and test errors for
%   dataset sizes from 1 up to m. In practice, when working with larger
%   datasets, you might want to do this in larger intervals.
%

% Number of training examples
m = size(X, 1);

% You need to return these values correctly
error_train = zeros(m, 1);
error_val   = zeros(m, 1);

% ====================== YOUR CODE HERE ======================
% Instructions: Fill in this function to return training errors in 
%               error_train and the cross validation errors in error_val. 
%               i.e., error_train(i) and 
%               error_val(i) should give you the errors
%               obtained after training on i examples.
%
% In practice, especially for small training sets, when you plot learning 
% curves to debug your algorithms, it is often helpful to average across 
% multiple sets of randomly selected examples to determine the training 
% error and cross validation error.

% Concretely, to determine the training error and cross validation error 
% for i examples, you should first randomly select i examples from the 
% training set and i examples from the cross validation set. You will 
% then learn the parameters θ using the randomly chosen training set 
% and evaluate the parameters θ on the randomly chosen training set 
% and cross validation set. The above steps should then be repeated 
% multiple times (say 50) and the averaged error should be used to 
% determine the training error and cross validation error for i examples.
% For this optional (ungraded) exercise, you should implement the above 
% strategy for computing the learning curves.

% ---------------------- Sample Solution ----------------------

for i = 1:m
	trials = 50;
	sum_J = 0;
	sum_J_val = 0;
	for j = 1:trials

		%-- Convery X and Y to one big matrix, with Y as the last column
		Xy = [X y];

		%-- Shuffle training set
		Xy_shuffle = shuffleMatrix(Xy);

		%-- Ravel up the matrix again
		total_cols = size(Xy, 2);
		X_sample = Xy_shuffle(:,1:total_cols - 1);
		y_sample = Xy_shuffle(:,total_cols);

		%-- Find theta parameters
		[theta] = trainLinearReg(X_sample(1:i,:), y_sample(1:i), lambda);

		lambda = 0;
		%-- Calculate cost and save them in the vector
		J = linearRegCostFunction(X_sample(1:i,:), y_sample(1:i), theta, lambda);
		sum_J = sum_J + J
		error_train(i) = J;


		Xy_val = [Xval, yval];
		Xy_val_shuffle = shuffleMatrix(Xy_val);
		total_cols = size(Xy_val_shuffle, 2);
		X_val_sample = Xy_val_shuffle(:,1:total_cols - 1);
		y_val_sample = Xy_val_shuffle(:,total_cols);

		J_val = linearRegCostFunction(X_val_sample(1:i,:), y_val_sample(1:i), theta, lambda);
		sum_J_val = sum_J_val + J_val;

	end
	error_train(i) = J / trials;
	error_val(i) = J_val / trials;
end





% -------------------------------------------------------------

% =========================================================================

end

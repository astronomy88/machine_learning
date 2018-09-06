function plotData(X, y)
%PLOTDATA Plots the data points X and y into a new figure 
%   PLOTDATA(x,y) plots the data points with + for the positive examples
%   and o for the negative examples. X is assumed to be a Mx2 matrix.

% Create New Figure
figure; hold on;

% ====================== YOUR CODE HERE ======================
% Instructions: Plot the positive and negative examples on a
%               2D plot, using the option 'k+' for the positive
%               examples and 'ko' for the negative examples.
%


%-- Need to separate the positive examples from the negative examples
%-- The following is my solution, and what follows is the more elegant solution
% m = length(y);

% m_pos = 0;
% m_neg = 0;
% X_pos = zeros(m, 2);
% X_neg = zeros(m, 2);

% for i = 1:m;
% 	if y(i) == 0
% 		m_neg = m_neg + 1;
% 		X_neg(m_neg,1) = X(i, 1);
% 		X_neg(m_neg, 2) = X(i, 2);
% 	else
% 		m_pos = m_pos + 1;
% 		X_pos(m_pos,1) = X(i, 1);
% 		X_pos(m_pos, 2) = X(i, 2);
% 	end
% end

% X_pos = X_pos(1:m_pos,:);
% X_neg = X_neg(1:m_neg,:);


% plot(X_pos(:,1), X_pos(:,2), 'linewidth', 10, 'k+')
% hold on;
% plot(X_neg(:,1), X_neg(:,2), 'markerfacecolor', 'y', 'ko')

%-- Andrew Ng's solution
% Find Indices of Positive and Negative Examples
pos = find(y==1);
neg = find(y==0);

plot(X(pos, 1), X(pos, 2), 'k+', 'LineWidth', 2, 'MarkerSize', 7);
plot(X(neg, 1), X(neg, 2), 'ko', 'MarkerFaceColor', 'y', 'MarkerSize', 7);

% =========================================================================



hold off;

end

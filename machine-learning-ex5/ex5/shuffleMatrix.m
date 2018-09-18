function [X_shuffle] = ...
    shuffleMatrix(X)
%SHUFFLEMATRIX Returns the matrix in randomized row order (keeps column intact)
%   [X_shuffle] = ...
%       SHUFFLEMATRIX(X) the matrix X with the rows randomized
%


% =========================================================================

m = size(X, 1);

%-- Everyday I'm shufflin', shufflin'
for i = 1:m-1
	first = i;
	second = i+1+floor((m-i)*rand(1));
	X = swapRows(X, first, second);
end

X_shuffle = X;

end
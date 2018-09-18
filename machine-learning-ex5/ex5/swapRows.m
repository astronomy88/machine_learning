function [X_row_swapped] = ...
    swapRows(X, i, j)
%SWAPROWS Returns the matrix with row i and j swapped.
%   [X_row_swapped] = ...
%       SWAPROWS(X) the matrix X with row i and j swapped
%


% =========================================================================

temp = X(i,:);
X(i,:) = X(j,:);
X(j,:) = temp;

X_row_swapped = X;
% =========================================================================

end
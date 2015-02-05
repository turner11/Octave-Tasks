function [theta] = normalEqn(X, y)
%NORMALEQN Computes the closed-form solution to linear regression 
%   NORMALEQN(X,y) computes the closed-form solution to linear 
%   regression using the normal equations.

theta = zeros(size(X, 2), 1);

% ====================== YOUR CODE HERE ======================
% Instructions: Complete the code to compute the closed form solution
%               to linear regression and put the result in theta.
%

% ---------------------- Sample Solution ----------------------

% a column of 1's was already added to the X matrix to have an intercept term (theta(0)).

XtX = X'*X;
XtXI = inv(XtX);
XtXIXt = XtXI *X';
theta = XtXIXt *y;


% -------------------------------------------------------------


% ============================================================

end

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

wX = X*theta;
costs = (wX -y).^2;
totalCost = sum(costs);
normalizedCosts = (1/(2*m)) * totalCost;

regulerizedCost = normalizedCosts + (lambda/(2*m)) *sum(theta(2:end).^2);

J = regulerizedCost;









% =========================================================================
%  Grad:
diff = wX - y;
diffX = (diff' *X)';

grad = (1/m) * diffX;
% Add regulization (for all BUT grad(0)):
grad(2:end) = grad(2:end) + (lambda/m)*theta(2:end);



end

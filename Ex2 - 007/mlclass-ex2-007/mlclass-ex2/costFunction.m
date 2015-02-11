function [J, grad] = costFunction(theta, X, y)
%COSTFUNCTION Compute cost and gradient for logistic regression
%   J = COSTFUNCTION(theta, X, y) computes the cost of using theta as the
%   parameter for logistic regression and the gradient of the cost
%   w.r.t. to the parameters.

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
%
% Note: grad should have the same dimensions as theta
%
pos = find(y==1);
neg = find(y == 0);

hx = X * theta;
sigmoidH = sigmoid(hx);

costFor_FalsePositive = -log(sigmoidH(pos) );%make sure that only predictions of POSITIVE (y == 1) is taken in considerations


costFor_FalseNegative = -log( 1- sigmoidH(neg) )  ;        %makes sure that only predictions of NEGITIVE (y == 0) is taken in considerations


%Note: I used indexes rather (y * ) or ([1-y]*) because inf * 0 results a NaN

summedCost = sum(costFor_FalsePositive) + sum(costFor_FalseNegative);

J= (1/m) * summedCost ;

%%-----------------Grad
diff = sigmoidH - y;
diffX = diff' *X;
grad = (1/m) * diffX;



% =============================================================

end

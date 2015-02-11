function [J, grad] = costFunctionReg(theta, X, y, lambda)
%COSTFUNCTIONREG Compute cost and gradient for logistic regression with regularization
%   J = COSTFUNCTIONREG(theta, X, y, lambda) computes the cost of using
%   theta as the parameter for regularized logistic regression and the
%   gradient of the cost w.r.t. to the parameters. 

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


pos = find(y==1);
neg = find(y == 0);

hx = X * theta;
sigmoidH = sigmoid(hx);

costFor_FalsePositive = -log(sigmoidH(pos) );%make sure that only predictions of POSITIVE (y == 1) is taken in considerations


costFor_FalseNegative = -log( 1- sigmoidH(neg) )  ;        %makes sure that only predictions of NEGITIVE (y == 0) is taken in considerations


%Note: I used indexes rather (y * ) or ([1-y]*) because inf * 0 results a NaN

summedCost = sum(costFor_FalsePositive) + sum(costFor_FalseNegative);

parameterPenalizing = (lambda/(2*m))* sum(theta(2:end).^2); %NOTE: theta(0) is not Penalized. It is the offset, typically 1...
J= (1/m) * summedCost +parameterPenalizing ;

%%-----------------Grad

diff = sigmoidH - y;
diffX = diff' *X;
grad = (1/m) * diffX;
% Add regulization (for all BUT grad(0)):
grad(2:end) = grad(2:end) + (lambda/m)*theta(2:end)';



% =============================================================

end

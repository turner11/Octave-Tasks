function g = sigmoid(z)
%SIGMOID Compute sigmoid functoon
%   J = SIGMOID(z) computes the sigmoid of z.

% ====================== FOR TESTING ======================
%clc
%clear
%z = -10000:1:10000;
% =========================================================



% You need to return the following variables correctly 
g = zeros(size(z));

% ====================== YOUR CODE HERE ======================
% Instructions: Compute the sigmoid of each value of z (z can be a matrix,
%               vector or scalar).


mz = -z;
e_exp = exp(mz);
g = 1 ./ (1 .+ e_exp );

%plot(z, g, 'g','LineWidth', 2, 'MarkerSize', 4);

% =============================================================

end

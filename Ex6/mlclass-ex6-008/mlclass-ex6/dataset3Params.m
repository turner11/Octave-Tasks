function [C, sigma] = dataset3Params(X, y, Xval, yval)
%EX6PARAMS returns your choice of C and sigma for Part 3 of the exercise
%where you select the optimal (C, sigma) learning parameters to use for SVM
%with RBF kernel
%   [C, sigma] = EX6PARAMS(X, y, Xval, yval) returns your choice of C and 
%   sigma. You should complete this function to return the optimal C and 
%   sigma based on a cross-validation set.
%

% You need to return the following variables correctly.
C = 1;
sigma = 0.3;

% ====================== YOUR CODE HERE ======================
% Instructions: Fill in this function to return the optimal C and sigma
%               learning parameters found using the cross validation set.
%               You can use svmPredict to predict the labels on the cross
%               validation set. For example, 
%                   predictions = svmPredict(model, Xval);
%               will return the predictions on the cross validation set.
%
%  Note: You can compute the prediction error using 
%        mean(double(predictions ~= yval))
%

% clc; help svmTrain
% clc; help svmPredict

% get all combinations of C % sigma^2
c_guesses = [0.01, 0.03, 0.1, 0.3, 1, 3, 10, 30];
sigma_squared_guesses = [0.01, 0.03, 0.1, 0.3, 1, 3, 10, 30];
[p,q] = meshgrid(c_guesses, sigma_squared_guesses);
pairs = [p(:) q(:)];

%find best combination
minError = inf;
m = size(pairs,1);
for i =1:m
  C_curr = pairs(i,1);
  sigma_curr =pairs(i,2);
 
  % Get model based on training set
  model = svmTrain(X, y, C_curr, @(x1, x2) gaussianKernel(x1, x2, sigma_curr)); 
  % Test the model on cross validation set
  predictions = svmPredict(model, Xval);  
  meanValidationError = mean(double(predictions ~= yval));
  % Is it the best so far?
  if meanValidationError < minError
    sigma = sigma_curr;
    C = C_curr;
    minError = meanValidationError;
  end 

end

% Used the code above for finding this...
%bestModel = pairs(20,:);
%C = bestModel(1);
%sigma = bestModel(2);








% =========================================================================

end

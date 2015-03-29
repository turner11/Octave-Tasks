function [mu sigma2] = estimateGaussian(X)
%ESTIMATEGAUSSIAN This function estimates the parameters of a 
%Gaussian distribution using the data in X
%   [mu sigma2] = estimateGaussian(X), 
%   The input X is the dataset with each n-dimensional data point in one row
%   The output is an n-dimensional vector mu, the mean of the data set
%   and the variances sigma^2, an n x 1 vector
% 

% Useful variables
[m, n] = size(X);

% You should return these values correctly
mu = zeros(n, 1);
sigma2 = zeros(n, 1);

% ====================== YOUR CODE HERE ======================
% Instructions: Compute the mean of the data and the variances
%               In particular, mu(i) should contain the mean of
%               the data for the i-th feature and sigma2(i)
%               should contain variance of the i-th feature.
%

mu  = mean(X);


featuresSums = sum(X);
sumsMinusAvg_2 = (featuresSums   - mu).^2;
sigma2 = 1/m * sumsMinusAvg_2 ;

sumF = zeros(n, 1)';

for i =1: m
  currX = X(i,:);
  sumF = sumF + currX ;
  
end

mu2 = 1/m * sumF ;

sumD = zeros(n, 1)';
for i = 1: m
  currX = X(i,:);
  currD = (currX - mu2).^2;
  sumD = sumD + currD; 
  
end

sigma22 = 1/m * sumD ;

aaa_sd  = max(max(sigma2 - sigma22));
aaaa = 3;



% =============================================================


end

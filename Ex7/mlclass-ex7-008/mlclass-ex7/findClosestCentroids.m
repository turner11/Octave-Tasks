function idx = findClosestCentroids(X, centroids)
%FINDCLOSESTCENTROIDS computes the centroid memberships for every example
%   idx = FINDCLOSESTCENTROIDS (X, centroids) returns the closest centroids
%   in idx for a dataset X where each row is a single example. idx = m x 1 
%   vector of centroid assignments (i.e. each entry in range [1..K])
%

% Set K
K = size(centroids, 1);

m =size(X,1);
% You need to return the following variables correctly.
idx = zeros(m, 1);

% ====================== YOUR CODE HERE ======================
% Instructions: Go over every example, find its closest centroid, and store
%               the index inside idx at the appropriate location.
%               Concretely, idx(i) should contain the index of the centroid
%               closest to example i. Hence, it should be a value in the 
%               range 1..K
%
% Note: You can use a for-loop over the examples to compute this.
%
distances = 1./zeros(m, 1)';%initizalize to infitity...

isStable = 0;
  while isStable == 0  
    % Cluster assignment step: Assign each data point to the
    % closest centroid. idx(i) corresponds to cˆ(i), the index
    % of the centroid assigned to example i
    for i = 1: m
      isStable  = 1; %until proven otherwise...
      currSample = X(i,:);      
      minIdx =idx(i);
     
      for j = 1:K    
        minDiff = distances(i);    
        currCentroid = centroids(j,:);        
        currDiff = norm( currSample - currCentroid , 2 );   
        if  currDiff < minDiff 
        % we found a new centroid to allocate
          distances(i) = currDiff;
          idx(i) = j;
          isStable = 0; %something has changed - we are not done... 
        end
      end
      
      
      
    end
   
    
   
  end





% =============================================================

end


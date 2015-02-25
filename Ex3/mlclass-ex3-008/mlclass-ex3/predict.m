function p = predict(Theta1, Theta2, X)
%PREDICT Predict the label of an input given a trained neural network
%   p = PREDICT(Theta1, Theta2, X) outputs the predicted label of X given the
%   trained weights of a neural network (Theta1, Theta2)

% Useful values
m = size(X, 1);
num_labels = size(Theta2, 1);

% You need to return the following variables correctly 
p = zeros(size(X, 1), 1);

% ====================== YOUR CODE HERE ======================
% Instructions: Complete the following code to make predictions using
%               your learned neural network. You should set p to a 
%               vector containing labels between 1 to num_labels.
%
% Hint: The max function might come in useful. In particular, the max
%       function can also return the index of the max element, for more
%       information see 'help max'. If your examples are in rows, then, you
%       can use max(A, [], 2) to obtain the max for each row.
%

input1 =[ones(m, 1) X];
%Get the out put of first layer (input of second layer)
output1 = sigmoid(input1*Theta1');  

input2 = [ones(m, 1) output1];
%Get the out put of second layer (out put of the entire neural network)
output2 = sigmoid(input2*Theta2');  
%calculate the highest propability

 [ max_value, max_index ]  = max(output2, [], 2); 

 p =max_index;
 
 %plot(p);
 %stem(p);





% =========================================================================


end

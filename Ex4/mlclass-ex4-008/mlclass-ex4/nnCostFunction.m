function [J grad] = nnCostFunction(nn_params, ...
                                   input_layer_size, ...
                                   hidden_layer_size, ...
                                   num_labels, ...
                                   X, y, lambda)
%NNCOSTFUNCTION Implements the neural network cost function for a two layer
%neural network which performs classification
%   [J grad] = NNCOSTFUNCTON(nn_params, hidden_layer_size, num_labels, ...
%   X, y, lambda) computes the cost and gradient of the neural network. The
%   parameters for the neural network are "unrolled" into the vector
%   nn_params and need to be converted back into the weight matrices. 
% 
%   The returned parameter grad should be a "unrolled" vector of the
%   partial derivatives of the neural network.
%

% Reshape nn_params back into the parameters Theta1 and Theta2, the weight matrices
% for our 2 layer neural network
Theta1 = reshape(nn_params(1:hidden_layer_size * (input_layer_size + 1)), ...
                 hidden_layer_size, (input_layer_size + 1));

Theta2 = reshape(nn_params((1 + (hidden_layer_size * (input_layer_size + 1))):end), ...
                 num_labels, (hidden_layer_size + 1));

% Setup some useful variables
m = size(X, 1);
         
% You need to return the following variables correctly 
J = 0;
Theta1_grad = zeros(size(Theta1));
Theta2_grad = zeros(size(Theta2));

% ====================== YOUR CODE HERE ======================
% Instructions: You should complete the code by working through the
%               following parts.
%
% Part 1: Feedforward the neural network and return the cost in the
%         variable J. After implementing Part 1, you can verify that your
%         cost function computation is correct by verifying the cost
%         computed in ex4.m
%
% Part 2: Implement the backpropagation algorithm to compute the gradients
%         Theta1_grad and Theta2_grad. You should return the partial derivatives of
%         the cost function with respect to Theta1 and Theta2 in Theta1_grad and
%         Theta2_grad, respectively. After implementing Part 2, you can check
%         that your implementation is correct by running checkNNGradients
%
%         Note: The vector y passed into the function is a vector of labels
%               containing values from 1..K. You need to map this vector into a 
%               binary vector of 1's and 0's to be used with the neural network
%               cost function.
%
%         Hint: We recommend implementing backpropagation using a for-loop
%               over the training examples if you are implementing it for the 
%               first time.
%
% Part 3: Implement regularization with the cost function and gradients.
%
%         Hint: You can implement this around the code for
%               backpropagation. That is, you can compute the gradients for
%               the regularization separately and then add them to Theta1_grad
%               and Theta2_grad from Part 2.
%

%% Calculate cost --------------------------------------------------------------
  
  %featureCount = size(X,2);   
  
   %% Varaiables that will be used in all itterations. ===========
  labels = unique(y);
  labelsCount = size(labels,1);
   %%==================================================
  I = eye(labelsCount);
    
    
  cost_total = 0;
  delta1 = 0;
  delta2 = 0;
  for i = 1:m 
    currSample = X(i,:);% The sample for training the network in this pass
    currLabel = y(i);%the label of current sample
   
    input_layer2 =[1  currSample];% Add the bias
    %Get the out put of first layer (input of second layer)
    z2 = input_layer2*Theta1';
    output_layer2 = sigmoid(z2);  %a2
   
    
    input_layer3 = [1 output_layer2];% Add the bias
    %Get the out put of second layer (out put of the entire neural network)
    z3 = input_layer3*Theta2';
    output_layer3 = sigmoid(z3); %a3

    %calculate the highest propability

    [ max_value, max_index ]  = max(output_layer3, [], 2); 
    prediction =max_index;
    

    %% Vectorize results for the cost function===========
    vectorizedP = double(I(:, prediction));
    vectorizedY = double(I(:, currLabel));   
    
    
    %% The actual cost calculation  ---------------
    %Note: I used indexes rather (y * ) or ([1-y]*) because inf * 0 results a NaN
    pos = find(vectorizedY==1)';%indexes of labels that were marked as current network output
    neg = find(vectorizedY == 0)';%indexes of labels that were NOT marked as current network output
    
    hx = output_layer3;
    %sigmoidH = sigmoid(hx);
    hxPos = hx(pos);
    costFor_FalsePositive = -log(hxPos );%make sure that only predictions of POSITIVE (y == 1) is taken in considerations

    hxNeg = hx(neg);
    hxNegDiff = 1- hxNeg;
    costFor_FalseNegative = -log(hxNegDiff)  ; %makes sure that only predictions of NEGITIVE (y == 0) is taken in considerations
    
    cost_curr = sum(costFor_FalsePositive) + sum(costFor_FalseNegative);

    %parameterPenalizing = (lambda/(2*m))* sum(theta(2:end).^2); %NOTE: theta(0) is not Penalized. It is the offset, typically 1...
    %cost_curr=  summedCost +parameterPenalizing ;
    cost_total  = cost_total  + cost_curr;
    
    %clear pos; clear neg; clear hx; clear hxPos; clear costFor_FalsePositive;
    %clear hxNeg; clear hxNeg; clear costFor_FalseNegative; clear cost_curr; 
    %%===-----------------Grad - backpropagation
    a2 = output_layer2;%input_layer2; %a2 as per definition in PPT
    a3 = output_layer3;%input_layer3; %a3 as per definition in PPT
    
    %Errors in output layer
    error_diff_output_layer = a3' - vectorizedY;
    
    %errors in hidden layer    
    error_diff_Layer2 = (Theta2' * error_diff_output_layer) .* sigmoidGradient([1 z2])';
    error_diff_Layer2 = error_diff_Layer2(2:end);%Remove unit resulted from theta's bias   
    
    %Get delta matrices    
    d_mult_a_L1 = error_diff_Layer2 * input_layer2;         %26X1 * 1X25 Expected 25X401
    d_mult_a_L2= error_diff_output_layer *input_layer3;    %10X1 * 1X10 Expected 10X26
    
    delta1 = delta1 + d_mult_a_L1;
    delta2 = delta2 + d_mult_a_L2;
 end

 %% Calculatae the regularization terms---------------------
 
 Theta1_ExcludingBias =Theta1(:,2:end);
 Theta2_ExcludingBias =Theta2(:,2:end);
 Theta1_reg = sum(sum(Theta1_ExcludingBias.^2));
 Theta2_reg = sum(sum(Theta2_ExcludingBias.^2));
 totalReg = Theta1_reg+Theta2_reg;
 
 
 
 
 J= (1/m) *cost_total + (lambda/(2*m))*totalReg;

% -------------------------------------------------------------

% =========================================================================

% Unroll gradients

 D1 = (1/m)*delta1 ;
 D2 = (1/m)*delta2 ;
 %regularization for non biased terms
D1(:,2:end) = D1(:,2:end) + (lambda/m)* Theta1(:,2:end); 
D2(:,2:end) = D2(:,2:end) + (lambda/m)* Theta2(:,2:end);

Theta1_grad = D1;
Theta2_grad = D2;

grad = [Theta1_grad(:) ; Theta2_grad(:)];


end

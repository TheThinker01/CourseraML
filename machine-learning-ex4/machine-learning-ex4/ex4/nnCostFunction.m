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

%% Add the feature of ones to X
X=[ones(m,1) X];


%% Then we need to find h using feedforward algorithm
z_2 = Theta1*X';
a_2 = sigmoid(z_2);% here a_2 is hidden_layer x m
a_2 = [ones(1,m);a_2];
z_3 = Theta2*a_2;
a_3 = sigmoid(z_3);
lh = log(a_3); % here lh is num_labels x m and is the lag of all the values 

%% create a modified y matrix where each column 
%% contains 1 for the row that is true , zero for all others
y_mod = zeros(num_labels,m);
y=y';
for i=1:m
  y_mod(y(i),i)=1;
endfor

%% Compute the cost function
%% first sum is for all the  examples
%% Second sum if for all the labels
%% the slicing is done so as to omit the first columns of the matrix 
J = -(sum(sum(y_mod.*lh+(1-y_mod).*log(1-a_3))))/m + ...
      lambda*(sum(sum((Theta1.^2)(:,2:end)))+sum(sum((Theta2.^2)(:,2:end))))/(2*m);
      

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
y_mod = y_mod';
delta_2 = zeros(size(Theta2)); %% delta_L has the same size as ThetaL 
  %% where L is the layer number 
delta_1 = zeros(size(Theta1));  
for t=1:m
  x = X(t,:); %% This produces a row vector 
  y = y_mod(t,:); %% this also produces a row vector
  a_1 = x';
  z_2 = Theta1*x';
  a_2 = sigmoid(z_2);
  a_2 = [1;a_2];
  z_3 = Theta2*a_2;
  a_3 = sigmoid(z_3); %% This is the output
  d_3 = a_3 - y';
  z_2 = [1;z_2];
  d_2 = (Theta2' * d_3).* sigmoidGradient(z_2);
  delta_2 = delta_2 + d_3*a_2';
  d_2 = d_2(2:end);
  delta_1 = delta_1 +d_2*a_1';

endfor
Theta1_grad = delta_1/m;
Theta2_grad = delta_2/m;


%
% Part 3: Implement regularization with the cost function and gradients.
%
%         Hint: You can implement this around the code for
%               backpropagation. That is, you can compute the gradients for
%               the regularization separately and then add them to Theta1_grad
%               and Theta2_grad from Part 2.
%

t1 = Theta1_grad(:,1);%% Save the first column 
t2 = Theta2_grad(:,1); %%Save the first column

Theta1_grad = Theta1_grad + (lambda/m)*Theta1;
Theta2_grad = Theta2_grad + (lambda/m)*Theta2;

Theta1_grad(:,1) = t1;
Theta2_grad(:,1) = t2;


% -------------------------------------------------------------

% =========================================================================

% Unroll gradients
grad = [Theta1_grad(:) ; Theta2_grad(:)];


end

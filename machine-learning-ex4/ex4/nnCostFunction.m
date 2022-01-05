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
Theta1_grad = zeros(size(Theta1)); %25*401=size of theta1
Theta2_grad = zeros(size(Theta2));%10*26=size of theta2

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
a1=X; 

a1 = [ones(m, 1) a1]; % 5000*401

z2=Theta1*a1';% 25*401 and 5000*401.

a2=sigmoid(z2);

% a2 would be 25*5000


a2 = [ones(1, m); a2]; %25*5000
%now it becomes 26*5000
z3=Theta2*a2; %10*26 and 26*5000

H=sigmoid(z3); %10*5000

%a3 is hypothesis
% y is 1*5000

y_matrix=eye(num_labels)(y,:); %5000*10


J = sum(sum(-y_matrix'.*log(H)-(1-y_matrix').*log(1-H)));

J=J/m;
% just make sure to remove ones layer when doing regularization


Theta2(:,[1]) = [];%10*25	%this  removes bias layer for both theta2 and theta1, regardless of their size.
Theta1(:,[1]) = [];%25*400



R= (lambda/(2*m)) * (sum(sum(Theta1.^2))+ sum(sum(Theta2.^2)));

J=J+R;

%Backpropagation:

%D1=zeros(25,5000);
%D2=zeros(10,26);
 %for t = 1:m
%d3=H(:,t)-y_matrix'(:,t);%10*1
%d2=Theta2'*d3.*sigmoidGradient(z2(:,t)); %25*1
%D1=D1+d2*a1(:,t)'; %(25*1)*(1*401)
%D2=D2+d3*a2(:,t)';%(10*1) * (1*25) 10*25
 %endfor
%D3=D3./m;
%D2=D2./m;
%D1=D1./m;

d3=H-y_matrix';
d2=(d3'*Theta2)'.*sigmoidGradient(z2); 
D2=d2*a1;
D3=d3*a2';% 10*5000 5000*26

Theta1_grad=D2/m;%25*401
Theta2_grad=D3/m;%10*26
% -------------------------------------------------------------


%adding regularization to gradient:

add1=(lambda/m)*Theta1; %25*400
add2=(lambda/m)*Theta2; %10*25
add1=[zeros(size(add1,1),1) add1];%25*401
add2=[zeros(size(add2,1),1) add2];%10*26

Theta1_grad=add1+Theta1_grad;
Theta2_grad=add2+Theta2_grad;
% =========================================================================

% Unroll gradients
grad = [Theta1_grad(:) ; Theta2_grad(:)];


end

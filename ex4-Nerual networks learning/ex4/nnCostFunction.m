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
%input_layer_size=400;
%hidden_layer_size=25;
%num_labels=10;
%lambda=0;
%X=zeros(5000,400);
%y=zeros(5000,1);
%nn_params=zeros(10285,1);
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
%               containing svalues from 1..K. You need to map this vector into a 
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
%--------前向传播 h=a2----------%
bias1=ones(m,1);
a_1=X;
X_1=[bias1 a_1];%加入偏置单元后的输入
z_2=X_1*Theta1';
a_2=sigmoid(z_2);%激活单元
m2=size(a_2,1);
bias2=ones(m2,1);
X_2=[bias2 a_2];%加入偏置单元后的隐含层输入
z_3=X_2*Theta2';
a_3=sigmoid(z_3);%计算theta1和theta2的假设函数
h=a_3;%输出层激活单元的值等于假设函数

%--------------计算代价函数-----------------%
Y=eye(size(Theta2,1));%构造10维单位矩阵作为10个标签的10个判断向量
for i=1:m
    value=y(i);
    Ylabel=Y(:,value);
    J=J-1/m*(log(h(i,:))*Ylabel+log(1-h(i,:))*(1-Ylabel));
end

%----------正则化代价函数自己的方法----------%
J=J+lambda/(2*m)*(sum(Theta1(:,2:end).^2,'all')+sum(Theta2(:,2:end).^2,'all'));

%-------------反向传播算法--------------%
for i=1:m
    y3(i,:)=Y(:,y(i))';%改变数字标签为向量
end
delta_3=a_3-y3;%计算输出层误差
Theta2_grad(:)=1/m.*delta_3'*X_2;
delta_2=delta_3*Theta2(:,2:end).*sigmoidGradient(z_2);%计算隐含层误差
Theta1_grad(:)=1/m.*delta_2'*X_1;

%-------------正则化反向传播算法--------------%
Theta2_grad(:,1)=1/m.*delta_3'*X_2(:,1);
Theta2_grad(:,2:end)=1/m.*delta_3'*X_2(:,2:end)+lambda/m.*Theta2(:,2:end);
Theta1_grad(:,1)=1/m.*delta_2'*X_1(:,1);
Theta1_grad(:,2:end)=1/m.*delta_2'*X_1(:,2:end)+lambda/m.*Theta1(:,2:end);

% Unroll gradients
grad = [Theta1_grad(:) ; Theta2_grad(:)];


end

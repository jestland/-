
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
%
h=1./(1+exp(-X*theta));%假设函数
J=-1/m*(y'*log(h)+(1-y)'*log(1-h))+lambda/(2*m)*(theta(2:end)'*theta(2:end));%正则化代价函数
x=X';
for i=1:size(theta)
    if i==1
        grad(i)=1/m*(x(i,:)*(h-y));
    else
        grad(i)=1/m*(x(i,:)*(h-y))+lambda/m*theta(i);
    end
end





% =============================================================

end

function p = predictOneVsAll(all_theta, X)
%PREDICT Predict the label for a trained one-vs-all classifier. The labels 
%are in the range 1..K, where K = size(all_theta, 1). 
%  p = PREDICTONEVSALL(all_theta, X) will return a vector of predictions
%  for each example in the matrix X. Note that X contains the examples in
%  rows. all_theta is a matrix where the i-th row is a trained logistic
%  regression theta vector for the i-th class. You should set p to a vector
%  of values from 1..K (e.g., p = [1; 3; 1; 2] predicts classes 1, 3, 1, 2
%  for 4 examples) 

m = size(X, 1);
num_labels = size(all_theta, 1);

% You need to return the following variables correctly 
p = zeros(size(X, 1), 1);

% Add ones to the X data matrix
X = [ones(m, 1) X];

% ====================== YOUR CODE HERE ======================
% Instructions: Complete the following code to make predictions using
%               your learned logistic regression parameters (one-vs-all).
%               You should set p to a vector of predictions (from 1 to
%               num_labels).
%
% Hint: This code can be done all vectorized using the max function.
%       In particular, the max function can also return the index of the 
%       max element, for more information see 'help max'. If your examples 
%       are in rows, then, you can use max(A, [], 2) to obtain the max 
%       for each row.
%       

% hypothesis hTheta(X) = g(theta'*X) = sigmoid (theta'*X),
% where X is (m x n+1), theta is (n+1 x 1) and y is (m x 1), htheta is (m x 1) 
% here all_theta is a matrix where every row is the theta vector

% keeping in mind dimensions of X(m x n+1) and all_theta(num_labels x n+1)
hThetaX = sigmoid(X * all_theta');
% result hThetaX is (m x num_labels)

% note - 
% in this exercise, y is (m x 1) i.e. there are 5000 examples and each
% example can be one of num_labels = 10 classes/labels (i.e. like '1','2',..'9','0')
% where label '0' stands for digit 10.
% and hThetaX is (m x num_labels). The index of the max value in each row of
% hThetaX gives the prediction - i.e. if in 3rd row of HThetaX array, the 
% 7th element is the maximum value, it means the 3rd example of X, approximates
% to the label '7' (i.e. digit 7).

maxHThetaXPerRow = zeros(size(X, 1), 1); 
indexMaxHThetaXPerRow = zeros(size(X, 1), 1);

[maxHThetaXPerRow, indexMaxHThetaXPerRow] = max(hThetaX, [], 2);

p = indexMaxHThetaXPerRow;

% =========================================================================
end

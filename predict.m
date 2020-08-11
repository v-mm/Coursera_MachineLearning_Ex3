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

% Add ones to the X data matrix
% a1 = X with bias 1s column i.e. 5000x401
% Theta1 is 25x401
% Theta2 is 10x26

a1 = [ones(m, 1) X];
%a1 is (5000x401)

% z2 = Theta1 x a1
% a2 = g(z2)
z2 = a1 * Theta1';
% z2 is (5000x25)

a2 = sigmoid (z2);
% a2 is (5000x25)

a2 = [ones(size(a2,1),1) a2];
% a2 is (5000x26)

% z3 = Theta2 x a2
% a3 = g(z3)
z3 = a2 * Theta2';
% z3 is (5000x26)*(26x10) = (5000x10)

a3 = sigmoid (z3);
% a3 is (5000x10)

maxHThetaXPerRow = zeros(size(a3, 1), 1); 
indexMaxHThetaXPerRow = zeros(size(a3, 1), 1);

[maxHThetaXPerRow, indexMaxHThetaXPerRow] = max(a3, [], 2);

p = indexMaxHThetaXPerRow;



% =========================================================================


end

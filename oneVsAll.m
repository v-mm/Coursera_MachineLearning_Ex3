function [all_theta] = oneVsAll(X, y, num_labels, lambda)
%ONEVSALL trains multiple logistic regression classifiers and returns all
%the classifiers in a matrix all_theta, where the i-th row of all_theta 
%corresponds to the classifier for label i
%   [all_theta] = ONEVSALL(X, y, num_labels, lambda) trains num_labels
%   logistic regression classifiers and returns each of these classifiers
%   in a matrix all_theta, where the i-th row of all_theta corresponds 
%   to the classifier for label i

% Some useful variables
m = size(X, 1);
n = size(X, 2);

% You need to return the following variables correctly 
all_theta = zeros(num_labels, n + 1);

% Add ones to the X data matrix
X = [ones(m, 1) X];

% NOTE:
% Calculations in linear regression / logistic regression for linear hypothesis
% use vector computations - where the training examples, parameters (small case 
% theta)and outputs are vector columns (or handled as vectors - x(i) being 
% a row vector in X).
% However for non-linear hypothesis using neural networks, the calculations involve
% matrix computaions - where the input layer, hidden layers, output layers and
% parameters (capital Theta) and gradients are all matrices. These matrices are
% either unrolled as vectors or reshaped back from vectors into matrices as the
% case requires.


% ====================== YOUR CODE HERE ======================
% Instructions: You should complete the following code to train num_labels
%               logistic regression classifiers with regularization
%               parameter lambda. 
%
% Hint: theta(:) will return a column vector.
%
% Hint: You can use y == c to obtain a vector of 1's and 0's that tell you
%       whether the ground truth is true/false for this class.
%
% Note: For this assignment, we recommend using fmincg to optimize the cost
%       function. It is okay to use a for-loop (for c = 1:num_labels) to
%       loop over the different classes.
%
%       fmincg works similarly to fminunc, but is more efficient when we
%       are dealing with large number of parameters.
%
% Example Code for fmincg:
%
%     % Set Initial theta
%     initial_theta = zeros(n + 1, 1);
%     
%     % Set options for fminunc
%     options = optimset('GradObj', 'on', 'MaxIter', 50);
% 
%     % Run fmincg to obtain the optimal theta
%     % This function will return theta and the cost 
%     [theta] = ...
%         fmincg (@(t)(lrCostFunction(t, X, (y == c), lambda)), ...
%                 initial_theta, options);
%

% ============================================================================
% basically we repeat the binary clasification for every new class label...
% making a pseudo 'y' filled with 0s or 1s where the class we are seeking is 1
% vs 0 for all the rest

% all_theta is a matrix where the i-th row is a trained logistic
% regression theta vector for the i-th class. 


% Initialize fitting parameters for a class of labels... 
% (i.e. for each row of X and y pair)
initial_theta = zeros(n + 1, 1);

% Set options for fmincg
options = optimset('GradObj', 'on', 'MaxIter', 50);

for c = 1:num_labels

%  Run fmincg to obtain the optimal theta
%  This function will return theta and the cost 
% assign the computed theta as the 'c'th row in matrix all_theta  
all_theta(c,:) = ...
fmincg(@(t)(lrCostFunction(t, X,(y==c), lambda)), initial_theta, options);
% see ex3.pdf

% in binary clasification, y is always 0 or 1
% here we are multi classifying digits, so corresponding position in y is made 1
% for every row in X, rest are 0. 

% y==c => a row vector of the form [0 0 0..1...0 0] where 1 appears at the 
% position where y - label corresponds to c label ('1', '2', '3'...'0')
% so a clasification between '1', '2','3', etc is transformed into a series of 
% clasifications between '1' (or '2', or '3', etc) vs 0 (the rest - vs all)
% for each row in X

% with every iteration of 'c', the corresponding row in all_theta is updated
% with the computed theta values with minimum cost for that classification.
end;

% =========================================================================

end

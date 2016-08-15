%Logistic Regression
%% Initialization
clear ; close all; clc
%% Load Data
%  The first two columns contains the exam scores and the third column
%  contains the label.
data = load('ex2data1.txt');
X = data(:, [1, 2]); y = data(:, 3);
%Plot the data 
fprintf(['Plotting data with + indicating (y = 1) examples and o ' ...
         'indicating (y = 0) examples.\n']);

% Create New Figure
figure; hold on;
%Find the indices where y is 1 and 0
pos = find(y==1);
neg = find(y==0);
plot(X(pos,1),X(pos,2),'k+','LineWidth', 2, 'MarkerSize', 7);
plot(X(neg,1),X(neg,2),'ko', 'MarkerFaceColor', 'y', 'MarkerSize', 7);
% Put some labels 
hold on;
% Labels and Legend
xlabel('Exam 1 score')
ylabel('Exam 2 score')
% Specified in plot order
legend('Admitted', 'Not admitted')
hold off;
%  Compute the cost and the gradient
%  Setup the data matrix appropriately, and add ones for the intercept term
[m, n] = size(X);
% Add intercept term to x and X_test
X = [ones(m, 1) X];
% Initialize fitting parameters
initial_theta = zeros(n + 1, 1);
% Compute and display initial cost and gradient
[cost, grad] = costFunc(initial_theta, X, y);
fprintf('Cost at initial theta (zeros): %f\n', cost);
fprintf('Gradient at initial theta (zeros): \n');
fprintf(' %f \n', grad);
%Use the built-in function (fminunc) to find the optimal parameters theta.
%  Set options for fminunc
options = optimset('GradObj', 'on', 'MaxIter', 400);
%  Run fminunc to obtain the optimal theta
%  This function will return theta and the cost 
[theta, cost] = ...
	fminunc(@(t)(costFunc(t, X, y)), initial_theta, options);
% Print theta to screen
fprintf('Cost at theta found by fminunc: %f\n', cost);
fprintf('theta: \n');
fprintf(' %f \n', theta);
%  Predict probability for a student with score 45 on exam 1 
%  and score 85 on exam 2 
prob = sigmoid([1 45 85] * theta);
fprintf(['For a student with scores 45 and 85, we predict an admission ' ...
         'probability of %f\n\n'], prob);
% Compute accuracy on our training set
p = round(sigmoid(X*theta));
fprintf('Train Accuracy: %f\n', mean(double(p == y)) * 100);


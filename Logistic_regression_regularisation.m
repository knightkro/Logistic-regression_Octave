%Logistic Regression with regularisation

%% Initialization
clear ; close all; clc
%% Load Data
data = load('ex2data2.txt');
X = data(:, [1, 2]); y = data(:, 3);
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
xlabel('Microchip Test 1')
ylabel('Microchip Test 2')

% Specified in plot order
legend('y = 1', 'y = 0')
hold off;
% Add Polynomial Features.  We use the mapFeature function from Coursera's course on machine learning 
% INset your own features if you want. Note that mapFeature also adds a column of ones for us, so the intercept
% term is handled
X = mapFeature(X(:,1), X(:,2));
% Initialize fitting parameters
initial_theta = zeros(size(X, 2), 1);
% Set regularization parameter lambda to 1
lambda = 1;
% Compute and display initial cost and gradient for regularized logistic
% regression
[cost, grad] = costFuncReg(initial_theta, X, y, lambda);
fprintf('Cost at initial theta (zeros): %f\n', cost);

%test out the programme
% Initialize fitting parameters
initial_theta = zeros(size(X, 2), 1);
% Set regularization parameter lambda to 1 , which can be varied
lambda = 1;
% Set Options
options = optimset('GradObj', 'on', 'MaxIter', 400);
% Optimize
[theta, J, exit_flag] = ...
	fminunc(@(t)(costFuncReg(t, X, y, lambda)), initial_theta, options);
% Compute accuracy on our training set
p = round(sigmoid(X*theta));
fprintf('Train Accuracy: %f\n', mean(double(p == y)) * 100);
[cost, grad] = costFuncReg(theta, X, y, lambda);
fprintf('Cost at final theta: %f\n', cost);



function [C,Ez,Ex] = FCMSC(X,opts)
%FCMSC - Feature Concatenation Multi-view Subspace Clustering
% 
% Syntax: [NMI,ACC,F,AVG,P,RI] = FCMSC(X,gt,numC,opts)
% Objective function: 
%         ||Ex||_2,1 + lambda_1*||Ez||_2,1 + lambda_2*||C||_* + lambda_3*sum(tr(C'*L_i*C))
%         s.t.  X = X*Z + Ex, Z = Z*C + Ez
%         introducing auxiliary variables J to replace C in the nuclear term of our objective function
%
% Input:  X    - cell(1,v), data contain v views, in each view, each column stands for each data point;
%         opts - contain hyperparameters
% 
% Output: C    - Coefficient matrix of the proposed method
%         Ez   - Ecs in paper is cluster-specific error, and Ez here is utilized to process Ecs
%         Ex   - Sample specific error
% 
% Copyright: zhengqinghai@stu.xjtu.edu.cn
% 2019/05/21

%% initialize variables
v = size(X,2);
n = size(X{1},2);

for i = 1:v
    L{i} = LaplacianConstruction(X{i});
    X{i} = X{i}./(repmat(sqrt(sum(X{i}.^2,1)),size(X{i},1),1)+1e-8);   
end
sumL = zeros(n,n);
for i = 1:v
    sumL = sumL+L{i}+L{i}';
end

lambda_1 = opts.lambda_1;
lambda_2 = opts.lambda_2;
lambda_3 = opts.lambda_3;

rho = opts.rho;
max_mu = 1e6;
mu = 1e-4;

%% some initilization of X_a,Ex,Z,Ez,C,J,Q,Y_1,Y_2,Y_3,Y_4
X_a = [];
d_a = 0;
for i = 1:v
    X_a = [X_a;X{i}];
    d_a = d_a + size(X{i},1);
end

Ex = zeros(d_a,n);
Z = zeros(n,n);
Ez = zeros(n,n);
C = zeros(n,n);
J = zeros(n,n);
Y_1 = zeros(d_a,n);
Y_2 = zeros(n,n);
Y_3 = zeros(n,n);

Z = rand(n,n);

%% optimization
iter_max = 100;
iter_curr = 0;
conv_flag = 0;
conv_threshold = 1e-6;
while conv_flag==0 && iter_curr<iter_max
    
    % update Ex
    % min ||Ex||_2,1 + mu/2*||Ex-A_Ex||_F_2, A_Ex = X_a-X_a*Z+Y_1/mu
    A_Ex = X_a-X_a*Z+Y_1/mu;
    Ex = solve_l1l2(A_Ex,1/mu);

    % update Ez
    % min lambda_1*||Ez||_2,1 + mu/2*||Ez - A_Ez||_F_2, A_Ez = Z-Z*C+Y_2/mu
    % the optimization of Ez is similar to the optimization of Ex
    A_Ez = Z-Z*C+Y_2/mu;
    Ez = solve_l1l2(A_Ez,lambda_1/mu);

    % update J
    % min lamda_2*||J||_* + mu/2*||J-A_J||_F_2. A_J = C+Y_3/mu;
    % A_J = C+Y_3/mu+eye(n)*1e-8;
    A_J = C+Y_3/mu;
    J = softth(A_J,lambda_2/mu);

    % update C
    % min lambda_3*sum(tr(C'*L_i*C)) + <Y_2,Z-Z*C-Ez> + mu/2*||Z-Z*C-Ez||_F_2 + <Y_3,C-J> + mu/2*||C-J||_F_2
    % approach: derivative with respect to C
    % A_C*C = B_C, we can get C = A_C\B_C;
    A_C = lambda_3*sumL+eye(n)*mu+mu*(Z'*Z);
    B_C = J*mu-Y_3+Z'*Y_2+mu*(Z'*Z)-mu*Z'*Ez;
    C = A_C\B_C;

    % update Z
    % min <Y_1,X-X*Z-Ex> + mu/2*||X-X*Z-Ex||_F_2 + <Y_2,Z-Z*C-Ez> + mu*||Z-Z*C-Ez||_F_2
    % approach: derivative with respect to Z
    % A_Z*Z + Z*B_Z = C_Z, we can get Z = sylvester(A_Z,B_Z,C_Z);
    C_Z = 1/mu*X_a'*Y_1+X_a'*X_a-X_a'*Ex-1/mu*Y_2+1/mu*Y_2*C'+Ez-Ez*C';
    A_Z = X_a'*X_a+eye(n);
    B_Z = C*C'-C'-C;
    Z = sylvester(A_Z,B_Z,C_Z);

    % update Y_1, Y_2, Y_3, Y_4 and mu
    Y_1 = Y_1+mu*(X_a-X_a*Z-Ex);
    Y_2 = Y_2+mu*(Z-Z*C-Ez);
    Y_3 = Y_3+mu*(C-J);

    mu = min(rho*mu,max_mu);

    % check the convergence conditions
    condition_1 = norm(X_a-X_a*Z-Ex,inf)<conv_threshold;
    condition_2 = norm(Z-Z*C-Ez,inf)<conv_threshold;
    condition_3 = norm(C-J,inf)<conv_threshold;
    if condition_1 && condition_2 && condition_3
        conv_flag = 1;
        fprintf('\tConvergence condition achieved\n');
    end

    iter_curr = iter_curr + 1;
end
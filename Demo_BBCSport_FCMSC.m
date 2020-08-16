clear;clc;
addpath('./Tools');
load('./data/bbcsport_2view.mat');
fprintf('FCMSC on BBCSport dataset\n');
X{1} = data{1}';
X{2} = data{2}';
gt = truth;
num_views = size(X,2);
num_C = size(unique(gt),1);

opts.lambda_1 = 100;
opts.lambda_2 = 100;

opts.lambda_3 = 0; % zero for FCMSC and non-zero for grFCMSC!
% opts.lambda_3 = 0.001; % 0.001 for lambda_3 used in experiments on BBCSport dataset

opts.rho = 1.9;

for i = 1:30
    fprintf('%d-th iteration\n', i);
    [C,~,~] = FCMSC(X,opts);
    [NMI_c(i),ACC_c(i),F_c(i),RI_c(i)]=clustering(abs(C)+abs(C'), num_C, gt);
    fprintf('\t--- Clusterint results of C: NMI = %f, ACC = %f, F = %f, RI = %f\n',NMI_c(i),ACC_c(i),F_c(i),RI_c(i));
end
fprintf('Mean Clustering results of C: NMI = %f(%f), ACC = %f(%f), F = %f(%f), RI = %f(%f)\n',mean(NMI_c),std(NMI_c),mean(ACC_c),std(ACC_c),mean(F_c),std(F_c),mean(RI_c),std(RI_c));
% Reference:
% Di Wang, Xinbo Gao, Xiumei Wang, and Lihuo He. 
% Label Consistent Matrix Factorization Hashing. 
% IEEE Transactions on Pattern Analysis and Machine Intelligence, 41(10):2466 - 2479, 2019.
% (Manuscript)
%
% Contant: Di Wang (wangdi@xidain.edu.cn)
%
clc;clear 
load mirflickr25k.mat
%% Calculate the groundtruth
GT = L_te*L_tr';
WtrueTestTraining = zeros(size(L_te,1),size(L_tr,1));
WtrueTestTraining(GT>0)=1;
%% Parameter setting
bit = 32; 
%% Learn LCMFH
[B_I,B_T,tB_I,tB_T] = main_LCMFH(I_tr, T_tr, I_te, T_te, L_tr, bit);
%% Compute mAP
Dhamm = hammingDist(tB_I, B_T)';    
[~, HammingRank]=sort(Dhamm,1);
mapIT = map_rank(L_tr,L_te,HammingRank); 
Dhamm = hammingDist(tB_T, B_I)';    
[~, HammingRank]=sort(Dhamm,1);
mapTI = map_rank(L_tr,L_te,HammingRank); 
map = [mapIT(100),mapTI(100)];

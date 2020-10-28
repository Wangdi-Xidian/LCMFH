function [Bi_Ir,Bt_Tr,Bi_Ie,Bt_Te,traintime,testtime] = main_LCMFH(I_tr, T_tr, I_te, T_te, L, bits, lambda, mu, gamma, maxIter)
% Reference:
% Di Wang, Xinbo Gao, Xiumei Wang, and Lihuo He. 
% Label Consistent Matrix Factorization Hashing. 
% IEEE Transactions on Pattern Analysis and Machine Intelligence, 41(10):2466 - 2479, 2019.
% (Manuscript)
%
% Contant: Di Wang (wangdi@xidain.edu.cn)
%
% Parameter Setting
if ~exist('lambda','var')
    lambda = 0.5;
end
if ~exist('mu','var')
    mu = 10;
end
if ~exist('gamma','var')
    gamma = 0.001;
end
if ~exist('maxIter','var')
    maxIter = 20;
end
traintime1 = cputime;

% Centering
I_tr = bsxfun(@minus, I_tr, mean(I_tr, 1));
T_tr = bsxfun(@minus, T_tr, mean(T_tr, 1));

if isvector(L) 
    L = sparse(1:length(L), double(L), 1); L = full(L);
end
L = normalizeFea(L);

fprintf('start solving LCMFH...\n');
[P1, P2, Z] = solveLCMFH(I_tr', T_tr', L, lambda, mu, gamma, bits, maxIter);
Yi_tr = sign((bsxfun(@minus, L*Z, mean(L*Z,1))));
Yt_tr = sign((bsxfun(@minus, L*Z, mean(L*Z,1))));
Yi_tr(Yi_tr<0) = 0;
Yt_tr(Yt_tr<0) = 0;
Bt_Tr = compactbit(Yt_tr);
Bi_Ir = compactbit(Yi_tr);
traintime2 = cputime;
traintime = traintime2 - traintime1;

testtime1 = cputime;
I_te = bsxfun(@minus, I_te, mean(I_te, 1));
T_te = bsxfun(@minus, T_te, mean(T_te, 1));
Yi_te = sign((bsxfun(@minus, I_te*P1, mean(I_te*P1,1))));
Yt_te = sign((bsxfun(@minus, T_te*P2, mean(T_te*P2,1))));
Yi_te(Yi_te<0) = 0;
Yt_te(Yt_te<0) = 0;
Bt_Te = compactbit(Yt_te);
Bi_Ie = compactbit(Yi_te);
testtime2 = cputime;
testtime = testtime2 - testtime1;
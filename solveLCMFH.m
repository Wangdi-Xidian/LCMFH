function [ P1, P2, Z, U1, U2, obj] = solveLCMFH( X1, X2, L, lambda, mu, gamma, bits, maxIter)
%solveLCMFH Summary of this function goes here
% Label Consistent Matrix Factorization Hashing
%   minimize_{U1, U2, Z1, Z2}    lambda*||X1 - U1 * Z' * L'||^2 + 
%      (1 - lambda)||X2 - U2 * Z' * L'||^2 + 
%      mu * ||L * Z - X1' * P1||^2 + mu * ||L * Z - X2' * P2||^2
%      gamma * (||U1||^2 + ||U2||^2 + ||L * Z||^2 + ||P1||^2  + ||P2||^2)
% Notation:
% X1: data matrix of View1, each column is a sample vector
% X2: data matrix of View2, each column is a sample vector
% L: label matrix of X1 and X2, each row is a label vector
% lambda: trade off between different views
% mu: trade off between matrix factorization and cross correlations
% gamma: parameter to control the model complexity
%
% Version1.0 -- May/2015
% Written by Di Wang 
%
%

%% Initialization
row = size(X1,1);
rowt = size(X2,1);
colL = size(L,2);
U1 = rand(row, bits);
U2 = rand(rowt, bits);
Z = rand(colL, bits);
P1 = rand(row, bits);
P2 = rand(rowt, bits);
threshold = 0.01;
lastF = 99999999;
iter = 1;
obj = zeros(maxIter, 1);

while (true)
    % update U1 and U2
    U1 = X1 * L * Z / (Z' * L' * L * Z + gamma/lambda * eye(bits));
    U2 = X2 * L * Z / (Z' * L' * L * Z + gamma/(1-lambda) * eye(bits));
    
    % update Z
    Z_left = (L' * L) \ (lambda * L' * X1' * U1 + (1-lambda) * L' * X2' * U2 + mu *  L' * X1' * P1 + mu *  L' * X2' * P2); 
    Z = Z_left / (lambda * U1' * U1 + (1-lambda) * U2' * U2 + (2*mu + gamma) * eye(bits));
    
    % update P1 and P2
    P1 = (X1 * X1' + gamma / mu * eye(row)) \ (X1 * L * Z);
    P2 = (X2 * X2' + gamma / mu * eye(rowt)) \ (X2 * L * Z);
    
    % compute objective function
    norm1 = lambda * norm(X1 - U1 * Z' * L', 'fro');
    norm2 = (1 - lambda) * norm(X2 - U2 * Z' * L', 'fro');
    norm3 = mu * norm(L*Z - X1'*P1, 'fro') + mu * norm(L*Z - X2'*P2, 'fro');
    norm4 = gamma * (norm(U1, 'fro') + norm(U2, 'fro') + norm(L*Z, 'fro') + norm(P1, 'fro') + norm(P2, 'fro'));
    currentF= norm1 + norm2 + norm3 + norm4;
    obj(iter) = currentF;
    fprintf('\nobj at iteration %d: %.4f\n reconstruction error for image: %.4f,\n reconstruction error for text: %.4f,\n reconstruction error for linear projection: %.4f,\n regularization term: %.4f\n\n', iter, currentF, norm1, norm2, norm3, norm4);
    if (lastF - currentF) < threshold
        fprintf('algorithm converges...\n');
        fprintf('final obj: %.4f\n reconstruction error for image: %.4f,\n reconstruction error for text: %.4f,\n reconstruction error for linear projection: %.4f,\n regularization term: %.4f\n\n', currentF, norm1, norm2, norm3, norm4);
        return;
    end
    if iter>=maxIter
        return
    end
    iter = iter + 1;
    lastF = currentF;
end
return;
end

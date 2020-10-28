function X = normalizeFea(X)
% X: n*m
Length = sqrt(sum(X.^2, 2));
Length(Length <= 0) = 1e-8; % avoid division by zero problem for unlabeled rows
Lambda = 1./ Length;
X = diag(sparse(Lambda)) * X;
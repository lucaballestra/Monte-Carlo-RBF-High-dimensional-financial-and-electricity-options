%% CallOnTheMaxLeland_ConditionNumbers.m
% MQ-RBF collocation + implicit Euler + (outer) MC over random/QMC node sets
% Pricing: max-call on d assets with additional LEL-type nonlinear term.
%
% This version also computes the condition numbers of:
%   A : interpolation/collocation matrix, called A0 in the code
%   L : implicit Euler matrix, called AA in the code
%
% Run:
%   >> CallOnTheMaxLeland_ConditionNumbers

clearvars; clc;

%% Model / contract parameters
r      = 0.03;
T      = 1.00;
strike = 100;

numAssets = 8;

dividend = zeros(numAssets,1);
vol      = zeros(numAssets,1);
for i = 1:numAssets
    vol(i) = 0.2 + 0.1 * mod(i+1,2);   % 0.2,0.3,0.2,0.3,...
end

corr = zeros(numAssets);
for i = 1:numAssets
    for j = 1:numAssets
        corr(i,j) = ((-1)^(i+j)) * (i+j) / (3*numAssets + 1);
    end
end
corr(1:numAssets+1:end) = 1;

spot0 = strike * ones(numAssets,1);

% Leland parameters
cLEL       = 0.01 * ones(numAssets,1);
deltaTLEL  = 0.01;

%% Discretization / simulation parameters
N            = 3000;   % number of RBF centers
numTimeSteps = 20;     % number of time levels
nMC          = 50;     % outer MC runs over random/QMC node sets

dt = T / (numTimeSteps - 1);

%% Node generation settings in log-space x = log(S/S0)
xMin = -1.0;
xMax =  1.0;

%% Output containers
priceSamples = zeros(nMC,1);

% Condition numbers:
% A0 is the matrix A in the paper.
% AA is the matrix L in the paper.
CondA = zeros(nMC,1);
CondL = zeros(nMC,1);

rng(1,'twister');  % reproducible scrambling if Sobol is used

tAll = tic;

for mc = 1:nMC

    %% Nodes: first node is x = 0, hence S = spot0
    [xNodes, sNodes] = generateNodesQMC(N, spot0, xMin, xMax);

    %% Shape parameter per node
    meanDist = meanNodeDistance(xNodes);
    shape    = 0.5 * meanDist;

    %% MQ-RBF matrices
    mats = assembleMQMatrices(xNodes, shape);

    A0 = mats.A0;  % interpolation matrix A in the paper
    A1 = mats.A1;  % cell{d,1} of first derivative matrices
    A2 = mats.A2;  % cell{d,d} of second derivative matrices

    %% Condition number of A
    CondA(mc) = cond(A0);

    %% Payoff: max(max_i S_i - K, 0)
    payoff = max(max(sNodes,[],1).' - strike, 0);

    %% RBF coefficients at maturity
    coef  = A0 \ payoff;
    value = payoff;

    %% PDE operator in x = log(S/S0)
    % drift_i = r - q_i - 0.5 sigma_i^2
    drift = r - dividend - 0.5 * (vol.^2);

    op = -r * A0;

    for i = 1:numAssets
        op = op + drift(i) * A1{i};
    end

    for i = 1:numAssets
        for j = 1:numAssets
            op = op + 0.5 * vol(i) * vol(j) * corr(i,j) * A2{i,j};
        end
    end

    %% Implicit Euler matrix
    % This is the matrix L in the paper.
    % It is called AA here to avoid confusion with the lower triangular
    % factor of the LU factorization.
    AA = A0 - dt * op;

    %% Condition number of L
    CondL(mc) = cond(AA);

    %% LU factorization reused at every time step
    [Lfac,Ufac,Pfac] = lu(AA);

    %% Leland coefficient
    lelCoeff   = cLEL(:);
    coefFactor = 0.5 * sqrt(2/(pi*deltaTLEL));

    invS = 1 ./ sNodes;   % [d x N]

    %% Backward time stepping
    for n = 1:(numTimeSteps-1)

        lelTerm = zeros(N,1);

        % For each i, build the row-wise quadratic form:
        %
        % q(node) = b(node,:) * corr * b(node,:)'
        %
        % where
        %
        % b_j = d^2V/(dS_i dS_j) * S_j * sigma_j.
        for i = 1:numAssets

            firstDer_i = A1{i} * coef;       % dV/dx_i at all nodes
            invSi      = invS(i,:).';        % [N x 1]
            invSi2     = invSi.^2;

            bMat = zeros(N, numAssets);

            for j = 1:numAssets

                secondDer_x = A2{i,j} * coef;  % d2V/(dx_i dx_j)

                % Chain rule for i ~= j:
                %
                % d2V/(dS_i dS_j)
                % =
                % d2V/(dx_i dx_j) * 1/(S_i S_j).
                secondDer_S = secondDer_x .* (invSi .* invS(j,:).');

                if i == j
                    % Chain rule for i = j:
                    %
                    % d2V/dS_i^2
                    % =
                    % (1/S_i^2) * (d2V/dx_i^2 - dV/dx_i).
                    secondDer_S = secondDer_S - firstDer_i .* invSi2;
                end

                bMat(:,j) = secondDer_S .* sNodes(j,:).' .* vol(j);

            end

            q = sum((bMat * corr) .* bMat, 2);

            Si = sNodes(i,:).';    % [N x 1]
            lelTerm = lelTerm + lelCoeff(i) .* Si .* sqrt(max(q,0));

        end

        rhs = value + dt * coefFactor * lelTerm;

        coef = Ufac \ (Lfac \ (Pfac * rhs));

        value = A0 * coef;

    end

    %% Price at the first node, which is x = 0, hence S = spot0
    priceSamples(mc) = value(1);

    fprintf(['MC %3d/%3d - running mean price: %.10f | ', ...
             'cond(A) mean: %.4e | cond(L) mean: %.4e\n'], ...
             mc, nMC, mean(priceSamples(1:mc)), ...
             mean(CondA(1:mc)), mean(CondL(1:mc)));

end

%% Summary statistics
timePerRun = toc(tAll) / nMC;

fprintf('\n================ SUMMARY RESULTS ================\n');

fprintf('\nAverage runtime per MC run: %.6f s\n', timePerRun);

fprintf('\nOption price:\n');
fprintf('  Mean:       %.10f\n', mean(priceSamples));
fprintf('  Std Error:  %.10f\n', std(priceSamples) / sqrt(nMC));
fprintf('  Min:        %.10f\n', min(priceSamples));
fprintf('  Max:        %.10f\n', max(priceSamples));

fprintf('\nCondition number of A:\n');
fprintf('  Mean:       %.10e\n', mean(CondA));
fprintf('  Std Dev:    %.10e\n', std(CondA));
fprintf('  Min:        %.10e\n', min(CondA));
fprintf('  Max:        %.10e\n', max(CondA));

fprintf('\nCondition number of L:\n');
fprintf('  Mean:       %.10e\n', mean(CondL));
fprintf('  Std Dev:    %.10e\n', std(CondL));
fprintf('  Min:        %.10e\n', min(CondL));
fprintf('  Max:        %.10e\n', max(CondL));

save('results_condition_numbers_callmax.mat', ...
     'N', 'numAssets', 'numTimeSteps', 'nMC', ...
     'r', 'T', 'strike', 'spot0', 'vol', 'corr', ...
     'cLEL', 'deltaTLEL', ...
     'timePerRun', ...
     'priceSamples', 'CondA', 'CondL');

%% ========================= Local functions ==============================

function [xNodes, sNodes] = generateNodesQMC(N, spot0, xMin, xMax)
% Generate N nodes in log-space x, centered at x = 0, inside [xMin,xMax]^d.
% First node is x = 0 so that sNodes(:,1) = spot0.

d = numel(spot0);

x0   = zeros(d,1);
xmin = xMin * ones(d,1);
xmax = xMax * ones(d,1);

sigmaVec = 0.2 * (xmax - xmin);

X = zeros(d, N);
X(:,1) = x0;

need = N - 1;
got  = 0;

useSobol = (exist('sobolset','file') == 2);

if useSobol
    p = sobolset(d);
    p = scramble(p,'MatousekAffineOwen');
    skip = 0;
end

while got < need

    batch = max(4*(need-got), 1024);

    if useSobol
        p.Skip = skip;
        U = net(p, batch);
        skip = skip + batch;
    else
        U = rand(batch, d);
    end

    % N(0,1) without norminv
    Z = sqrt(2) * erfinv(2*U - 1);

    % Gaussian cloud around x0, anisotropic by sigmaVec
    Y = bsxfun(@plus, x0.', bsxfun(@times, Z, sigmaVec.'));

    % Rejection to box
    ok = all(bsxfun(@ge, Y, xmin.'), 2) & ...
         all(bsxfun(@le, Y, xmax.'), 2);

    Y = Y(ok,:);

    take = min(size(Y,1), need-got);

    if take > 0
        X(:, got+2 : got+1+take) = Y(1:take,:).';
        got = got + take;
    end

end

xNodes = X;
sNodes = spot0(:) .* exp(xNodes);

end

function meanDist = meanNodeDistance(xNodes)
% meanDist(j) = mean Euclidean distance of node j from all other nodes.

X = xNodes.';                 % [N x d]
N = size(X,1);

ss = sum(X.^2,2);
D2 = ss + ss.' - 2*(X*X.');
D2(D2 < 0) = 0;

D = sqrt(D2);
D(1:N+1:end) = 0;

meanDist = sum(D,2) / (N-1);

end

function mats = assembleMQMatrices(xNodes, shape)
% Multiquadric RBF:
%
%   phi_j(x) = sqrt(1 + sum_k ((x_k - x_{k,j})^2 / c_j^2))
%
% with node-dependent shape parameter c_j = shape(j).
%
% Returns:
%   A0:      [N x N]
%   A1{k}:   d/dx_k operator matrix
%   A2{k,h}: d2/(dx_k dx_h) operator matrix

shape = shape(:);

c2 = (shape.^2).';   % [1 x N], per-column center j
c4 = (shape.^4).';   % [1 x N], per-column center j

d = size(xNodes,1);
N = size(xNodes,2);

rc2 = ones(N,N);

for k = 1:d
    Dk  = bsxfun(@minus, xNodes(k,:).', xNodes(k,:));
    rc2 = rc2 + bsxfun(@rdivide, Dk.^2, c2);
end

rc   = sqrt(rc2);
inv1 = 1 ./ rc;
inv2 = inv1 ./ rc2;

A0 = rc;

A1 = cell(d,1);

for k = 1:d
    Dk = bsxfun(@minus, xNodes(k,:).', xNodes(k,:));
    A1{k} = bsxfun(@rdivide, inv1 .* Dk, c2);
end

A2 = cell(d,d);

for k = 1:d

    Dk = bsxfun(@minus, xNodes(k,:).', xNodes(k,:));

    A2{k,k} = -bsxfun(@rdivide, inv2 .* (Dk.^2), c4) + ...
               bsxfun(@rdivide, inv1, c2);

    for h = (k+1):d

        Dh = bsxfun(@minus, xNodes(h,:).', xNodes(h,:));

        A2kh = -bsxfun(@rdivide, inv2 .* (Dk .* Dh), c4);

        A2{k,h} = A2kh;
        A2{h,k} = A2kh;

    end

end

mats = struct('A0', A0, 'A1', {A1}, 'A2', {A2});

end
%% CallOnTheMaxLeland_Greeks.m
% MQ-RBF collocation + implicit Euler + outer MC over random/QMC node sets
% Pricing: call on the maximum of d assets with Leland transaction costs.
%
% Output:
%   price
%   Delta1  = dV/dS1
%   Gamma11 = d2V/dS1^2
%   Gamma12 = d2V/(dS1 dS2)
%
% IMPORTANT:
% The PDE is solved in log-space:
%       x_i = log(S_i / S_i0).
% Greeks are converted back to S-space by the chain rule:
%
%   dV/dS1 = (1/S1) dV/dx1
%
%   d2V/dS1^2 = (1/S1^2) * (d2V/dx1^2 - dV/dx1)
%
%   d2V/(dS1 dS2) = (1/(S1*S2)) * d2V/(dx1 dx2),   for 1 ~= 2.
%
% Run:
%   >> CallOnTheMaxLeland_Greeks

clearvars; clc;

%% Model / contract parameters
r      = 0.03;
T      = 1.00;
strike = 100;

numAssets = 8;

dividend = zeros(numAssets,1);

vol = zeros(numAssets,1);
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

%% Leland parameters
% Set cLEL = 0 for the linear Black-Scholes benchmark.
% Set cLEL = 0.01 for the nonlinear Leland case of the paper.
cLEL      = 0*0.01 * ones(numAssets,1);
deltaTLEL = 0.01;

%% Discretization / simulation parameters
N            = 3000;   % number of RBF centers
numTimeSteps = 20;     % number of backward time steps
nMC          = 50;     % outer MC/QMC repetitions

dt = T / numTimeSteps;

%% Node generation settings in log-space x = log(S/S0)
xMin = -1.0;
xMax =  1.0;

%% Output arrays
priceSamples   = zeros(nMC,1);
delta1Samples  = zeros(nMC,1);
gamma11Samples = zeros(nMC,1);
gamma12Samples = zeros(nMC,1);

rng(1,'twister');

tAll = tic;

for mc = 1:nMC

    %% Nodes: first node is x = 0, hence S = spot0
    [xNodes, sNodes] = generateNodesQMC(N, spot0, xMin, xMax);

    %% Shape parameter per node: epsilon_k = q * mean distance
    meanDist = meanNodeDistance(xNodes);
    shape    = 0.5 * meanDist;

    %% MQ-RBF matrices
    mats = assembleMQMatrices(xNodes, shape);

    A0 = mats.A0;
    A1 = mats.A1;     % A1{i}   gives d/dx_i
    A2 = mats.A2;     % A2{i,j} gives d2/(dx_i dx_j)

    %% Terminal payoff: max(max_i S_i - K, 0)
    payoff = max(max(sNodes,[],1).' - strike, 0);

    %% Coefficients at maturity
    coef  = A0 \ payoff;
    value = payoff;

    %% Linear PDE operator in x = log(S/S0)
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
    AA = A0 - dt * op;
    [L,U,P] = lu(AA);

    %% Leland constants
    coefFactor = 0.5 * sqrt(2/(pi*deltaTLEL));
    useLeland  = any(abs(cLEL) > 0);

    %% Backward time stepping
    for n = 1:numTimeSteps

        if useLeland
            lelTerm = computeLelandTermY(coef, A1, A2, vol, corr, cLEL);
        else
            lelTerm = zeros(N,1);
        end

        rhs   = value + dt * coefFactor * lelTerm;
        coef  = U \ (L \ (P * rhs));
        value = A0 * coef;

    end

    %% Price and Greeks at the first node x = 0, S = spot0
    idx0 = 1;

    price0 = value(idx0);

    Vx1    = A1{1}   * coef;
    Vx1x1  = A2{1,1} * coef;
    Vx1x2  = A2{1,2} * coef;

    S1 = spot0(1);
    S2 = spot0(2);

    delta1  = Vx1(idx0) / S1;
    gamma11 = (Vx1x1(idx0) - Vx1(idx0)) / (S1^2);
    gamma12 = Vx1x2(idx0) / (S1*S2);

    priceSamples(mc)   = price0;
    delta1Samples(mc)  = delta1;
    gamma11Samples(mc) = gamma11;
    gamma12Samples(mc) = gamma12;

    fprintf(['MC %3d/%3d | ', ...
             'price mean = %.10f | ', ...
             'Delta1 mean = %.10f | ', ...
             'Gamma11 mean = %.10e | ', ...
             'Gamma12 mean = %.10e\n'], ...
             mc, nMC, ...
             mean(priceSamples(1:mc)), ...
             mean(delta1Samples(1:mc)), ...
             mean(gamma11Samples(1:mc)), ...
             mean(gamma12Samples(1:mc)));

end

timePerRun = toc(tAll) / nMC;

%% Final summary
fprintf('\n================ FINAL RESULTS ================\n');
fprintf('Average runtime per MC run: %.6f s\n', timePerRun);

fprintf('\nPrice:\n');
fprintf('  Mean:       %.10f\n', mean(priceSamples));
fprintf('  Std Error:  %.10f\n', std(priceSamples) / sqrt(nMC));
fprintf('  Min:        %.10f\n', min(priceSamples));
fprintf('  Max:        %.10f\n', max(priceSamples));

fprintf('\nDelta1 = dV/dS1:\n');
fprintf('  Mean:       %.10f\n', mean(delta1Samples));
fprintf('  Std Error:  %.10f\n', std(delta1Samples) / sqrt(nMC));
fprintf('  Min:        %.10f\n', min(delta1Samples));
fprintf('  Max:        %.10f\n', max(delta1Samples));

fprintf('\nGamma11 = d2V/dS1^2:\n');
fprintf('  Mean:       %.10e\n', mean(gamma11Samples));
fprintf('  Std Error:  %.10e\n', std(gamma11Samples) / sqrt(nMC));
fprintf('  Min:        %.10e\n', min(gamma11Samples));
fprintf('  Max:        %.10e\n', max(gamma11Samples));

fprintf('\nGamma12 = d2V/(dS1 dS2):\n');
fprintf('  Mean:       %.10e\n', mean(gamma12Samples));
fprintf('  Std Error:  %.10e\n', std(gamma12Samples) / sqrt(nMC));
fprintf('  Min:        %.10e\n', min(gamma12Samples));
fprintf('  Max:        %.10e\n', max(gamma12Samples));

save('results_greeks.mat', ...
     'N', 'numAssets', 'numTimeSteps', 'nMC', ...
     'r', 'T', 'strike', 'spot0', 'vol', 'corr', ...
     'cLEL', 'deltaTLEL', ...
     'timePerRun', ...
     'priceSamples', ...
     'delta1Samples', ...
     'gamma11Samples', ...
     'gamma12Samples');

%% ========================= Local functions ==============================

function [xNodes, sNodes] = generateNodesQMC(N, spot0, xMin, xMax)
% Generate N nodes in log-space x, centered at x = 0, inside [xMin,xMax]^d.
% First node is x = 0, so that sNodes(:,1) = spot0.

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
% with node-dependent shape c_j = shape(j).
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

function lelTerm = computeLelandTermY(coef, A1, A2, vol, corr, cLEL)
% Compute the Leland nonlinear term in log variables.
%
% In y = log(S/S0), the paper writes:
%
%   G_y = 0.5 * sqrt(2/(pi*deltaTLEL)) *
%         sum_i gamma_i * sqrt( sum_j sum_k H_ij rho_jk sigma_j sigma_k H_ik )
%
% Here the external factor 0.5*sqrt(2/(pi*deltaTLEL)) is applied outside this
% function in the time stepping.
%
% H_ij =
%   V_{y_i y_i} - V_{y_i},  if i = j
%   V_{y_i y_j},            if i ~= j

d = numel(vol);
N = size(A1{1},1);

lelTerm = zeros(N,1);

% Precompute first derivatives and second derivatives in y
Vy  = cell(d,1);
Vyy = cell(d,d);

for i = 1:d
    Vy{i} = A1{i} * coef;
end

for i = 1:d
    for j = 1:d
        Vyy{i,j} = A2{i,j} * coef;
    end
end

% Build the nonlinear term nodewise
for i = 1:d

    H = zeros(N,d);

    for j = 1:d
        if i == j
            H(:,j) = Vyy{i,i} - Vy{i};
        else
            H(:,j) = Vyy{i,j};
        end
    end

    % Quadratic form:
    % q_l = H_l * diag(vol) * corr * diag(vol) * H_l'
    Cvol = diag(vol) * corr * diag(vol);

    q = sum((H * Cvol) .* H, 2);

    lelTerm = lelTerm + cLEL(i) * sqrt(max(q,0));

end

end
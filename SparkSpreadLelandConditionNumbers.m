%% SparkSpreadLeland_RBF_QMC.m
% MQ-RBF collocation + implicit Euler + (outer) MC over random/QMC node sets
%
% Pricing: spark-spread option under a 2-asset Heston-type model
%          with Leland-style transaction costs on the second asset (gas leg).
%
% State variables (as in the paper):
%   y1 = log(S1/S1,0),  y2 = v1,  y3 = log(S2/S2,0),  y4 = v2
%
% The nonlinear term follows the Leland model and is treated explicitly (IMEX Euler):
%   Linear operator: implicit
%   Leland nonlinearity: explicit
%
% Run:
%   >> SparkSpreadLeland_RBF_QMC

clearvars; clc;

%% Seed random fisso
seed = 1;
rng(seed,'twister');


%% Model / contract parameters (Spark-spread with stochastic vol)
r = 0.1;      % risk-free rate
T = 0.5;      % maturity (years)

% Asset 1 (electricity) Heston-like parameters
qdiv  = 0;        % dividend yield (or convenience yield)
kHest = 2;        % mean reversion speed
theta = 0.01;     % long-run variance
sigma = 0.1;      % vol of variance
rho   = -0.5;     % corr(dW_S1, dW_v1)
S0    = 43.73;    % initial spot
V0    = 0.01;     % initial variance

% Asset 2 (gas) Heston-like parameters
S02    = 20.29;
V02    = 0.01;
qdiv2  = 0;
kHest2 = 0.5;
theta2 = 0.01;
sigma2 = 0.2;

% Cross-correlations among the 4 Brownian drivers
% (S1,v1,S2,v2) structure consistent with the paper
rhoSS2  = 0.8;   % corr(S1, S2)
rhoS2V2 = -0.5;  % corr(S2, v2)
rhoVV2  = 0.6;   % corr(v1, v2)
rhoSV2  = -0.6;  % corr(S1, v2)
rhoS2V  = -0.6;  % corr(S2, v1)

eta = 0.5;       % conversion efficiency in spark spread payoff

%% Leland transaction-cost parameters (applied to gas leg)
% In this script the nonlinear term is tied to asset 2 (S2, v2) only.
cLEL      = 0.01;   % transaction-cost intensity gamma (paper notation)
deltatLEL = 0.01;  % re-hedging interval (delta t in Leland)

%% Discretization / simulation parameters
N          = 1000;  % number of RBF centers (collocation nodes)
NtimesRBF  = 20;    % number of time levels (implicit Euler steps are NtimesRBF-1)
nMC        = 50;    % outer Monte Carlo runs over random/QMC node sets

dt = T/(NtimesRBF-1);

%% Output containers
SolVector = zeros(nMC,1);

% Condition numbers of the matrices appearing in the RBF method:
% A0 is the interpolation matrix A in the paper.
% AA is the implicit Euler matrix L in the paper.
CondA = zeros(nMC,1);
CondL = zeros(nMC,1);

tAll = tic;

%% ========================================================================
%  Outer Monte Carlo loop (each run regenerates nodes and rebuilds RBF mats)
%  ========================================================================
for kMC = 1:nMC

    %----------------------------------------------------------------------
    % 1) Generate node set in the 4D state space:
    %    x = [ log(S1/S0); v1; log(S2/S02); v2 ]
    %    First node is forced to be the initial state (x0).
    %----------------------------------------------------------------------
    [xvet,Svet,Vvet,S2vet,V2vet] = generateNodesQMC(N,S0,V0,S02,V02);

    %----------------------------------------------------------------------
    % 2) Node-dependent MQ shape parameters:
    %    epsilon_k = 0.5 * average distance of node k from all others
    %----------------------------------------------------------------------
    distMean = meanNodeDistance(xvet);
    cvet     = 0.5 * distMean;

    %----------------------------------------------------------------------
    % 3) Assemble MQ-RBF collocation matrices:
    %    A0  : interpolation matrix
    %    A1* : first-derivative operators
    %    A2* : second-derivative (and mixed) operators
    %
    % IMPORTANT: Derivatives are with respect to the "y" variables:
    %   y1=log(S1/S0), y2=v1, y3=log(S2/S02), y4=v2.
    %----------------------------------------------------------------------
   mats = assembleMQMatrices(xvet, cvet);

A0 = mats.A0;

% Condition number of the interpolation matrix A in the paper
CondA(kMC) = cond(A0);

    % First and second derivatives for (y1,y2) = (logS1, v1)
    A1S = mats.A1S;   A2S  = mats.A2S;    A2SV = mats.A2SV;
    A1V = mats.A1V;   A2V  = mats.A2V;

    % First and second derivatives for (y3,y4) = (logS2, v2)
    A1Ssecond = mats.A1Ssecond;  A2Ssecond = mats.A2Ssecond;
    A1Vsecond = mats.A1Vsecond;  A2Vsecond = mats.A2Vsecond;

    % Cross derivatives among the 4 coordinates
    A2SSsecond        = mats.A2SSsecond;        % y1-y3
    A2SsecondVsecond  = mats.A2SsecondVsecond;  % y3-y4
    A2VVsecond        = mats.A2VVsecond;        % y2-y4
    A2SVsecond        = mats.A2SVsecond;        % y1-y4
    A2VSsecond        = mats.A2VSsecond;        % y2-y3

    %----------------------------------------------------------------------
    % 4) Terminal payoff at maturity (spark spread):
    %    Pi = max(S1 - S2/eta, 0)
    %----------------------------------------------------------------------
    payoff  = max(Svet - S2vet./eta, 0);

    % RBF coefficients at maturity: A0 * c(T) = payoff
    coefRBF = A0 \ payoff;
    solold  = payoff;

    % Precompute sqrt(v1), sqrt(v2) since they appear in cross terms
    sqrtV  = sqrt(Vvet);
    sqrtV2 = sqrt(V2vet);

    %----------------------------------------------------------------------
    % 5) PDE coefficients at collocation nodes (in log-space variables)
    %
    % For asset 1:
    %   dU/dt + (r-q1-0.5*v1) U_y1 + k1(theta1-v1) U_y2
    %   + 0.5*v1 U_y1y1 + 0.5*sigma1^2*v1 U_y2y2 + rho*sigma1*v1 U_y1y2 + ...
    %
    % For asset 2 analogous on (y3,y4), plus cross terms among processes.
    %----------------------------------------------------------------------
    Coef1S  = (r - qdiv  - 0.5*Vvet);
    Coef2S  = (0.5*Vvet);
    Coef1V  = (kHest *(theta  - Vvet));
    Coef2V  = (0.5*sigma*sigma*Vvet);
    Coef2SV = (rho*sigma*Vvet);

    Coef1Ssecond = (r - qdiv2 - 0.5*V2vet);
    Coef2Ssecond = (0.5*V2vet);
    Coef1Vsecond = (kHest2*(theta2 - V2vet));
    Coef2Vsecond = (0.5*sigma2*sigma2*V2vet);

    % Cross diffusion coefficients (collocated)
    Coef2SSsecond        = (rhoSS2  * (sqrtV.*sqrtV2));
    Coef2SsecondVsecond  = (rhoS2V2 * sigma2 * V2vet);
    Coef2VVsecond        = (rhoVV2  * sigma * sigma2 * (sqrtV.*sqrtV2));
    Coef2SVsecond        = (rhoSV2  * sigma2 * (sqrtV.*sqrtV2));
    Coef2VSsecond        = (rhoS2V  * sigma  * (sqrtV.*sqrtV2));

    %----------------------------------------------------------------------
    % 6) Implicit Euler matrix for the linear operator.
    %
    % We build:
    %   AA * c(t_{m-1}) = RHS(t_m)
    % in coefficient space, using A0 and derivative operator matrices.
    %
    % The discount term "-r U" leads to a factor (1 + r*dt) * A0 on the LHS
    % under backward Euler.
    %----------------------------------------------------------------------
    AA = (1 + r*dt) * A0;

    % First-order (drift) terms
    AA = AA - dt * bsxfun(@times, A1S,        Coef1S);
    AA = AA - dt * bsxfun(@times, A1V,        Coef1V);
    AA = AA - dt * bsxfun(@times, A1Ssecond,  Coef1Ssecond);
    AA = AA - dt * bsxfun(@times, A1Vsecond,  Coef1Vsecond);

    % Second-order (diffusion) terms
    AA = AA - dt * bsxfun(@times, A2S,                 Coef2S);
    AA = AA - dt * bsxfun(@times, A2V,                 Coef2V);
    AA = AA - dt * bsxfun(@times, A2SV,                Coef2SV);
    AA = AA - dt * bsxfun(@times, A2Ssecond,           Coef2Ssecond);
    AA = AA - dt * bsxfun(@times, A2Vsecond,           Coef2Vsecond);
    AA = AA - dt * bsxfun(@times, A2SSsecond,          Coef2SSsecond);
    AA = AA - dt * bsxfun(@times, A2SsecondVsecond,    Coef2SsecondVsecond);
    AA = AA - dt * bsxfun(@times, A2VVsecond,          Coef2VVsecond);
    AA = AA - dt * bsxfun(@times, A2SVsecond,          Coef2SVsecond);
    AA = AA - dt * bsxfun(@times, A2VSsecond,          Coef2VSsecond);

% Condition number of the implicit Euler matrix L in the paper.
% In this code the matrix L of the paper is denoted by AA, because the name L
% is reserved below for the LU factorization.
CondL(kMC) = cond(AA);

% LU factorization reused at every time step (matrix is time-independent)
[Lfac,Ufac,Pfac] = lu(AA);

    %----------------------------------------------------------------------
    % 7) Leland nonlinear term (explicit).
    %
    % In the paper (log variables), the Leland term for the gas leg involves:
    %   sqrt(2/(pi*delta_t)) * gamma * sqrt(v2) * sqrt( (...) )
    %
    % Here we implement the same structure:
    %   CoefLEL_dt collects dt * sqrt(2/(pi*delta_t)) * gamma * sqrt(v2) * S2
    %   termLEL uses derivatives scaled so that the final product matches
    %   the log-variable formula (no extra S2 remains after simplification).
    %----------------------------------------------------------------------
    CoefLEL_dt = dt * sqrt(2/(pi*deltatLEL)) * cLEL * sqrt(V2vet) .* S2vet;

    % Useful scaling for converting log-derivatives to S2-derivatives:
    % invS2 = 1/S2
    invS2 = 1 ./ S2vet;

    % (d^2/dy3^2 - d/dy3) operator in log-space, used in S2 second derivative:
    % For y3 = log(S2/S2,0),
    %   U_{S2S2} = (1/S2^2) * (U_{y3y3} - U_{y3})
    % so:
    %   S2 * U_{S2S2} = (1/S2) * (U_{y3y3} - U_{y3})
    D2S2 = A2Ssecond - A1Ssecond;

    %----------------------------------------------------------------------
    % 8) Backward time stepping (from maturity to t=0)
    %----------------------------------------------------------------------
    for k = 1:(NtimesRBF-1)

        % Compute the needed derivatives at nodes:
        % der2SS approximates: S2 * U_{S2S2}
        % der2SV approximates: S2 * U_{S2,v2}
        der2SS = (D2S2 * coefRBF) .* invS2;
        der2SV = (A2SsecondVsecond * coefRBF) .* invS2;

        % Leland "local volatility correction" term inside the square root:
        % sqrt( (S2 U_{S2S2})^2 + 2*rho*S2*... + (sigma_v2 U_{S2v2})^2 )
        termLEL = sqrt( ...
            der2SS.^2 ...
            + 2*rhoS2V2*sigma2*(der2SV.*der2SS) ...
            + (sigma2*sigma2)*(der2SV.^2) );

        % Explicit inclusion of the nonlinear term on the RHS (IMEX Euler):
        rhs = solold + termLEL .* CoefLEL_dt;

        % Solve the implicit linear step for new coefficients
        coefRBF = Ufac \ (Lfac \ (Pfac * rhs));

        % Recover nodal solution values U(t_{m-1}) = A0 * c(t_{m-1})
        solold = A0 * coefRBF;
    end

    % Price at the first node, which is forced to be the initial state
    SolVector(kMC) = solold(1);

    fprintf(['MC %3d/%3d - running mean price: %.10f | ', ...
         'cond(A) mean: %.4e | cond(L) mean: %.4e\n'], ...
    kMC, nMC, mean(SolVector(1:kMC)), ...
    mean(CondA(1:kMC)), mean(CondL(1:kMC)));
end

%% Summary stats
time_per_Run = toc(tAll) / nMC;

fprintf('\n================ SUMMARY RESULTS ================\n');

fprintf('\nAverage runtime per MC run: %.6f s\n', time_per_Run);

fprintf('\nOption price:\n');
fprintf('  Mean:       %.10f\n', mean(SolVector));
fprintf('  Std Error:  %.10f\n', std(SolVector)/sqrt(nMC));
fprintf('  Min:        %.10f\n', min(SolVector));
fprintf('  Max:        %.10f\n', max(SolVector));

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

save('results.mat', ...
     'N', 'nMC', 'NtimesRBF', 'time_per_Run', ...
     'SolVector', 'CondA', 'CondL');
%% ========================= Local functions ==============================

function [xvet,Svet,Vvet,S2vet,V2vet] = generateNodesQMC(N,S0,V0,S02,V02)
% Generate N nodes in the 4D state space:
%   x = [ log(S1/S0); v1; log(S2/S02); v2 ].
%
% Nodes are drawn from a Gaussian cloud centered at x0 and then accepted if
% they lie in the hyper-rectangle [xmin,xmax]. First node is exactly x0.

% Truncation bounds (chosen empirically, consistent with the paper)
Retmin = -1;  Retmax = 1;     % log-return bounds
Vmin   = 1e-4; Vmax  = 0.3;   % variance bounds

% Initial state (must be included as first node)
x0   = [0; V0; 0; V02];

% Box bounds
xmin = [Retmin; Vmin;  Retmin; Vmin];
xmax = [Retmax; Vmax;  Retmax; Vmax];

% Gaussian cloud scale (20% of box width per coordinate)
sigmaVec = 0.2 * (xmax - xmin);

% Allocate node matrix
X = zeros(4, N);
X(:,1) = x0;

need = N - 1;
got  = 0;

% Use Sobol + Owen scrambling if available (otherwise pseudo-random)
use_sobol = (exist('sobolset','file') == 2);
if use_sobol
    p = sobolset(4);
    p = scramble(p,'MatousekAffineOwen');
    skip = 0;
end

% Acceptance-rejection sampling loop
while got < need
    batch = max(4*(need-got), 1024);

    if use_sobol
        p.Skip = skip;
        U = net(p, batch);
        skip = skip + batch;
    else
        U = rand(batch, 4);
    end

    % Transform U ~ Uniform(0,1) to Z ~ N(0,1) without norminv
    Z = sqrt(2) * erfinv(2*U - 1);

    % Candidate points around x0 with anisotropic scaling
    Y = bsxfun(@plus, x0.', bsxfun(@times, Z, sigmaVec.'));

    % Accept only points inside the box
    okLo = all(bsxfun(@ge, Y, xmin.'), 2);
    okHi = all(bsxfun(@le, Y, xmax.'), 2);
    Y = Y(okLo & okHi,:);

    % Take as many as needed
    take = min(size(Y,1), need-got);
    if take > 0
        X(:, got+2 : got+1+take) = Y(1:take,:).';
        got = got + take;
    end
end

xvet = X;

% Map back to levels:
% y1=log(S1/S0), y3=log(S2/S02)  =>  S = S0 * exp(y)
Retvet  = xvet(1,:).';
Vvet    = xvet(2,:).';
Ret2vet = xvet(3,:).';
V2vet   = xvet(4,:).';

Svet  = S0  * exp(Retvet);
S2vet = S02 * exp(Ret2vet);

end

function meanDist = meanNodeDistance(xvet)
% meanDist(i) = average Euclidean distance from node i to all other nodes.

X = xvet.';     % [N x d]
N = size(X,1);

ss = sum(X.^2,2);
D2 = ss + ss.' - 2*(X*X.');
D2(D2 < 0) = 0;

D = sqrt(D2);
D(1:N+1:end) = 0;

meanDist = sum(D,2) / (N-1);

end

function mats = assembleMQMatrices(xvet, c)
% Assemble MQ-RBF interpolation and differentiation matrices.
%
% Multiquadric basis centered at node j:
%   phi_j(r) = sqrt( 1 + (r / c_j)^2 )
% with node-dependent shape parameter c_j.
%
% Here r^2 is computed in the 4D state space (y1,y2,y3,y4).

c  = c(:);
c2 = (c.^2).';
c4 = (c.^4).';

% Pairwise differences per coordinate (NxN matrices)
D1 = bsxfun(@minus, xvet(1,:).', xvet(1,:)); % y1 = log(S1/S0)
D2 = bsxfun(@minus, xvet(2,:).', xvet(2,:)); % y2 = v1
D3 = bsxfun(@minus, xvet(3,:).', xvet(3,:)); % y3 = log(S2/S02)
D4 = bsxfun(@minus, xvet(4,:).', xvet(4,:)); % y4 = v2

% r^2 / c^2 (center-dependent scaling via column-wise division by c_j^2)
rc2 = 1 ...
    + bsxfun(@rdivide, D1.^2, c2) ...
    + bsxfun(@rdivide, D2.^2, c2) ...
    + bsxfun(@rdivide, D3.^2, c2) ...
    + bsxfun(@rdivide, D4.^2, c2);

rc  = sqrt(rc2);

% Useful factors for derivatives
inv1 = 1 ./ rc;
inv2 = inv1 ./ rc2;

% Interpolation matrix
A0 = rc;

% First derivatives (d/dy_k)
A1S        = bsxfun(@rdivide, inv1 .* D1, c2);
A1V        = bsxfun(@rdivide, inv1 .* D2, c2);
A1Ssecond  = bsxfun(@rdivide, inv1 .* D3, c2);
A1Vsecond  = bsxfun(@rdivide, inv1 .* D4, c2);

% Second derivatives (d^2/dy_k^2)
A2S        = -bsxfun(@rdivide, inv2 .* (D1.^2), c4) + bsxfun(@rdivide, inv1, c2);
A2V        = -bsxfun(@rdivide, inv2 .* (D2.^2), c4) + bsxfun(@rdivide, inv1, c2);
A2Ssecond  = -bsxfun(@rdivide, inv2 .* (D3.^2), c4) + bsxfun(@rdivide, inv1, c2);
A2Vsecond  = -bsxfun(@rdivide, inv2 .* (D4.^2), c4) + bsxfun(@rdivide, inv1, c2);

% Mixed second derivatives (d^2/dy_k dy_h)
A2SV               = -bsxfun(@rdivide, inv2 .* (D1 .* D2), c4); % y1-y2
A2SSsecond         = -bsxfun(@rdivide, inv2 .* (D1 .* D3), c4); % y1-y3
A2SsecondVsecond   = -bsxfun(@rdivide, inv2 .* (D3 .* D4), c4); % y3-y4
A2VVsecond         = -bsxfun(@rdivide, inv2 .* (D2 .* D4), c4); % y2-y4
A2SVsecond         = -bsxfun(@rdivide, inv2 .* (D1 .* D4), c4); % y1-y4
A2VSsecond         = -bsxfun(@rdivide, inv2 .* (D2 .* D3), c4); % y2-y3

% Package outputs
mats = struct();
mats.A0  = A0;
% Condition number of the interpolation matrix A
mats.A1S = A1S;  mats.A2S  = A2S;   mats.A2SV = A2SV;
mats.A1V = A1V;  mats.A2V  = A2V;

mats.A1Ssecond = A1Ssecond;
mats.A2Ssecond = A2Ssecond;
mats.A1Vsecond = A1Vsecond;
mats.A2Vsecond = A2Vsecond;

mats.A2SSsecond        = A2SSsecond;
mats.A2SsecondVsecond  = A2SsecondVsecond;
mats.A2VVsecond        = A2VVsecond;
mats.A2SVsecond        = A2SVsecond;
mats.A2VSsecond        = A2VSsecond;

end

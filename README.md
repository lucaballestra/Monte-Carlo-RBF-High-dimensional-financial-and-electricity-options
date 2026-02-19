README.txt
==========

Meshless MQ-RBF + (Q)MC collocation codes (MATLAB)
-------------------------------------------------

This repository contains two self-contained MATLAB scripts implementing the meshless
multiquadric radial-basis-function (MQ-RBF) collocation method described in:

  L.V. Ballestra (2026), "A Meshless Method for Non-Linear High-Dimensional PDEs:
  Application to Financial and Electricity Options", work in preparation

The scripts solve high-dimensional option-pricing partial differential equations (PDEs) 
backward in time. The PDEs are nonlinear because they incorporate transaction costs,
modeled according to the Leland model, introduced in H. E. Leland (1985), 
Option Pricing and Replication with Transaction Costs, The Journal of Finance, 40(5), 
1283–1301.

Quasi-Monte Carlo (Sobol) point sets are used as RBF centers; the PDE is then solved
by implicit (or IMEX) Euler time stepping on the resulting collocation system.

Contents
--------
1) SparkSpreadLeland.m
   - 4D PDE (two-asset Heston-type stochastic volatility model)
   - Payoff: spark-spread  max(S1 - S2/eta, 0)
   - Nonlinear term: Leland-style transaction costs applied to the second asset (gas leg)
   - Time stepping: IMEX Euler (linear operator implicit, Leland term explicit)

2) CallOnTheMaxLeland.m
   - n-dimensional PDE (here set to n = 8) under a correlated Black–Scholes model
   - Payoff: max( max_i(S_i) - K, 0 )
   - Nonlinear term: Leland-type correction built from second derivatives (explicit)
   - Time stepping: implicit Euler for the linear part + explicit Leland term (IMEX)

Quick start
-----------
1. Open MATLAB and set the current folder to the repository root.
2. Run one script at a time from the Command Window, e.g.:

   >> SparkSpreadLeland
   >> CallOnTheMaxLeland

Both scripts:
- print a running estimate of the option price across repeated node-set simulations,
- save a file  results.mat  in the working directory with the samples and run metadata.

No external files are required: each script includes its helper functions at the bottom.

Dependencies
------------
- MATLAB (tested with standard functions such as LU factorization and erfinv).
- Optional: Statistics and Machine Learning Toolbox for Sobol sequences
  (functions sobolset / scramble). If unavailable, the codes automatically fall back to
  pseudo-random sampling (rand).

Method overview (high level)
----------------------------
For each outer run (k = 1,...,nMC):
1) Generate N collocation nodes in the transformed state space.
   - The first node is forced to be the initial state, so the price is read directly at node 1.
2) Build the MQ-RBF interpolation matrix A0 and the derivative matrices (first/second/mixed).
3) Impose the terminal condition (payoff at maturity) and compute RBF coefficients.
4) March backward in time with (I)MEX Euler:
   - linear differential operator handled implicitly via one LU factorization per run,
   - nonlinear Leland term evaluated explicitly at the known time level.
5) Store the price at the initial state; repeat and report mean / standard error.

Script details
--------------
A) SparkSpreadLeland.m
   State variables (as in the paper):
     y1 = log(S1/S1,0),  y2 = v1,  y3 = log(S2/S2,0),  y4 = v2

   Key parameters (editable near the top of the file):
   - Model parameters for both Heston-type factors (kappa, theta, sigma_v, correlations)
   - Contract parameters: r, T, eta, initial levels (S0, V0, S02, V02)
   - Leland parameters: gamma (cLEL) and re-hedging interval delta_t (deltatLEL)
   - Discretization: N (nodes), NtimesRBF (time levels), nMC (outer repetitions)

   Output variables saved in results.mat:
   - SolVector   : solution (option price) for each outer run
   - time_per_Run: average time per run
   - N           : number of nodes

B) CallOnTheMaxLeland.m
   Uses the log-space variables  x_i = log(S_i / S_i,0), i=1,...,n.

   Key parameters (editable near the top of the file):
   - numAssets   : number of assets (set to 8 to match the experiment in the paper)
   - r, T, strike, spot0, vol, correlation matrix
   - Leland parameters: cLEL (vector of gammas) and deltaTLEL
   - Discretization: N (nodes), numTimeSteps, nMC (outer repetitions)
   - Node box: [xMin, xMax]^n controls the log-return domain

   Output variables saved in results.mat:
   - priceSamples: solution (option price) for each outer run
   - timePerRun  : average time per run
   - N, numAssets

Performance / memory notes
--------------------------
These are dense RBF collocation matrices. With N=3000 the codes may require several GB
of RAM (especially CallOnTheMaxLeland.m, which stores many second-derivative matrices).
If you run into memory limits, consider reducing N, reducing numAssets, or using fewer
time steps / outer runs.

Reproducibility
---------------
CallOnTheMaxLeland.m sets a fixed RNG seed (rng(1,'twister')) for reproducibility.
Both scripts attempt to use scrambled Sobol points when available; different MATLAB
versions may produce slightly different results depending on Sobol implementation.


Author
------
Luca Vincenzo Ballestra
Department of Statistical Sciences "Paolo Fortunati"
Alma Mater Studiorum – University of Bologna

Contacts (see the official UniBo profile):
- Email: luca.ballestra@unibo.it
- Address: Via Belle Arti 41, Bologna (Italy)

License / disclaimer
--------------------
This research code is provided "as is", without warranty of any kind. Use at your own risk.

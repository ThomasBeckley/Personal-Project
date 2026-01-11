function [price, se, ci95] = heston_mc_call()
% Monte Carlo pricing of a European call option under the Heston model.

% contract inputs
S0 = 172;        % initial NVIDIA price (18 Dec)
K  = 220;        % strike price
T  = 1.0;        % maturity (years)
r  = 0.07;       % risk-free rate
q  = 0.0005;     % dividend yield

% heston parameters
v0    = 0.249;   % initial variance
kappa = 7.23;    % speed of mean reversion
theta = 0.249;   % long-run variance
xi    = 1.79;    % volatility of volatility
rho   = -0.70;   % correlation

% monte carlo controls
steps = 252;         % trading days in one year
paths = 200000;      % number of simulated paths
seed  = 42;          % fixed seed for reproducibility

nPlot  = 10;                     % number of paths stored for plotting
nStore = min(nPlot, paths);

dt      = T/steps;
sqrt_dt = sqrt(dt);
tgrid   = linspace(0, T, steps+1);

% fix random number generator so results are reproducible
rng(seed, 'twister');

% initialise vectors for all paths
S = S0 * ones(paths,1);     % stock price paths
v = v0 * ones(paths,1);     % variance paths

% store a small number of paths for plotting only
S_store      = zeros(steps+1, nStore);
S_store(1,:) = S0;

% generate correlated standard normal shocks
Z1 = randn(steps, paths);        % stock shocks
Zp = randn(steps, paths);        % independent shocks
Z2 = rho .* Z1 + sqrt(1 - rho^2) .* Zp;   % variance shocks

% time stepping
for t = 1:steps
    % full truncation to prevent negative variance
    v_pos = max(v, 0);

    % variance update (Euler–Maruyama)
    v = v ...
        + kappa*(theta - v_pos)*dt ...
        + xi*sqrt(v_pos).*(sqrt_dt * Z2(t,:)');
    v = max(v, 0);

    % stock update using exponential scheme
    S = S .* exp( ...
        (r - q - 0.5*v_pos)*dt ...
        + sqrt(v_pos).*(sqrt_dt * Z1(t,:)') );

    % store sample paths for plotting
    S_store(t+1,:) = S(1:nStore);
end

% payoff and discounting
ST         = S;
payoff     = max(ST - K, 0);
discPayoff = exp(-r*T) * payoff;

% monte carlo estimate and uncertainty
price = mean(discPayoff);
se    = std(discPayoff, 0) / sqrt(paths);
ci95  = price + 1.96 * se * [-1, 1];

fprintf('Heston MC European Call Price (NVDA): %.6f\n', price);
fprintf('Standard Error: %.6f\n', se);
fprintf('95%% CI: [%.6f, %.6f]\n', ci95(1), ci95(2));

% plots

% sample stock price paths
figure;
plot(tgrid, S_store, 'LineWidth', 1.0);
hold on;
yline(K, '--', 'Strike K', 'LineWidth', 1.5);
title(sprintf('Sample simulated NVDA paths (n = %d of %d)', nStore, paths));
xlabel('Time (years)');
ylabel('Stock price S_t');
grid on;

% histogram of payoffs
figure;
histogram(payoff, 80);
title(sprintf('Call payoffs from %d simulations', paths));
xlabel('Payoff');
ylabel('Frequency');
grid on;

% histogram of terminal stock prices
figure;
histogram(ST, 80);
hold on;
xline(K, '--', 'Strike K', 'LineWidth', 1.5);
title(sprintf('Terminal stock prices S_T from %d simulations', paths));
xlabel('S_T');
ylabel('Frequency');
grid on;

end

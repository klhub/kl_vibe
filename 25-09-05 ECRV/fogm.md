Prompt : How to use first-order Gauss Markov process to simulate a process with a given mean and standard deviation?

Here’s the simplest, correct way to simulate a **first-order Gauss-Markov (FOGM)** (a.k.a. Ornstein–Uhlenbeck / AR(1)) process with a **target mean** $\mu$ and **target stationary standard deviation** $\sigma$.

## Core model (discrete-time)

Pick a time step $\Delta t$ and a correlation time constant $\tau$ (how fast the process forgets). Then

$$
\phi = e^{-\Delta t/\tau},\qquad
q = \sigma^2\bigl(1-\phi^2\bigr).
$$

Simulate with

$$
x_k = \mu + \phi\,(x_{k-1}-\mu) + w_k,\quad w_k \sim \mathcal N(0,\,q).
$$

If you draw $x_0 \sim \mathcal N(\mu,\sigma^2)$ (or just set $x_0=\mu$), the sequence has stationary mean $\mu$ and stationary variance $\sigma^2$, and autocorrelation $\rho(m) = \phi^{\,m} = e^{-m\Delta t/\tau}$.

> If you only know the “memory” qualitatively, a handy pick is $\tau = t_{1/2} / \ln 2$ where $t_{1/2}$ is the correlation half-life.

## MATLAB: drop-in function

```matlab
function x = fogm_sim(N, dt, tau, mu, sigma, x0)
% FOGM / OU simulator with prescribed mean and stationary std
% N: number of samples
% dt: sample time
% tau: correlation time constant
% mu: target mean
% sigma: target stationary std
% x0: (optional) initial value; if omitted, draws from N(mu, sigma^2)

    phi = exp(-dt/tau);
    q   = sigma^2*(1 - phi^2);  % process noise variance

    if nargin < 6 || isempty(x0)
        x_prev = mu + sigma*randn;  % start in stationarity
    else
        x_prev = x0;
    end

    x = zeros(N,1);
    for k = 1:N
        w  = sqrt(q)*randn;
        xk = mu + phi*(x_prev - mu) + w;
        x(k) = xk;
        x_prev = xk;
    end
end
```

### Minimal usage example

```matlab
% Desired stats and dynamics
mu    = 10;         % mean
sigma = 2;          % stationary std
tau   = 5;          % seconds (correlation time)
dt    = 0.1;        % sample time
N     = 5000;

x = fogm_sim(N, dt, tau, mu, sigma);

% Quick checks
fprintf('Sample mean: %.3f (target %.3f)\n', mean(x), mu);
fprintf('Sample std : %.3f (target %.3f)\n', std(x,1), sigma); % use population std

% Optional: visualize
t = (0:N-1)'*dt;
plot(t, x); xlabel('time [s]'); ylabel('x');
title('FOGM realization');
```

## Notes & tips

* **Stationarity:** Using $x_0=\mu$ or $x_0\sim\mathcal N(\mu,\sigma^2)$ avoids a transient.
* **Choosing $\tau$:** Larger $\tau$ → slower drift (stronger correlation). If $\Delta t \gg \tau$, $\phi$ becomes small and the sequence approaches white noise with std $\sigma$.
* **Continuous-time form (FYI):** $\mathrm{d}x = -\frac{1}{\tau}(x-\mu)\,\mathrm{d}t + \sqrt{\tfrac{2\sigma^2}{\tau}}\,\mathrm{d}W_t$. Exact discretization over $\Delta t$ yields the same $\phi$ and $q$ above.

That’s it—set $\mu, \sigma, \tau, \Delta t$, and you’ll get a process with exactly the mean, standard deviation, and exponential autocorrelation you want.

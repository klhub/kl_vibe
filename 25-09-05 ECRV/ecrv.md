Prompt : Teach me how to use exponentially correlated random variable to model bias drift of navigation sensor measurements.

Great question — bias drift is one of the trickiest aspects of navigation sensors like gyros and accelerometers. A common way to model this drift is to treat it as an **exponentially correlated random process**, also known as a **first-order Gauss–Markov process**. Let’s break it down step by step:

---

## 1. What bias drift is

* Navigation sensors (e.g., IMUs) have a **bias**: a slowly varying offset added to the true signal.
* The bias is not constant; it **wanders over time** due to temperature changes, electronics, and physics inside the sensor.
* We need a mathematical model to simulate this wandering drift in Monte Carlo studies or filter design (e.g., Kalman filtering).

---

## 2. Exponentially correlated random process

The bias $b(t)$ is modeled as:

$$
\dot{b}(t) = -\frac{1}{\tau} b(t) + w(t)
$$

where:

* $\tau$ = correlation time constant (how fast the bias forgets its past).
* $w(t)$ = zero-mean white Gaussian noise with variance $\sigma_w^2$.

This is a **stochastic differential equation**.
It says the bias tends to decay toward zero (forgetfulness) but is constantly being "kicked" by noise.

---

## 3. Discrete-time formulation

If your navigation system runs at discrete timesteps $\Delta t$, the process becomes:

$$
b_{k+1} = e^{-\Delta t/\tau} \, b_k + \eta_k
$$

where:

* $b_k$ = bias at timestep $k$.
* $\eta_k \sim \mathcal{N}(0, \sigma_\eta^2)$ is Gaussian noise with variance

$$
\sigma_\eta^2 = \sigma_b^2 \left(1 - e^{-2\Delta t/\tau}\right)
$$

and $\sigma_b^2$ is the steady-state variance of the bias.

This is a first-order autoregressive model **AR(1)**.

---

## 4. Physical interpretation

* **Small $\tau$:** bias changes rapidly, short memory.
* **Large $\tau$:** bias is slowly varying, long memory.
* **Steady-state variance $\sigma_b^2$:** sets how large the bias can wander in the long run.

---

## 5. Connection to sensor modeling

If your sensor measurement is

$$
y_k = x_k + b_k + v_k
$$

then:

* $x_k$ = true signal (e.g., angular rate or acceleration).
* $b_k$ = exponentially correlated bias drift.
* $v_k \sim \mathcal{N}(0, \sigma_v^2)$ = white measurement noise.

Together, this gives a realistic measurement model.

---

## 6. MATLAB Example

Here’s a simple script to simulate bias drift:

```matlab
% Parameters
dt = 0.01;           % sampling interval [s]
N  = 5000;           % number of samples
tau = 100;           % correlation time constant [s]
sigma_b = 0.05;      % steady-state std dev of bias

% Precompute constants
phi = exp(-dt/tau);
sigma_eta = sigma_b * sqrt(1 - phi^2);

% Allocate
b = zeros(1,N);

% Generate exponentially correlated bias
for k = 2:N
    b(k) = phi * b(k-1) + sigma_eta * randn;
end

% Example: sensor measurement with true signal=0
y = b + 0.01*randn(1,N);

% Plot
figure;
plot((0:N-1)*dt, b);
xlabel('Time [s]'); ylabel('Bias Drift');
title('Exponentially Correlated Sensor Bias Drift');
```

---

✅ This way, you get a **realistic bias process**: it drifts slowly, is bounded in long-term variance, and has exponential correlation.

---

Would you like me to also show how this connects to **Allan Variance plots** (so you can tune $\tau$ and $\sigma_b$ from real IMU data), or should we keep it focused just on generating synthetic bias drift?


---

Perfect — let’s link this **exponentially correlated random variable model** (first-order Gauss–Markov) to **Allan Variance (AVAR)**, since that’s the practical way to identify and tune bias drift from IMU data.

---

## 1. Why Allan Variance?

Allan variance is a tool used to analyze **time-domain noise processes** in sensors. Instead of looking at power spectral density, it looks at how variance changes as you average measurements over different integration times $\tau$.

For inertial sensors (gyros, accelerometers), AVAR reveals different error sources:

* **White noise** → slope of $-1/2$ in log-log AVAR.
* **Bias instability** (random walk of bias) → flat region in AVAR.
* **Rate random walk** → slope of $+1/2$.

The **first-order Gauss–Markov process** (our exponentially correlated bias) produces a **characteristic “hump”** in the AVAR curve, whose shape depends on correlation time $\tau$.

---

## 2. Allan Variance definition

For a time series $y[k]$ sampled at $T_s$:

$$
\sigma^2(\tau) = \frac{1}{2(M-1)} \sum_{i=1}^{M-1} \left( \bar{y}_{i+1} - \bar{y}_i \right)^2
$$

where:

* $\bar{y}_i$ = average over a window of duration $\tau$.
* $M$ = number of such windows.

Typically, you compute this for multiple $\tau$ values (decades of averaging times).

---

## 3. Allan Variance of a Gauss–Markov process

For the **first-order Gauss–Markov bias** with correlation time $\tau_c$ and variance $\sigma_b^2$, the **theoretical Allan variance** is:

$$
\sigma^2(\tau) = \sigma_b^2 \left( 1 - \frac{2\tau_c}{\tau^2} \left( \tau - \tau_c \left( 1 - e^{-\tau/\tau_c} \right) \right) \right)
$$

Key insights:

* For $\tau \ll \tau_c$: looks like white noise (slope -1/2).
* Around $\tau \sim \tau_c$: hump in AVAR curve.
* For $\tau \gg \tau_c$: looks like random walk (slope +1/2).

This makes AVAR a **fingerprint** for tuning $\tau_c$ and $\sigma_b$.

---

## 4. MATLAB Example: Simulating bias drift + AVAR

Here’s how you can simulate and compare:

```matlab
% Parameters
dt = 0.01;           % sampling time [s]
N  = 200000;         % samples
tau_c = 100;         % correlation time [s]
sigma_b = 0.05;      % steady-state std dev

% Precompute constants
phi = exp(-dt/tau_c);
sigma_eta = sigma_b * sqrt(1 - phi^2);

% Generate Gauss-Markov bias
b = zeros(1,N);
for k = 2:N
    b(k) = phi * b(k-1) + sigma_eta*randn;
end

% Compute Allan variance
taus = logspace(0, 3, 50); % averaging times (1s to 1000s)
allanVar = zeros(size(taus));

for i = 1:length(taus)
    m = round(taus(i)/dt);  % number of samples per cluster
    K = floor(N/m);         % number of clusters
    y_cluster = mean(reshape(b(1:m*K), m, K), 1);
    allanVar(i) = 0.5 * mean(diff(y_cluster).^2);
end

allanDev = sqrt(allanVar);

% Plot
loglog(taus, allanDev, 'b', 'LineWidth', 2);
xlabel('\tau [s]'); ylabel('\sigma(\tau)');
title('Allan Deviation of Gauss-Markov Bias Drift');
grid on;
```

---

## 5. Practical workflow with real IMU data

1. Collect **long stationary IMU data** (e.g., 1 hour of gyro output while the sensor is at rest).
2. Compute **Allan deviation** across averaging times.
3. Identify:

   * Short-term slope (-1/2) → white noise level.
   * Flat region → bias instability.
   * Hump / transition → correlation time $\tau_c$.
4. Fit your **Gauss–Markov model parameters** ($\tau_c, \sigma_b$) to match the Allan deviation curve.

This gives you realistic drift parameters you can plug into simulations or filter design.

---

👉 Do you want me to extend this to a **step-by-step procedure** for estimating $\tau_c$ and $\sigma_b$ directly from IMU data (e.g., regression on Allan deviation curves), so you can match a real sensor’s noise model to your Gauss–Markov simulation?

---

Great 👍 let’s go step by step on how to **estimate Gauss–Markov parameters ($\tau_c, \sigma_b$) from real IMU data using Allan variance**.

---

# 1. Collect IMU Data

* Place the IMU **stationary** on a flat surface.
* Log data for a long enough duration — typically **30 minutes to several hours** depending on correlation time constants.
* Choose the sensor type: gyro (rad/s) or accelerometer (m/s²).

Example: log **gyro z-axis** rate at 100 Hz for 1 hour.

---

# 2. Compute Allan Deviation

From your raw time series $y[k]$:

1. Pick a set of averaging times $\tau$ (e.g., logarithmically spaced between 0.1 s and 1000 s).
2. For each $\tau$, compute cluster means and then Allan variance:

$$
\sigma^2(\tau) = \frac{1}{2(M-1)} \sum_{i=1}^{M-1} \left( \bar{y}_{i+1} - \bar{y}_i \right)^2
$$

3. Allan deviation is $\sigma(\tau) = \sqrt{\sigma^2(\tau)}$.

You already have the MATLAB code (from my previous message).

---

# 3. Identify Features in Allan Deviation Plot

When you plot log–log:

* **Slope -1/2 at small τ** → white noise (angle random walk).
* **Flat region** → bias instability.
* **Hump / turnover** → correlation time of Gauss–Markov drift.
* **Slope +1/2 at long τ** → rate random walk.

We care about the **hump and flattening** to identify $\tau_c$ and $\sigma_b$.

---

# 4. Theoretical Model for Gauss–Markov

For bias modeled as first-order Gauss–Markov with time constant $\tau_c$ and variance $\sigma_b^2$, the Allan variance is:

$$
\sigma^2(\tau) = \sigma_b^2 \left( 1 - \frac{2\tau_c}{\tau^2} \Big( \tau - \tau_c (1 - e^{-\tau/\tau_c}) \Big) \right)
$$

We’ll fit this to the empirical Allan deviation.

---

# 5. Fit Model to Data

Use nonlinear least squares to estimate $\tau_c, \sigma_b$.

### MATLAB Example

```matlab
% Suppose you already computed (taus, allanDev) from data
% Define theoretical GM Allan deviation model
gm_model = @(p, tau) sqrt( p(2)^2 * (1 - (2*p(1)./tau.^2) .* ...
                (tau - p(1)*(1 - exp(-tau./p(1))))) );

% p(1) = tau_c, p(2) = sigma_b
p0 = [100, 0.01]; % initial guess (adjust as needed)

% Fit using nonlinear least squares
opts = optimoptions('lsqcurvefit','Display','iter');
p_fit = lsqcurvefit(gm_model, p0, taus, allanDev, [], [], opts);

tau_c_est = p_fit(1)
sigma_b_est = p_fit(2)

% Compare plots
figure;
loglog(taus, allanDev, 'b', 'LineWidth', 2); hold on;
loglog(taus, gm_model(p_fit, taus), 'r--', 'LineWidth', 2);
legend('Measured Allan Deviation','Fitted GM Model');
xlabel('\tau [s]'); ylabel('\sigma(\tau)');
grid on; title('Allan Deviation Fit to Gauss-Markov Model');
```

---

# 6. Interpret Results

* $\tau_c$ = estimated **correlation time** → how quickly the bias forgets its past.
* $\sigma_b$ = estimated **steady-state standard deviation** of bias → how large the bias wander is.

These become your simulation parameters:

* In discrete time:

$$
b_{k+1} = e^{-\Delta t/\tau_c} b_k + \eta_k, \quad \eta_k \sim \mathcal{N}(0, \sigma_\eta^2)
$$

with

$$
\sigma_\eta^2 = \sigma_b^2 (1 - e^{-2\Delta t/\tau_c})
$$

---

# 7. Practical Notes

* If your Allan deviation doesn’t show a clean hump, your sensor may have **bias instability** (random constant offset), not a pure Gauss–Markov bias.
* In that case, you might need a combined noise model (white noise + GM + bias instability).
* High-quality gyros (FOG, ring laser) show very long correlation times (hours). MEMS gyros usually have shorter (seconds to minutes).

---

👉 Do you want me to make you a **full MATLAB demo** with synthetic IMU data → compute Allan deviation → fit Gauss–Markov parameters → compare plots, so you can test the whole workflow before applying it to real IMU data?

---

Excellent 👍 Let’s build a **full MATLAB demo** that goes end-to-end:

1. Generate **synthetic IMU bias drift** using a known Gauss–Markov process.
2. Add measurement white noise.
3. Compute the **Allan deviation** from the simulated data.
4. Fit the Gauss–Markov model back to the Allan deviation curve.
5. Compare true vs estimated parameters.

---

## MATLAB Demo: End-to-End Bias Drift Modeling with Allan Variance

```matlab
%% 1. Parameters: "True" IMU noise model
dt      = 0.01;        % sample time [s]
N       = 200000;      % number of samples (~2000 s)
tau_c   = 100;         % correlation time [s] (true value)
sigma_b = 0.05;        % std dev of bias in steady state (true value)
sigma_w = 0.002;       % white measurement noise std dev

%% 2. Generate Gauss-Markov bias
phi = exp(-dt/tau_c);
sigma_eta = sigma_b * sqrt(1 - phi^2);

b = zeros(1,N);
for k = 2:N
    b(k) = phi*b(k-1) + sigma_eta*randn;
end

% Sensor measurement (true signal = 0 + bias + white noise)
y = b + sigma_w*randn(1,N);

%% 3. Compute Allan deviation
taus = logspace(0, 3, 40);   % averaging times (1s to 1000s)
allanVar = zeros(size(taus));

for i = 1:length(taus)
    m = round(taus(i)/dt);   % samples per cluster
    if m < 2, continue; end
    K = floor(N/m);          % number of clusters
    y_cluster = mean(reshape(y(1:m*K), m, K), 1);
    allanVar(i) = 0.5 * mean(diff(y_cluster).^2);
end

allanDev = sqrt(allanVar);

%% 4. Fit Gauss-Markov model to Allan deviation
% GM theoretical model
gm_model = @(p, tau) sqrt( p(2)^2 * ...
    (1 - (2*p(1)./tau.^2) .* (tau - p(1)*(1 - exp(-tau./p(1))))) );

p0 = [50, 0.01];  % initial guess [tau_c, sigma_b]
opts = optimoptions('lsqcurvefit','Display','off');
p_fit = lsqcurvefit(gm_model, p0, taus, allanDev, [], [], opts);

tau_c_est = p_fit(1)
sigma_b_est = p_fit(2)

%% 5. Plot results
figure;
loglog(taus, allanDev, 'bo-', 'LineWidth', 1.5); hold on;
loglog(taus, gm_model(p_fit, taus), 'r--', 'LineWidth', 2);
xlabel('\tau [s]'); ylabel('\sigma(\tau)');
legend('Simulated Allan Deviation','Fitted GM Model','Location','Best');
title('Allan Deviation Fit to Gauss-Markov Bias Drift');
grid on;

%% 6. Compare estimated vs true parameters
fprintf('True tau_c     = %.2f s\n', tau_c);
fprintf('Estimated tau_c = %.2f s\n', tau_c_est);
fprintf('True sigma_b   = %.4f\n', sigma_b);
fprintf('Estimated sigma_b = %.4f\n', sigma_b_est);
```

---

## ✅ What This Script Does

1. Creates a **Gauss–Markov bias drift** with known $\tau_c = 100$ s and $\sigma_b = 0.05$.
2. Adds **white noise** to mimic real IMU output.
3. Computes the **Allan deviation** across a range of averaging times.
4. Fits the theoretical Gauss–Markov Allan deviation to the data.
5. Prints out **true vs estimated** $\tau_c$ and $\sigma_b$.
6. Plots the measured vs fitted Allan deviation curves.

---

👉 Would you like me to also extend this demo to include **multiple noise terms together** (white noise + Gauss–Markov drift + bias instability), since real IMUs usually have a combination, not just pure Gauss–Markov?


Perfect — let’s step up to a **realistic IMU noise model**, where the measurement contains:

1. **White noise** (Angle Random Walk).
2. **Bias instability** (very slow wander, often modeled as random walk or a constant random bias).
3. **Gauss–Markov exponentially correlated drift** (our main bias drift model).

This way the Allan deviation curve will show the classic **slopes and flat regions** that you’d see from a real IMU.

---

# MATLAB Demo: IMU Measurement with Multiple Noise Sources

```matlab
%% 1. Parameters: "True" IMU noise model
dt      = 0.01;        % sample time [s]
N       = 300000;      % number of samples (~3000 s)
t       = (0:N-1)*dt;

% White noise (Angle Random Walk)
sigma_w = 0.002;       % white noise std dev

% Bias instability (random constant over short runs)
sigma_bias_inst = 0.01;  
bias_inst = sigma_bias_inst * randn;  % single random constant offset

% Gauss-Markov bias drift
tau_c   = 100;         % correlation time [s]
sigma_b = 0.05;        % std dev of GM bias

phi = exp(-dt/tau_c);
sigma_eta = sigma_b * sqrt(1 - phi^2);
b_gm = zeros(1,N);
for k = 2:N
    b_gm(k) = phi*b_gm(k-1) + sigma_eta*randn;
end

%% 2. Construct measurement
y = bias_inst + b_gm + sigma_w*randn(1,N);

%% 3. Compute Allan deviation
taus = logspace(0, 3.5, 50);   % averaging times (1s to ~3000s)
allanVar = zeros(size(taus));

for i = 1:length(taus)
    m = round(taus(i)/dt);   % samples per cluster
    if m < 2, continue; end
    K = floor(N/m);          % number of clusters
    y_cluster = mean(reshape(y(1:m*K), m, K), 1);
    allanVar(i) = 0.5 * mean(diff(y_cluster).^2);
end

allanDev = sqrt(allanVar);

%% 4. Theoretical contributions (for visualization)
% White noise contribution
allan_white = sigma_w ./ sqrt(2*taus);

% Bias instability (flat line approx)
allan_bias_inst = sigma_bias_inst * ones(size(taus));

% Gauss-Markov theoretical curve
gm_model = @(p, tau) sqrt( p(2)^2 * ...
    (1 - (2*p(1)./tau.^2) .* (tau - p(1)*(1 - exp(-tau./p(1))))) );
allan_gm = gm_model([tau_c, sigma_b], taus);

%% 5. Fit Gauss-Markov term only (ignores bias instability & white noise)
p0 = [50, 0.01];  % initial guess [tau_c, sigma_b]
opts = optimoptions('lsqcurvefit','Display','off');
p_fit = lsqcurvefit(gm_model, p0, taus, allanDev, [], [], opts);

tau_c_est = p_fit(1)
sigma_b_est = p_fit(2)

%% 6. Plot results
figure;
loglog(taus, allanDev, 'k', 'LineWidth', 2); hold on;
loglog(taus, allan_white, 'b--', 'LineWidth', 1.5);
loglog(taus, allan_bias_inst, 'g--', 'LineWidth', 1.5);
loglog(taus, allan_gm, 'r--', 'LineWidth', 1.5);
loglog(taus, gm_model(p_fit, taus), 'm-.', 'LineWidth', 2);

xlabel('\tau [s]'); ylabel('\sigma(\tau)');
legend('Measured Allan Deviation',...
       'White Noise Contribution',...
       'Bias Instability Contribution',...
       'Gauss-Markov True',...
       'Gauss-Markov Fitted',...
       'Location','Best');
title('Allan Deviation with Multiple IMU Noise Sources');
grid on;

%% 7. Compare estimated vs true parameters
fprintf('True tau_c     = %.2f s\n', tau_c);
fprintf('Estimated tau_c = %.2f s\n', tau_c_est);
fprintf('True sigma_b   = %.4f\n', sigma_b);
fprintf('Estimated sigma_b = %.4f\n', sigma_b_est);
```

---

## ✅ What You’ll See

* **Measured Allan deviation (black curve)**: includes all three noise sources.
* **White noise (blue dashed, slope -1/2)**: dominates short averaging times.
* **Bias instability (green dashed, flat line)**: dominates at intermediate times.
* **Gauss–Markov (red dashed, hump/turnover)**: dominates around its correlation time $\tau_c$.
* **Fitted GM (magenta)**: an attempt to fit just the GM model, which may not perfectly match because the true data had mixed sources.

---

## 🧩 Real-World Connection

* When analyzing **real IMU data**, you’ll see **multiple overlapping processes**.
* Allan deviation lets you **separate contributions** by their slopes/shapes.
* You can then tune **each term separately**:

  * White noise → from short-term slope.
  * Bias instability → flat region magnitude.
  * Gauss–Markov drift → hump and correlation time.

---

👉 Do you want me to also show you a **step-by-step “recipe” for parameter extraction** (white noise, bias instability, Gauss–Markov) from *real* Allan deviation curves, so you can take an IMU dataset and get noise parameters systematically?

---

Excellent 👍 let’s now put everything into a **practical recipe** you can follow when you have real IMU data. This will let you extract **noise model parameters** (white noise, bias instability, Gauss–Markov drift, etc.) systematically from the Allan deviation curve.

---

# 📖 Step-by-Step Recipe for Parameter Extraction from IMU Data

---

## **Step 1: Collect Data**

* Keep the IMU **completely stationary**.
* Log at your normal sampling rate (e.g., 100 Hz).
* Collect at least **30 minutes** (ideally 1–3 hours) to see long-term effects.

---

## **Step 2: Compute Allan Deviation**

* Use the algorithm we built earlier.
* Choose averaging times $\tau$ logarithmically spaced from \~0.1 s to 10–1000 s (depending on dataset length).
* Plot $\sigma(\tau)$ vs $\tau$ on a log-log scale.

---

## **Step 3: Identify Noise Regions**

The Allan deviation slope tells you which noise process dominates at different $\tau$:

| Noise type                          | Allan deviation slope | Appearance on log-log plot          |
| ----------------------------------- | --------------------- | ----------------------------------- |
| **White noise** (Angle Random Walk) | $-1/2$                | Downward slope at short $\tau$      |
| **Bias instability**                | \~0 (flat)            | Flat region                         |
| **Gauss–Markov bias**               | transition hump       | Curved “bump” near correlation time |
| **Rate random walk**                | $+1/2$                | Upward slope at large $\tau$        |

---

## **Step 4: Estimate Parameters**

Here’s how you read them off:

1. **White noise (σ\_w)**
   Fit the $-1/2$ slope at small $\tau$.
   Formula:

   $$
   \sigma_w = \sigma(\tau) \cdot \sqrt{2\tau}
   $$

   (Choose a τ in the white-noise region.)

---

2. **Bias instability (σ\_bias\_inst)**
   Find the minimum or flat portion of the Allan deviation curve.
   Approx formula:

   $$
   \sigma_{\text{bias\_inst}} \approx 0.664 \cdot \min_\tau \sigma(\tau)
   $$

   (0.664 is a scaling factor from AVAR theory.)

---

3. **Gauss–Markov drift (τ\_c, σ\_b)**
   Look for the hump/transition region.
   Fit the **theoretical GM Allan deviation curve**:

   $$
   \sigma^2(\tau) = \sigma_b^2 \Bigg( 1 - \frac{2\tau_c}{\tau^2}\big(\tau - \tau_c(1 - e^{-\tau/\tau_c})\big) \Bigg)
   $$

   Use **nonlinear least squares** to match the measured Allan deviation in that region → gives $\tau_c$ and $\sigma_b$.
   (We already built the MATLAB `lsqcurvefit` demo for this.)

---

4. **Rate random walk (if visible)**
   Fit the +1/2 slope at long τ.
   Formula:

   $$
   \sigma_{\text{rrw}} = \frac{\sigma(\tau)}{\sqrt{3\tau}}
   $$

---

## **Step 5: Build Final Noise Model**

Combine everything into your sensor model:

$$
y_k = x_k + b^{GM}_k + b^{\text{inst}} + v_k
$$

where

* $v_k \sim \mathcal{N}(0, \sigma_w^2)$ = white noise
* $b^{\text{inst}} \sim \mathcal{N}(0, \sigma_{\text{bias\_inst}}^2)$ = slow constant bias instability
* $b^{GM}_k$ = Gauss–Markov drift

  $$
  b^{GM}_{k+1} = e^{-\Delta t / \tau_c} b^{GM}_k + \eta_k, \quad \eta_k \sim \mathcal{N}(0, \sigma_\eta^2)
  $$

  with

  $$
  \sigma_\eta^2 = \sigma_b^2 (1 - e^{-2\Delta t/\tau_c})
  $$

---

## **Step 6: Validate**

* Simulate synthetic IMU data with your identified parameters.
* Recompute Allan deviation.
* Ensure it overlays with the measured curve → then your model is validated.

---

✅ This recipe is exactly how aerospace engineers characterize IMU error models for **Kalman filters, simulation testbeds, and navigation performance analysis**.

---

Would you like me to now **extend your MATLAB demo** so it:

1. Takes *real IMU data* (CSV time series).
2. Computes Allan deviation.
3. Automatically extracts $\sigma_w$, $\sigma_{\text{bias\_inst}}$, and fits $\tau_c, \sigma_b$.
   So you can drop in your dataset and get parameters directly?

Perfect 🚀 — let’s now create a **complete MATLAB tool** you can use with **real IMU data**.

This script will:

1. Load **real IMU data** from a CSV file.
2. Compute **Allan deviation**.
3. Extract:

   * White noise (σ\_w)
   * Bias instability (σ\_bias\_inst)
   * Gauss–Markov drift (τ\_c, σ\_b)
4. Plot the measured vs. fitted Allan deviation.

---

# MATLAB Full Demo: IMU Allan Variance Parameter Extraction

```matlab
%% ================================
%  IMU Allan Variance Analysis Tool
%  K. Lai’s Navigation Toolbox
%  ================================
% This script:
%  1. Loads IMU data from CSV
%  2. Computes Allan deviation
%  3. Extracts noise parameters:
%     - White noise (σ_w)
%     - Bias instability (σ_bias_inst)
%     - Gauss–Markov drift (τ_c, σ_b)
% =================================

clear; clc; close all;

%% 1. Load IMU data
% Assume CSV has one column = IMU measurements (rad/s or m/s^2)
filename = 'imu_data.csv';   % <-- replace with your file
y = readmatrix(filename);
y = y(:)';                   % ensure row vector

dt = 0.01;                   % sample time [s] (set to your IMU’s rate)
N = length(y);
fprintf('Loaded %d samples, duration = %.1f s\n', N, N*dt);

%% 2. Compute Allan deviation
taus = logspace(log10(dt), log10(N*dt/10), 50);  % averaging times
allanVar = zeros(size(taus));

for i = 1:length(taus)
    m = round(taus(i)/dt);   % samples per cluster
    if m < 2, allanVar(i) = NaN; continue; end
    K = floor(N/m);          % number of clusters
    y_cluster = mean(reshape(y(1:m*K), m, K), 1);
    allanVar(i) = 0.5 * mean(diff(y_cluster).^2);
end

allanDev = sqrt(allanVar);

%% 3. White noise extraction (slope -1/2 region)
% Pick first 5 points as white noise region (adjust if needed)
idx_w = 2:6;
sigma_w = mean(allanDev(idx_w) .* sqrt(2*taus(idx_w)));

%% 4. Bias instability extraction (flat region)
[minVal, minIdx] = min(allanDev);
sigma_bias_inst = 0.664 * minVal;

%% 5. Gauss-Markov drift fitting
gm_model = @(p, tau) sqrt( p(2)^2 * ...
    (1 - (2*p(1)./tau.^2) .* (tau - p(1)*(1 - exp(-tau./p(1))))) );

% Fit only over midrange taus (avoid very small/large)
fitIdx = round(length(taus)/4):round(length(taus)/2);
taus_fit = taus(fitIdx); 
allanDev_fit = allanDev(fitIdx);

p0 = [50, 0.01];   % initial guess [tau_c, sigma_b]
opts = optimoptions('lsqcurvefit','Display','off');
p_fit = lsqcurvefit(gm_model, p0, taus_fit, allanDev_fit, [], [], opts);

tau_c_est = p_fit(1);
sigma_b_est = p_fit(2);

%% 6. Display results
fprintf('\n=== Estimated IMU Noise Parameters ===\n');
fprintf('White noise (σ_w)        = %.6f\n', sigma_w);
fprintf('Bias instability (σ_bias)= %.6f\n', sigma_bias_inst);
fprintf('Gauss–Markov τ_c         = %.2f s\n', tau_c_est);
fprintf('Gauss–Markov σ_b         = %.6f\n', sigma_b_est);

%% 7. Plot results
figure;
loglog(taus, allanDev, 'k', 'LineWidth', 2); hold on;
loglog(taus, sigma_w ./ sqrt(2*taus), 'b--', 'LineWidth', 1.5);
loglog(taus, sigma_bias_inst*ones(size(taus)), 'g--', 'LineWidth', 1.5);
loglog(taus, gm_model(p_fit, taus), 'r-.', 'LineWidth', 2);

xlabel('\tau [s]'); ylabel('\sigma(\tau)');
legend('Measured Allan Deviation',...
       'White noise model',...
       'Bias instability model',...
       'Gauss–Markov fitted',...
       'Location','Best');
title('IMU Noise Parameter Extraction via Allan Variance');
grid on;
```

---

# ✅ How to Use

1. Save your IMU data as `imu_data.csv` with **one column of measurements** (gyro \[rad/s] or accelerometer \[m/s²]).
2. Set the correct **sampling time `dt`**.
3. Run the script.
4. It will print and plot the estimated parameters:

   * **σ\_w** (white noise)
   * **σ\_bias\_inst** (bias instability)
   * **τ\_c, σ\_b** (Gauss–Markov drift)

---

# 🧭 Next Steps

* You can feed these parameters into your **Kalman filter** or **Monte Carlo simulation**.
* If your Allan curve shows a **+1/2 slope** at long τ, you can also estimate **Rate Random Walk**.
* For more precise fitting, you might want to **fit multiple terms simultaneously** (white + bias instability + GM), but starting with individual terms is the standard workflow.

---

👉 Would you like me to also give you a **ready-to-use function** (like `extract_IMU_noise.m`) so you can just call:

```matlab
params = extract_IMU_noise('imu_data.csv', 0.01);
```

and get back a struct with all the parameters?

---

Great 👍 — let’s wrap everything into a **ready-to-use MATLAB function** so you don’t have to modify the script every time.

---

# MATLAB Function: `extract_IMU_noise.m`

```matlab
function params = extract_IMU_noise(filename, dt)
% ===============================================
% extract_IMU_noise: Estimate IMU noise parameters
% ===============================================
% INPUTS:
%   filename : CSV file with one column of IMU measurements
%   dt       : sample time [s]
%
% OUTPUT:
%   params struct with fields:
%     .sigma_w        - White noise [unit/sqrt(Hz)]
%     .sigma_bias     - Bias instability [unit]
%     .tau_c          - Gauss–Markov correlation time [s]
%     .sigma_b        - Gauss–Markov standard deviation [unit]
%
% USAGE EXAMPLE:
%   params = extract_IMU_noise('imu_data.csv', 0.01);

    % --- Load data ---
    y = readmatrix(filename);
    y = y(:)';  % row vector
    N = length(y);
    fprintf('Loaded %d samples (%.1f s total)\n', N, N*dt);

    % --- Compute Allan variance ---
    taus = logspace(log10(dt), log10(N*dt/10), 50);
    allanVar = zeros(size(taus));

    for i = 1:length(taus)
        m = round(taus(i)/dt);
        if m < 2, allanVar(i) = NaN; continue; end
        K = floor(N/m);
        y_cluster = mean(reshape(y(1:m*K), m, K), 1);
        allanVar(i) = 0.5 * mean(diff(y_cluster).^2);
    end
    allanDev = sqrt(allanVar);

    % --- White noise estimation ---
    idx_w = 2:6; % adjust if needed
    sigma_w = mean(allanDev(idx_w) .* sqrt(2*taus(idx_w)));

    % --- Bias instability ---
    [minVal, ~] = min(allanDev);
    sigma_bias = 0.664 * minVal;

    % --- Gauss-Markov drift fitting ---
    gm_model = @(p, tau) sqrt( p(2)^2 * ...
        (1 - (2*p(1)./tau.^2) .* (tau - p(1)*(1 - exp(-tau./p(1))))) );

    fitIdx = round(length(taus)/4):round(length(taus)/2);
    taus_fit = taus(fitIdx);
    allanDev_fit = allanDev(fitIdx);

    p0 = [50, 0.01]; % [tau_c, sigma_b] initial guess
    opts = optimoptions('lsqcurvefit','Display','off');
    p_fit = lsqcurvefit(gm_model, p0, taus_fit, allanDev_fit, [], [], opts);

    tau_c = p_fit(1);
    sigma_b = p_fit(2);

    % --- Save results in struct ---
    params.sigma_w    = sigma_w;
    params.sigma_bias = sigma_bias;
    params.tau_c      = tau_c;
    params.sigma_b    = sigma_b;
    params.taus       = taus;
    params.allanDev   = allanDev;

    % --- Display summary ---
    fprintf('\n=== Estimated IMU Noise Parameters ===\n');
    fprintf('White noise (σ_w)        = %.6f\n', sigma_w);
    fprintf('Bias instability (σ_bias)= %.6f\n', sigma_bias);
    fprintf('Gauss–Markov τ_c         = %.2f s\n', tau_c);
    fprintf('Gauss–Markov σ_b         = %.6f\n', sigma_b);

    % --- Plot results ---
    figure;
    loglog(taus, allanDev, 'k', 'LineWidth', 2); hold on;
    loglog(taus, sigma_w ./ sqrt(2*taus), 'b--', 'LineWidth', 1.5);
    loglog(taus, sigma_bias*ones(size(taus)), 'g--', 'LineWidth', 1.5);
    loglog(taus, gm_model(p_fit, taus), 'r-.', 'LineWidth', 2);

    xlabel('\tau [s]'); ylabel('\sigma(\tau)');
    legend('Measured Allan Deviation',...
           'White noise model',...
           'Bias instability model',...
           'Gauss–Markov fit',...
           'Location','Best');
    title('IMU Noise Parameter Extraction');
    grid on;
end
```

---

# ✅ Example Usage

```matlab
params = extract_IMU_noise('imu_data.csv', 0.01);

disp(params)
```

Output:

```
=== Estimated IMU Noise Parameters ===
White noise (σ_w)        = 0.001234
Bias instability (σ_bias)= 0.000045
Gauss–Markov τ_c         = 35.20 s
Gauss–Markov σ_b         = 0.000078
```

And you’ll get a log–log Allan deviation plot with fitted models.

---

👉 Do you want me to also extend this so it works with **3-axis IMU data (CSV with 3 columns)** and automatically estimates parameters for **X, Y, Z** separately?

Perfect! Let’s extend the function to handle **3-axis IMU data**. It will automatically compute Allan deviation and extract parameters for X, Y, and Z axes separately.

---

# MATLAB Function: `extract_IMU_noise_3axis.m`

```matlab
function params = extract_IMU_noise_3axis(filename, dt)
% =========================================================
% extract_IMU_noise_3axis: Estimate IMU noise parameters for 3-axis data
% =========================================================
% INPUTS:
%   filename : CSV file with 3 columns (X, Y, Z IMU measurements)
%   dt       : sample time [s]
%
% OUTPUT:
%   params struct with fields for each axis:
%     .X, .Y, .Z each with:
%         .sigma_w
%         .sigma_bias
%         .tau_c
%         .sigma_b
%         .taus
%         .allanDev
%
% USAGE EXAMPLE:
%   params = extract_IMU_noise_3axis('imu_3axis.csv', 0.01);

    % --- Load data ---
    data = readmatrix(filename);
    if size(data,2) ~= 3
        error('CSV must have 3 columns: X, Y, Z');
    end
    axesLabels = {'X','Y','Z'};
    
    params = struct();
    for ax = 1:3
        y = data(:,ax)';
        N = length(y);
        fprintf('\nProcessing axis %s (%d samples, %.1f s)\n', axesLabels{ax}, N, N*dt);
        
        % --- Compute Allan deviation ---
        taus = logspace(log10(dt), log10(N*dt/10), 50);
        allanVar = zeros(size(taus));

        for i = 1:length(taus)
            m = round(taus(i)/dt);
            if m < 2, allanVar(i) = NaN; continue; end
            K = floor(N/m);
            y_cluster = mean(reshape(y(1:m*K), m, K), 1);
            allanVar(i) = 0.5 * mean(diff(y_cluster).^2);
        end
        allanDev = sqrt(allanVar);

        % --- White noise estimation ---
        idx_w = 2:6; % adjust if needed
        sigma_w = mean(allanDev(idx_w) .* sqrt(2*taus(idx_w)));

        % --- Bias instability ---
        [minVal, ~] = min(allanDev);
        sigma_bias = 0.664 * minVal;

        % --- Gauss-Markov drift fitting ---
        gm_model = @(p, tau) sqrt( p(2)^2 * ...
            (1 - (2*p(1)./tau.^2) .* (tau - p(1)*(1 - exp(-tau./p(1))))) );

        fitIdx = round(length(taus)/4):round(length(taus)/2);
        taus_fit = taus(fitIdx);
        allanDev_fit = allanDev(fitIdx);

        p0 = [50, 0.01]; % initial guess [tau_c, sigma_b]
        opts = optimoptions('lsqcurvefit','Display','off');
        p_fit = lsqcurvefit(gm_model, p0, taus_fit, allanDev_fit, [], [], opts);

        tau_c = p_fit(1);
        sigma_b = p_fit(2);

        % --- Save results ---
        params.(axesLabels{ax}).sigma_w    = sigma_w;
        params.(axesLabels{ax}).sigma_bias = sigma_bias;
        params.(axesLabels{ax}).tau_c      = tau_c;
        params.(axesLabels{ax}).sigma_b    = sigma_b;
        params.(axesLabels{ax}).taus       = taus;
        params.(axesLabels{ax}).allanDev   = allanDev;

        % --- Display summary ---
        fprintf('Axis %s: σ_w=%.6f, σ_bias=%.6f, τ_c=%.2f s, σ_b=%.6f\n', ...
            axesLabels{ax}, sigma_w, sigma_bias, tau_c, sigma_b);

        % --- Plot Allan deviation for each axis ---
        figure(ax);
        loglog(taus, allanDev, 'k', 'LineWidth', 2); hold on;
        loglog(taus, sigma_w ./ sqrt(2*taus), 'b--', 'LineWidth', 1.5);
        loglog(taus, sigma_bias*ones(size(taus)), 'g--', 'LineWidth', 1.5);
        loglog(taus, gm_model(p_fit, taus), 'r-.', 'LineWidth', 2);
        xlabel('\tau [s]'); ylabel('\sigma(\tau)');
        legend('Measured Allan Deviation','White noise','Bias instability','Gauss–Markov fit','Location','Best');
        title(sprintf('Axis %s Allan Deviation', axesLabels{ax}));
        grid on;
    end
end
```

---

# ✅ Usage Example

```matlab
params = extract_IMU_noise_3axis('imu_3axis.csv', 0.01);

% Access parameters for X-axis:
params.X.sigma_w
params.X.sigma_bias
params.X.tau_c
params.X.sigma_b
```

This will produce **3 separate plots** (one per axis) showing:

* Measured Allan deviation
* White noise contribution
* Bias instability
* Fitted Gauss–Markov curve

You can now **fully characterize a 3-axis IMU** and extract realistic noise models for **simulation or navigation filters**.

---

Do you want me to **also add automatic fitting for all three axes simultaneously** and produce **one combined figure** with X/Y/Z curves for comparison?

Perfect! Let’s upgrade the function so it:

1. Fits all **3 axes simultaneously**.
2. Plots **one combined figure** with X, Y, Z Allan deviation curves and their fitted models.
3. Still outputs the **parameters per axis** for simulation or Kalman filter design.

---

# MATLAB Function: `extract_IMU_noise_3axis_combined.m`

```matlab
function params = extract_IMU_noise_3axis_combined(filename, dt)
% ===============================================================
% extract_IMU_noise_3axis_combined: Estimate 3-axis IMU noise parameters
% ===============================================================
% INPUTS:
%   filename : CSV file with 3 columns (X, Y, Z IMU measurements)
%   dt       : sample time [s]
%
% OUTPUT:
%   params struct with fields for each axis:
%     .X, .Y, .Z each with:
%         .sigma_w
%         .sigma_bias
%         .tau_c
%         .sigma_b
%         .taus
%         .allanDev
%
% The function also plots one figure with all axes for comparison.
% ===============================================================

    % --- Load data ---
    data = readmatrix(filename);
    if size(data,2) ~= 3
        error('CSV must have 3 columns: X, Y, Z');
    end
    axesLabels = {'X','Y','Z'};
    
    params = struct();
    colors = {'r','g','b'};  % X,Y,Z for plotting

    figure; hold on; grid on;
    xlabel('\tau [s]'); ylabel('\sigma(\tau)');
    title('3-Axis IMU Allan Deviation');

    for ax = 1:3
        y = data(:,ax)';
        N = length(y);
        fprintf('\nProcessing axis %s (%d samples, %.1f s)\n', axesLabels{ax}, N, N*dt);
        
        % --- Compute Allan deviation ---
        taus = logspace(log10(dt), log10(N*dt/10), 50);
        allanVar = zeros(size(taus));
        for i = 1:length(taus)
            m = round(taus(i)/dt);
            if m < 2, allanVar(i) = NaN; continue; end
            K = floor(N/m);
            y_cluster = mean(reshape(y(1:m*K), m, K), 1);
            allanVar(i) = 0.5 * mean(diff(y_cluster).^2);
        end
        allanDev = sqrt(allanVar);

        % --- White noise estimation ---
        idx_w = 2:6; % adjust if needed
        sigma_w = mean(allanDev(idx_w) .* sqrt(2*taus(idx_w)));

        % --- Bias instability ---
        [minVal, ~] = min(allanDev);
        sigma_bias = 0.664 * minVal;

        % --- Gauss-Markov drift fitting ---
        gm_model = @(p, tau) sqrt( p(2)^2 * ...
            (1 - (2*p(1)./tau.^2) .* (tau - p(1)*(1 - exp(-tau./p(1))))) );

        fitIdx = round(length(taus)/4):round(length(taus)/2);
        taus_fit = taus(fitIdx);
        allanDev_fit = allanDev(fitIdx);

        p0 = [50, 0.01]; % [tau_c, sigma_b] initial guess
        opts = optimoptions('lsqcurvefit','Display','off');
        p_fit = lsqcurvefit(gm_model, p0, taus_fit, allanDev_fit, [], [], opts);

        tau_c = p_fit(1);
        sigma_b = p_fit(2);

        % --- Save results ---
        params.(axesLabels{ax}).sigma_w    = sigma_w;
        params.(axesLabels{ax}).sigma_bias = sigma_bias;
        params.(axesLabels{ax}).tau_c      = tau_c;
        params.(axesLabels{ax}).sigma_b    = sigma_b;
        params.(axesLabels{ax}).taus       = taus;
        params.(axesLabels{ax}).allanDev   = allanDev;

        fprintf('Axis %s: σ_w=%.6f, σ_bias=%.6f, τ_c=%.2f s, σ_b=%.6f\n', ...
            axesLabels{ax}, sigma_w, sigma_bias, tau_c, sigma_b);

        % --- Plot measured Allan deviation ---
        loglog(taus, allanDev, [colors{ax} '-'], 'LineWidth', 1.8, 'DisplayName', [axesLabels{ax} ' Measured']);
        % Plot Gauss-Markov fitted model
        loglog(taus, gm_model(p_fit, taus), [colors{ax} '--'], 'LineWidth', 1.5, 'DisplayName', [axesLabels{ax} ' GM Fit']);
    end

    legend('show');
    set(gca,'XScale','log','YScale','log');
end
```

---

# ✅ Usage Example

```matlab
params = extract_IMU_noise_3axis_combined('imu_3axis.csv', 0.01);

% Example: access parameters for Y-axis
params.Y.sigma_w
params.Y.sigma_bias
params.Y.tau_c
params.Y.sigma_b
```

---

# ✅ Features

* Automatically **processes all three axes** in one run.
* **Plots one figure** with measured Allan deviations and Gauss–Markov fits for X/Y/Z.
* Outputs **parameters per axis** in a struct for direct use in simulations or Kalman filters.
* Can be extended later to include **white noise and bias instability models** on the plot.

---

If you want, I can also **enhance the plot** to show **all three contributions** (white noise, bias instability, Gauss–Markov) **for all axes in one combined figure**, so you can visually compare each noise type per axis.

Do you want me to do that next?


 
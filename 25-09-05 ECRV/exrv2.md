Prompt : Teach me how to use exponentially correlated random variable to model bias drift of navigation sensor measurements.

Absolutely! Let’s go **step by step**. I’ll focus on navigation sensors (gyros, accelerometers) and show you how **exponentially correlated random processes** are used to model **bias drift**, both conceptually and mathematically.

---

### **1. What is bias drift in sensors?**

Bias drift is the slow change in the sensor’s zero-level output over time. For example:

* A gyro might output 0.01°/s when stationary.
* Over time, this bias is not constant—it drifts randomly.
* If we ignore this drift, integrating measurements (for attitude, position) accumulates errors.

**Key idea:** The drift is **random**, but not completely independent at each time step—it has **memory**.

---

### **2. Exponentially correlated (first-order Gauss-Markov) process**

Bias drift is often modeled as a **first-order Gauss-Markov process**, which is a type of **exponentially correlated random process**.

**Definition:**

Let $b(t)$ be the bias. Then its dynamics are:

$$
\dot{b}(t) = -\frac{1}{\tau} b(t) + w(t)
$$

Where:

* $\tau$ = correlation time (seconds)
* $w(t)$ = zero-mean white Gaussian noise with PSD $q$
* $b(t)$ = exponentially correlated random variable

**Properties:**

1. **Correlation function:**

$$
\text{E}[b(t)b(t+\Delta t)] = \sigma_b^2 e^{-|\Delta t|/\tau}
$$

2. **Steady-state variance:**

$$
\sigma_b^2 = \frac{q \tau}{2}
$$

* Large $\tau$ → slowly drifting bias
* Small $\tau$ → quickly changing bias, almost white noise

---

### **3. Discrete-time implementation**

In simulations or Kalman filters, we usually discretize with a timestep $\Delta t$:

$$
b_{k+1} = e^{-\Delta t / \tau} b_k + \eta_k
$$

Where $\eta_k$ is Gaussian noise:

$$
\eta_k \sim \mathcal{N}\left(0, \sigma_\eta^2\right), \quad 
\sigma_\eta^2 = \sigma_b^2 \left(1 - e^{-2 \Delta t / \tau}\right)
$$

* This ensures the discrete-time bias has the same exponential correlation as the continuous-time version.
* Very useful in **simulating sensor data** or **modeling in Kalman filters**.

---

### **4. Practical steps for simulating sensor bias drift**

1. **Choose parameters:**

   * Correlation time $\tau$ (seconds), based on sensor datasheet
   * Steady-state standard deviation $\sigma_b$

2. **Initialize bias:**

$$
b_0 = 0 \quad \text{or drawn from } \mathcal{N}(0, \sigma_b^2)
$$

3. **Propagate in discrete steps:**

$$
b_{k+1} = e^{-\Delta t / \tau} b_k + \sigma_b \sqrt{1 - e^{-2 \Delta t / \tau}} \cdot \text{randn}()
$$

4. **Add bias to sensor measurement:**

$$
y_k = y_{\text{true},k} + b_k + v_k
$$

* $v_k$ = white measurement noise

---

### **5. MATLAB example**

```matlab
dt = 0.01;       % time step [s]
T = 100;         % total simulation time [s]
tau = 50;        % correlation time [s]
sigma_b = 0.05;  % steady-state bias [unit of sensor]

N = T/dt;
b = zeros(N,1);
b(1) = sigma_b*randn;

for k = 1:N-1
    phi = exp(-dt/tau);
    sigma_eta = sigma_b * sqrt(1 - phi^2);
    b(k+1) = phi * b(k) + sigma_eta * randn;
end

plot((0:N-1)*dt, b)
xlabel('Time [s]')
ylabel('Bias drift')
title('Exponentially correlated sensor bias drift')
```

* This generates a realistic bias drift that slowly evolves over time.
* You can use it for **gyro/accelerometer simulation** or **Kalman filter testing**.

---

### **6. Key points**

* Exponentially correlated bias captures **realistic memory effects** in sensors.
* $\tau$ determines **how fast the bias changes**.
* $\sigma_b$ determines **bias amplitude**.
* Discrete-time propagation uses the **phi + scaled white noise** method.
* Works directly in **sensor simulations** or **state estimation models**.

---

If you want, I can also **show how to include this bias model directly in a discrete-time Kalman filter for navigation**. It’s exactly what GNSS/INS integration often does.

Do you want me to go that far?

---

Prompt : What's the steady state (in discrete form) noise statistics of this ECRV process?

Perfect, let’s carefully work through this. You’re asking for the **steady-state noise statistics** of a **discrete-time exponentially correlated random variable (ECRV)**.

---

### **1. Discrete-time ECRV equation**

For a timestep $\Delta t$:

$$
b_{k+1} = \phi \, b_k + \eta_k
$$

Where:

* $\phi = e^{-\Delta t / \tau}$ (discrete-time correlation coefficient)
* $\eta_k \sim \mathcal{N}(0, \sigma_\eta^2)$ is independent white noise

We want **steady-state variance of $b_k$**.

---

### **2. Propagate variance**

Let $\sigma_b^2 = \text{Var}[b_k]$ at steady state.

$$
\text{Var}[b_{k+1}] = \text{Var}[\phi b_k + \eta_k]
= \phi^2 \text{Var}[b_k] + \text{Var}[\eta_k]
$$

At **steady state**, $\text{Var}[b_{k+1}] = \text{Var}[b_k] = \sigma_b^2$. So:

$$
\sigma_b^2 = \phi^2 \sigma_b^2 + \sigma_\eta^2
$$

Solve for $\sigma_b^2$:

$$
\sigma_b^2 (1 - \phi^2) = \sigma_\eta^2
\quad\Rightarrow\quad
\sigma_b^2 = \frac{\sigma_\eta^2}{1 - \phi^2}
$$

✅ This is the **discrete-time steady-state variance**.

---

### **3. Relation to continuous-time parameters**

From continuous-time ECRV:

$$
\dot{b}(t) = -\frac{1}{\tau} b(t) + w(t), \quad w(t) \sim \mathcal{N}(0, q)
$$

* Steady-state variance in continuous time: $\sigma_b^2 = q \tau / 2$
* Discrete-time equivalent:

$$
\phi = e^{-\Delta t / \tau}, \quad
\sigma_\eta^2 = \sigma_b^2 (1 - \phi^2)
$$

So the **discrete-time white noise driving the ECRV** has variance:

$$
\sigma_\eta^2 = \frac{q \tau}{2} \left( 1 - e^{-2 \Delta t / \tau} \right)
$$

---

### **4. Summary**

**Discrete-time ECRV:**

$$
\boxed{
b_{k+1} = e^{-\Delta t / \tau} b_k + \eta_k, \quad \eta_k \sim \mathcal{N}(0, \sigma_\eta^2)
}
$$

**Steady-state statistics:**

$$
\begin{aligned}
\text{Var}[b_k] &= \sigma_b^2 = \frac{\sigma_\eta^2}{1 - e^{-2\Delta t/\tau}} \\
\sigma_\eta^2 &= \sigma_b^2 (1 - e^{-2\Delta t/\tau})
\end{aligned}
$$

* Autocorrelation decays exponentially:

$$
\text{E}[b_k b_{k+m}] = \sigma_b^2 e^{-m \Delta t / \tau}
$$

* These formulas are **exact for any timestep $\Delta t$**.

---

If you want, I can make a **tiny MATLAB snippet** that **verifies this variance numerically** for a long simulated ECRV series. It’s a nice sanity check for simulations.

Do you want me to do that?

Perfect! Let’s do a **MATLAB simulation** to verify the **steady-state variance** of a discrete-time ECRV.

Here’s a concise example:

```matlab
%% Parameters
dt = 0.01;        % time step [s]
T = 100;          % total simulation time [s]
tau = 50;         % correlation time [s]
sigma_b = 0.05;   % desired steady-state bias standard deviation

N = T/dt;         % number of steps

%% Discrete-time ECRV parameters
phi = exp(-dt/tau);                % correlation coefficient
sigma_eta = sigma_b*sqrt(1 - phi^2); % driving white noise std

%% Simulate ECRV
b = zeros(N,1);
b(1) = sigma_b*randn; % initial bias

for k = 1:N-1
    b(k+1) = phi*b(k) + sigma_eta*randn;
end

%% Compute empirical variance
empirical_var = var(b);
fprintf('Theoretical variance: %.6f\n', sigma_b^2);
fprintf('Empirical variance: %.6f\n', empirical_var);

%% Plot
time = (0:N-1)*dt;
figure;
plot(time, b);
xlabel('Time [s]');
ylabel('Bias [unit]');
title('Discrete-time Exponentially Correlated Random Variable');
grid on;
```

---

### **Explanation**

1. `phi = exp(-dt/tau)` → discrete correlation coefficient.
2. `sigma_eta = sigma_b*sqrt(1 - phi^2)` → ensures the **steady-state variance** matches `sigma_b^2`.
3. `b(k+1) = phi*b(k) + sigma_eta*randn` → the discrete ECRV update.
4. `var(b)` → should match `sigma_b^2` after enough samples (steady state).

If you run this, the **empirical variance** should converge very closely to the theoretical $\sigma_b^2 = 0.05^2 = 0.0025$.

---

I can also make a **plot of the autocorrelation function** to verify it decays exactly like $e^{-m\Delta t/\tau}$. This is very useful for sensor simulation validation.

Do you want me to include that too?

Great! Let’s extend the MATLAB code to **compute and plot the autocorrelation** of the simulated ECRV and compare it to the theoretical exponential decay.

```matlab
%% Parameters (same as before)
dt = 0.01;        % time step [s]
T = 100;          % total simulation time [s]
tau = 50;         % correlation time [s]
sigma_b = 0.05;   % desired steady-state bias standard deviation

N = T/dt;         % number of steps

%% Discrete-time ECRV parameters
phi = exp(-dt/tau);                
sigma_eta = sigma_b*sqrt(1 - phi^2); 

%% Simulate ECRV
b = zeros(N,1);
b(1) = sigma_b*randn; 

for k = 1:N-1
    b(k+1) = phi*b(k) + sigma_eta*randn;
end

%% Compute empirical variance
empirical_var = var(b);
fprintf('Theoretical variance: %.6f\n', sigma_b^2);
fprintf('Empirical variance: %.6f\n', empirical_var);

%% Plot ECRV
time = (0:N-1)*dt;
figure;
subplot(2,1,1);
plot(time, b);
xlabel('Time [s]');
ylabel('Bias [unit]');
title('Discrete-time Exponentially Correlated Random Variable');
grid on;

%% Compute and plot autocorrelation
max_lag = 500; % number of lags to compute
acf_empirical = xcorr(b, max_lag, 'coeff'); % normalized autocorrelation
lags = -max_lag:max_lag;

% Theoretical autocorrelation
acf_theoretical = sigma_b^2 * exp(-abs(lags)*dt/tau) / sigma_b^2; % normalized

subplot(2,1,2);
plot(lags*dt, acf_empirical, 'b', 'LineWidth', 1.2); hold on;
plot(lags*dt, acf_theoretical, 'r--', 'LineWidth', 1.5);
xlabel('Lag [s]');
ylabel('Autocorrelation');
legend('Empirical', 'Theoretical');
title('Autocorrelation of ECRV');
grid on;
```

---

### **What this does**

1. Computes **empirical autocorrelation** using `xcorr`.
2. Plots it versus **theoretical exponential decay**:

$$
\text{ACF}(\Delta t) = e^{-|\Delta t| / \tau}
$$

3. You should see the **blue line** (empirical) closely following the **red dashed line** (theoretical).

This is a powerful sanity check for simulating **sensor bias drift** realistically.

I can also show a **version where multiple sensors are simulated simultaneously** to visualize **correlated vs independent biases** if you want.

Do you want me to do that?


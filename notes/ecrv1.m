clear all; close all; clc;

dt          = 0.01;     % time step [s]
T           = 1000;     % total simulation time [s]
N           = T/dt;
time_vec_s  = (0:N-1)*dt;
tau         = 50;       % correlation time [s]
sigma_b     = 0.05;     % steady-state bias [unit of sensor]
b           = zeros(N,1);
b(1)        = sigma_b*randn;
% b(1)    = 0.5; % cannot do this, this will not produce the right statistics

phi         = exp(-dt/tau);
sigma_eta   = sigma_b * sqrt(1 - phi^2);
for k = 1:N-1
    b(k+1)      = phi * b(k) + sigma_eta * randn;
end

mean(b)
std(b)

plot(time_vec_s, b)
xlabel('Time [s]')
ylabel('Bias drift')
title('Exponentially correlated sensor bias drift')

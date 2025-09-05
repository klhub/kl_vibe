close all; clear all; clc;

% Desired stats and dynamics
mu          = -0.5123;      % mean
sigma       =  0.6321;      % stationary std
tau         = 30*60;        % seconds (correlation time)
dt          = 1;            % sample time
time_end    = 10*60*60;     % end time
MC          = 1000;          % Monte Carlo runs

time_vec_s  = 0 : dt : time_end-dt;
NN          = length(time_vec_s);
x_all       = zeros(MC,NN);

for ii=1:MC
    x_all(ii, :) = fogm_sim(time_end, dt, tau, mu, sigma);
end

x_flatten = reshape(x_all.', 1, []);

% Quick checks
fprintf('Sample mean: %.3f (target %.3f)\n', mean(x_flatten), mu);
fprintf('Sample std : %.3f (target %.3f)\n', std(x_flatten), sigma); % use population std

% Plot the last run
plot(time_vec_s/60/60, x_all(end, :)); 
xlabel('Time, hr'); 
ylabel('FOGM Process');
title('The Last Monte Carlo Run');

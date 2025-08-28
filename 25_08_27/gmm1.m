% Gaussian Mixture Model Example without toolboxes
clear; clc; rng(1);

%% Step 1. Generate signal from 3 Gaussians
N           = 30000; % total samples
true_means  = [ 0.7, 1.0, 1.3];
true_stds   = [ 0.2, 0.3, 0.4];
weights     = [ 0.4, 0.2, 0.4]; % mixture weights (must sum to 1)

% Sample component indices (manual categorical sampling)
u           = rand(N,1);
cumw        = cumsum(weights);
components  = zeros(N,1);
for i = 1:N
    components(i) = find(u(i) <= cumw,1);
end

% Generate samples
data = zeros(N,1);
for k = 1:3
    idx         = (components == k);
    data(idx)   = true_means(k) + true_stds(k)*randn(sum(idx),1);
end

%% Step 2. Estimate parameters using EM Algorithm
K           = 3; % number of Gaussians
% Initialize means, stds, weights
mu          = linspace(min(data), max(data), K);
sigma       = ones(1,K);
pi_k        = ones(1,K)/K;

max_iter    = 100;
tol         = 1e-16;
loglik_old  = -inf;

for iter = 1:max_iter
    % ---- E-step: responsibilities ----
    resp = zeros(N,K);
    for k = 1:K
        resp(:,k) = pi_k(k) * (1/(sqrt(2*pi)*sigma(k))) * ...
                    exp(-0.5*((data - mu(k))/sigma(k)).^2);
    end
    resp = resp ./ sum(resp,2);
    
    % ---- M-step: update params ----
    Nk = sum(resp,1);
    for k = 1:K
        mu(k)       = sum(resp(:,k).*data)/Nk(k);
        sigma(k)    = sqrt(sum(resp(:,k).*(data-mu(k)).^2)/Nk(k));
        pi_k(k)     = Nk(k)/N;
    end
    
    % ---- Check convergence ----
    loglik = sum(log(sum(resp,2)));
    if abs(loglik-loglik_old) < tol
        break;
    end
    loglik_old = loglik;
end

disp('Estimated Means  :'), disp(mu   )
disp('Estimated Stds   :'), disp(sigma)
disp('Estimated Weights:'), disp(pi_k )

%% Step 3. Reconstruct synthetic signal from estimated GMM
synthetic       = zeros(N,1);
u               = rand(N,1);
components_est  = zeros(N,1);
cumw_est        = cumsum(pi_k);
for i = 1:N
    components_est(i)   = find(u(i) <= cumw_est,1);
end

for k = 1:K
    idx                 = (components_est == k);
    synthetic(idx)      = mu(k) + sigma(k)*randn(sum(idx),1);
end

%% Step 4. Compare original vs reconstructed
figure;
subplot(2,1,1); 
histogram(data,50,'Normalization','pdf','FaceAlpha',0.6);
hold on; grid on
x       = linspace(min(data),max(data),500);
pdf_est = zeros(size(x));
for k = 1:K
    pdf_est = pdf_est + pi_k(k)*(1/(sqrt(2*pi)*sigma(k)))* ...
                        exp(-0.5*((x-mu(k))/sigma(k)).^2);
end
plot(x,pdf_est,'r','LineWidth',2)
xlim([-0.5 2.5])
ylim([0 1.2])
title('Original Data with Estimated PDF')

subplot(2,1,2)
histogram(synthetic,50,'Normalization','pdf','FaceAlpha',0.6);
hold on; grid on
plot(x,pdf_est,'r','LineWidth',2)
xlim([-0.5 2.5])
ylim([0 1.2])
title('Synthetic Data from Estimated GMM')

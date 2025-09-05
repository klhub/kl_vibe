function x = fogm_sim(time_end, dt, tau, mu, sigma, x0)
% FOGM / OU simulator with prescribed mean and stationary std
% N: number of samples
% dt: sample time
% tau: correlation time constant
% mu: target mean
% sigma: target stationary std
% x0: (optional) initial value; if omitted, draws from N(mu, sigma^2)

time_vec_s  = 0 : dt : time_end-dt;
NN          = length(time_vec_s);
x           = zeros(1,NN);
phi         = exp(-dt/tau);
q           = sigma^2*(1 - phi^2);  % process noise variance

if nargin < 6 || isempty(x0)
    x(1,1) = mu + sigma*randn;  % start in stationarity
else
    x(1,1) = x0;
end

for jj=2:NN
    x(1, jj) = mu + phi*(x(1, jj-1) - mu) + sqrt(q)*randn;
end

end % function

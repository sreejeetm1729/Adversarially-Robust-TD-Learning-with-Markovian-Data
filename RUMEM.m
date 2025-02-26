%% Sub-sampled Univariate Median of Means Estimator
%---------------------------------------------------%
% Takes as input a data set S of N samples (as a row vector), a sub-sampling gap \tau,
% a confidence parameter \delta, a corruption fraction \epsilon.

% Assumption: N \geq 4 L * \tau (Need enough samples!)

% Returns MoM estimate. 


function [Est] = RUMEM(S, tau, delta, eps)

N = size(S,2); % Number of samples in original set.
%% Step 1: Create Subsampled Set
%----------------------------------%
X=downsample(S,tau); % X is the subsampled set.
n= size(X,2); % number of samples in subsampled set.
%---------------------------------------------------

%% Step2: Bucketing the Subsampled Set
%--------------------------------------%
eps_1 = eps + (tau)/(N) * log(1/delta); % inflated corruption fraction
L=ceil(eps_1*n + log(N/delta)); % Number of buckets
M=floor(n/L); % number of elements in each bucket
B=zeros(L,M); % Matrix that stores bucketed elements. Row i corresponds to bucket i.

% Creating the buckets
for i=1:L
B(i,:) = X(1, (i-1)*M+1: i*M);
end
%--------------------------------------------------------------------------------%


%% Step3: Computing MoM Estimate
%-----------------------------------%
Muhat=zeros(1,L);
for i=1:L
Muhat(1,i)=sum(B(i,:))/M; 
end
Est=median(Muhat);
%------------------------------------------------
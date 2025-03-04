%% Robust Temporal Difference Learning with Linear Function Approximation %%
%-------------------------------------------------------------------------% 
% Rewards are generated based on the Huber-Contamination Model
% Robust Estimator is the Median-of-Means; No Thresholding 
% Comparison between 3 different values of epsilon

%-------------------------------------------------------------------------%
clc; clear all; close all; %#ok<CLALL> 
S=100; % no of states
gamma=0.5; % discount factor
r=10; % rank of feature matrix 

% Note: Length of each feature vector is 'r'.

% Description of Notation
%--------------------------  
% theta_st ---> True fixed point
% P ---> Transition matrix
% R ---> Reward vector
% p ---> Stationary distribution
% phi ---> Feature matrix (Dimension is S by r)

[theta_st,P,R,p,phi]= markov_gen(S,gamma,r); 

%--------------------------------------------------------------------------
% Markov Setting
%--------------------------------------------------------------------------
T=30000; % no of iterations
D=zeros(S,S); % stores the elements of p
for i=1:S
D(i,i)=p(i);
end

Ep=10; % no of epochs over which we average
avg_err1=zeros(1,T); % Stores MSE for eps=0.05
avg_err2=zeros(1,T); % Stores MSE for eps=0.01
avg_err3=zeros(1,T); % Stores MSE for eps=0.001

for k=1:Ep
x1=zeros(r,T); % stores iterates for eps1
x2=zeros(r,T); % stores iterates for eps2
x3=zeros(r,T); % stores iterates for eps3

R_str1=zeros(r,T); % stores historical data needed for MoM
R_str2=zeros(r,T); % ..
R_str3=zeros(r,T); % ..

alpha=0.1; % learning rate 
err1=zeros(1,T); % error for eps1
err2=zeros(1,T); % error for eps2
err3=zeros(1,T); % error for eps3

err1(1,1)=(norm(theta_st))^2;
err2(1,1)=(norm(theta_st))^2;
err3(1,1)=(norm(theta_st))^2;

% initialize from stationary distribution
h=cumsum(p);
z=rand(1);
s_old=find(h > z, 1);


Tinit=10*ceil((log(r*T))^2); % Initial Burn-in time
hatb=zeros(1,r); % Estimate of \bar{b} 

for i=1:T
% Generating the next state s_t+1
d=P(s_old,:); % distribution of s_t+1|s_t 
h=cumsum(d);
z=rand(1);
s_new=find(h > z, 1); % new state s_t+1

%------ Generating Corrupted Rewards ----------%
nse=normrnd(0,1); % nse

rew=R(s_old,1) + nse; % true noisy reward

eps1=0.01;  % probability of corruption
eps2=0.005;  % ...
eps3=0.001; % 

Bias=100;

z=rand(1); 

if (z <= eps1)
    rew1=Bias/eps1;
else
    rew1=rew;
end

if (z <= eps2)
    rew2=Bias/eps2;
else
    rew2=rew;
end

if (z <= eps3)
    rew3=Bias/eps3;
else
    rew3=rew;
end

R_str1(:,i)= rew1*phi(s_old,:)';
R_str2(:,i)= rew2*phi(s_old,:)';
R_str3(:,i)= rew3*phi(s_old,:)';

%----------------------------------------------%

%--------- Robust TD Update -------------------%
if (i <= Tinit)
    x1(:,i+1)=x1(:,i); % No updates prior to burn-in time
    x2(:,i+1)=x2(:,i); % No updates prior to burn-in time
    x3(:,i+1)=x3(:,i); % No updates prior to burn-in time
else


% Update for eps1
%--------------------------
% Constructing \hat{b}:
for j=1:r
tau=ceil(log(T)); % subsampling gap.
delta=1/(r*T); % error probability.
Data=R_str1(j, 1:i); % Construct data set for estimating b_j. 
hatb(1,j)=RUMEM(Data, 1, delta, eps1); % Invoke RUMEM Estimator 
end

g_R=(gamma*(phi(s_new,:))*x1(:,i)-(phi(s_old,:))*x1(:,i))*(phi(s_old,:))'+hatb'; 
% Robust TD update direction
x1(:,i+1)=x1(:,i)+alpha*g_R;
%---------------------------

% Update for eps2
%--------------------------
% Constructing \hat{b}:
for j=1:r
tau=ceil(log(T)); % subsampling gap.
delta=1/(r*T); % error probability.
Data=R_str2(j, 1:i); % Construct data set for estimating b_j. 
hatb(1,j)=RUMEM(Data, 1, delta, eps2); % Invoke RUMEM Estimator 
end

g_R=(gamma*(phi(s_new,:))*x2(:,i)-(phi(s_old,:))*x2(:,i))*(phi(s_old,:))'+hatb'; 
% Robust TD update direction
x2(:,i+1)=x2(:,i)+alpha*g_R;
%---------------------------

% Update for eps3
%--------------------------
% Constructing \hat{b}:
for j=1:r
tau=ceil(log(T)); % subsampling gap.
delta=1/(r*T); % error probability.
Data=R_str3(j, 1:i); % Construct data set for estimating b_j. 
hatb(1,j)=RUMEM(Data, 1, delta, eps3); % Invoke RUMEM Estimator 
end

g_R=(gamma*(phi(s_new,:))*x3(:,i)-(phi(s_old,:))*x3(:,i))*(phi(s_old,:))'+hatb'; 
% Robust TD update direction
x3(:,i+1)=x3(:,i)+alpha*g_R;
%---------------------------
end
err1(:,i)=(norm(theta_st-x1(:,i)))^2;
err2(:,i)=(norm(theta_st-x2(:,i)))^2;
err3(:,i)=(norm(theta_st-x3(:,i)))^2;
%----------------------------------------------%
s_old=s_new; % continuity of trajectory
end
avg_err1=avg_err1+err1;
avg_err2=avg_err2+err2;
avg_err3=avg_err3+err3;
end

figure
plot(avg_err1/Ep, 'Color', uint8([17 17 17]), 'LineWidth',2);
hold on;
p1=plot(avg_err2/Ep, '--o', 'Color','r', 'LineWidth', 1.5);
p1.MarkerSize = 5;
p1.MarkerIndices = 1:500:T;
hold on;
p2=plot(avg_err3/Ep, '-^', 'Color','b', 'LineWidth', 1.5);
p2.MarkerSize = 5;
p2.MarkerIndices = 10:550:T;
xlim([1 T]);
leg1=legend('$\epsilon=0.01$', '$\epsilon=0.005$', '$\epsilon=0.001$'); 
set(leg1,'Interpreter','latex','fontsize',22);
ax=gca;
set(ax, 'fontsize',15, 'fontname', 'times','FontWeight','bold');
ax.LineWidth=1.2;
xlab=xlabel('${{t}}$','Interpreter','latex');
set(xlab,'fontsize',30,'fontname', 'times','FontWeight','bold');
ylab=ylabel('$E_t$','Interpreter','latex');
set(ylab,'fontsize',30, 'fontname', 'times','FontWeight','bold');
grid on;

ax = gca;
ax.XAxis.LineWidth = 1.5;
ax.YAxis.LineWidth = 1.5;
outerpos = ax.OuterPosition;
ti = ax.TightInset; 
left = outerpos(1) + ti(1);
bottom = outerpos(2) + ti(2);
ax_width = outerpos(3) - ti(1) - ti(3)-0.01;
ax_height = outerpos(4) - ti(2) - ti(4)-0.01;
ax.Position = [left bottom ax_width ax_height]; 

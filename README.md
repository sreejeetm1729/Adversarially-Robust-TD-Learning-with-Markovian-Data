# Adversarially-Robust-TD-Learning-with-Markovian-Data
One of the most basic problems in reinforcement learning (RL) is policy evaluation: estimating
the long-term return, i.e., value function, corresponding to a given fixed policy. The celebrated
Temporal Difference (TD) learning algorithm addresses this problem, and recent work has
investigated finite-time convergence guarantees for this algorithm and variants thereof. However,
these guarantees hinge on the reward observations being always generated from a well-behaved
(e.g., sub-Gaussian) true reward distribution. Motivated by harsh, real-world environments where
such an idealistic assumption may no longer hold, we revisit the policy evaluation problem from
the perspective of adversarial robustness. In particular, we consider a Huber-contaminated reward
model where an adversary can arbitrarily corrupt each reward sample with a small probability
ϵ. Under this observation model, we first show that the adversary can cause the vanilla TD
algorithm to converge to any arbitrary value function. We then develop a novel algorithm called
Robust-TD and prove that its finite-time guarantees match that of vanilla TD with linear function
approximation up to a small O(ϵ) term that captures the effect of corruption. We complement
this result with a minimax lower bound, revealing that such an additive corruption-induced
term is unavoidable. To our knowledge, these results are the first of their kind in the context of
adversarial robustness of stochastic approximation schemes driven by Markov noise. The key
new technical tool that enables our results is an analysis of the Median-of-Means estimator with
corrupted, time-correlated data that might be of independent interest to the literature on robust
statistics.
<table>
<tr>
  <td>
    <img src="https://github.com/sreejeetm1729/Adversarially-Robust-TD-Learning-with-Markovian-Data/blob/main/Vulnerability%20(2).jpg" style="width:250px">
    <img src="https://github.com/sreejeetm1729/Adversarially-Robust-TD-Learning-with-Markovian-Data/blob/main/dim_step_vuln%20(2).jpg" style="width:250px">
    <img src="https://github.com/sreejeetm1729/Adversarially-Robust-TD-Learning-with-Markovian-Data/blob/main/Exp1_plots%20(2).jpg" style="width:250px">
    <img src="https://github.com/sreejeetm1729/Adversarially-Robust-TD-Learning-with-Markovian-Data/blob/main/Noise_effect%20(2).jpg" style="width:250px">
    <img src="https://github.com/sreejeetm1729/Adversarially-Robust-TD-Learning-with-Markovian-Data/blob/main/vulnerabilityexp1%20(2)%20(1).jpg" style="width:250px">
    <img src="https://github.com/sreejeetm1729/Adversarially-Robust-TD-Learning-with-Markovian-Data/blob/main/vulnerabilityexp1%20(2).jpg" style="width:250px">
 </td>
</tr>

# Design and verification

## Anchor policy and shared score

At each state s, choose action a with probability proportional to w(s,a) exp(phi(s,a) dot theta). We audit theta=0, so p(a|s)=w(s,a)/sum_b w(s,b). Differentiating its log gives score vector phi(s,a) minus its policy-weighted feature mean. Integer weights/features make p and the score rational at zero. Positive weights give full action support; environment transitions may have zeros. The same theta coordinates are shared across states and all times. Transitions, transition rewards and the initial state are supplied and independent of theta.

The objective J(theta) is expected total undiscounted reward over the stated finite horizon. This is a derivative at one anchored policy, not an optimization run or a rational expression for probabilities at arbitrary theta.

## Likelihood ratios and reward-to-go

A complete trajectory probability is the product of policy probabilities and environment transition probabilities. Differentiating a finite sum is exact. Its log derivative is the sum of visited scores, giving the full-return estimator G_full = total_reward times sum_of_scores. Zero environment probabilities have no contribution and no theta derivative.

For any action time t, its score has conditional mean zero given the prefix through the current state. Earlier rewards are determined by this prefix, so their product with the current score has zero expectation. Dropping those earlier rewards preserves the gradient mean, yielding G0 = sum_t score_t times sum_{u>=t} reward_u. This is the reward-to-go estimator. The two means agree, but no universal ordering of their total covariance traces is asserted.

A separate backward sensitivity recursion evaluates J and its derivative. With terminal value and derivative zero, value at time/state is the policy/transition average of reward plus next value. Its derivative averages score times that reward-plus-next-value and the next value's derivative. This handles shared policy parameters and random next states without enumerating complete trajectories. We require every estimator's path-integrated mean to agree exactly with this root derivative.

## Baseline control variates and optimal trace variance

For node j=(time,state), define Z_j as the visited score vector at that node, zero if the path does not visit it. A deterministic scalar b_j that depends only on time/state is known before the corresponding action. Therefore E[Z_j]=0 and G_b = G0 - sum_j b_j Z_j has unchanged mean. The estimator holds b fixed; action-dependent baselines, random fitted baselines and differentiating a critic inside this estimator are outside its input contract.

Let H_ij = E[Z_i dot Z_j] and h_i=E[Z_i dot G0]. For equal-time distinct-state nodes the product is zero pathwise. For different times, condition on the later action's prefix: the earlier Z is fixed and the later score has conditional mean zero. Thus every off-diagonal H entry is zero. This remains true with shared parameters, random state transitions and scores correlated with future rewards. The correlation h_i uses the entire G0, not merely the local return term.

Since all means equal the true gradient, minimizing trace covariance equals minimizing E[||G_b||^2]. Expanding gives a constant minus 2 sum_i b_i h_i plus sum_i b_i^2 H_ii. For H_ii>0, complete the square: b_i*=h_i/H_ii, and excess trace variance is exactly sum_i H_ii (b_i-b_i*)^2. If H_ii=0 then Z_i=0 almost surely and h_i=0, so that baseline is unidentifiable; any choice has no effect and zero is reported canonically. Unvisited nodes may still have nonzero conditional state values. The code returns the entire Gram matrix and checks its diagonality rather than hiding this structure.

The result is optimal only within scalar deterministic time/state baselines for G0, using the current Euclidean parameter-space norm. It is parameterization-dependent, need not minimize each coordinate variance separately, and does not optimize the different full-return estimator or all unbiased estimators. Exact value baselines need not solve these normal equations.

## One-step counterexample

With Bernoulli action probability p and reward equal to that action, the scalar logit score is 1-p after action 1 and -p after action 0. True gradient is p(1-p). The no-baseline estimator has variance p(1-p)^3. The exact value baseline p gives variance p(1-p)(1-2p)^2. The variance-minimizing baseline is 1-p and gives the identical estimate p(1-p) for both actions, hence zero variance. At p=4/5 these are 4/625, 36/625 and zero. This establishes a counterexample to universal improvement by the value baseline, not a typical-case comparison or large-scale RL conclusion.

The separate two-state, three-step example has two non-collinear shared feature directions, random transitions and signed rewards. Full-return, reward-to-go, value-baseline and optimal-baseline trace variances are approximately 5.4508, 2.2574, 1.5773 and 1.4625. Exact fractions and all 64 paths remain in JSON/SVG metadata. These supplied fixtures are not data splits, tuning runs, confidence intervals or training evidence.

## Support preflight and output contract

For the initial state, set its path count to one. At each depth, the next-state count is the sum over current states/actions of current count times the indicator that the supplied transition is positive. Full action support makes every such supported branch positive. Summing final-state counts gives the exact complete-path count, before trajectory generation. There are no dead ends because each transition row sums to one. Every depth's prefix count is no greater than the final complete count. Horizon zero therefore has exactly one empty path. A refusal returns no partial result. Shapes, integer/rational ranges, horizon <=6 and the path cap define the supported tiny domain; arithmetic bit sizes, serialization, RAM and wall time are not claimed bounded by the path count alone.

All result fractions are exact numerator/positive-denominator pairs. Each complete path includes all steps, returns-to-go and estimator vectors. Every estimator includes its full second-moment matrix and covariance, not merely a trace. Every time/state node is included even if unreachable. Input and returned nested data are independent, and no RNG or network is used.

## Independent verification and reproducible figure

For all 256 integer Bernoulli weight pairs in 1..16, analytic formulas independently check gradient, exact value baseline, optimal baseline and all relevant variances. A separate oracle enumerates the full Cartesian products of action strings and next-state strings, including zero-probability paths; it does not call the production recursive path generator or backward sensitivity recursion. Dual numbers differentiate normalized affine weights w(1+phi_j x), which have the same derivative at x=0 as the actual exponential family. The oracle uses products and quotient differentiation, not production score vectors, and checks 48 coordinates across 24 model/horizon settings. Central differences of the actual exponential-family objective give another independent numerical check, not the exact proof.

A second direct Cartesian calculation checks every covariance entry and estimator mean, with hand-reconstructed scores and baseline values. Other checks inspect every returned path probability/reward/return-to-go, covariance symmetry and nonnegative principal minors, control-variate Gram orthogonality, baseline normal equations, the full excess-variance identity and unvisited-baseline invariance. Zero horizon, single action, zero scores, three-state/action/coordinate models, support-count boundaries, malformed models, strict CLI, isolation and constant feature/reward-shift identities cover edge cases. A constant reward shift changes each identifiable optimal baseline by the remaining deterministic reward shift, leaving the adjusted estimator covariance and true gradient unchanged.

The SVG has distinct explicitly labeled scales for its separate models. Bar coordinates derive from exact fractions and only the final coordinates/visible numeric labels are rounded. Full unchanged demo metadata is zlib-compressed base64. Four figure tests check all seven bars, numeric text positions, full metadata equality, byte regeneration and rejection of changed undisplayed path/covariance data. The image is also rendered and inspected as pixels; geometric tests alone cannot establish readability.

## Attribution

Greensmith, Bartlett and Baxter, Variance Reduction Techniques for Gradient Estimates in Reinforcement Learning, JMLR 5 (2004), Sections 4-6: https://jmlr.org/papers/volume5/greensmith04a/greensmith04a.pdf. The paper's setting is GPOMDP/average reward. ScoreLab uses standard score/control-variate ideas but its code, tiny finite-horizon formulation, exact covariance/Gram audit, fixtures and independent oracles are original. It does not implement the paper's algorithms, reproduce its experiments or claim its mixing-time bounds. No external code or data is copied. MIT licensed.

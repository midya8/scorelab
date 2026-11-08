# Matrix score controls: proof and verification

This extends the finite-model scalar audit without changing it. All probabilities, scores and moments are exact rational numbers at the supplied softmax anchor. No control is trained. The matrix acts on the score, not on rewards or model parameters.

## The control class

Let G be the full reward-to-go gradient estimator vector of dimension d. For each time/state node j, Z_j is the score vector if that node is visited and zero otherwise. Choose any deterministic d-by-d matrix B_j, fixed inside the estimator. Define G_B = G - sum_j B_j Z_j.

Conditional score means are zero given the prefix through the current state. Thus E[Z_j]=0 and every G_B has the same mean, the exact expected-return derivative. Different nodes at the same time are mutually exclusive. At different times the later score has conditional mean zero and the earlier score is prefix-measurable. Therefore E[Z_i Z_j^T]=0 for i != j, entry by entry. This is stronger than merely zero expected dot products. Shared parameters and random transitions do not break it. The implementation enumerates the entire coordinate Gram matrix and checks all off-node blocks, rather than presuming independence of scores.

## Singular normal equations

Set S_j=E[Z_j Z_j^T] and C_j=E[G Z_j^T]. S_j is symmetric positive semidefinite. If v is in its null space, E[(v^T Z_j)^2]=v^T S_j v=0. On the finite positive-probability support, v^T Z_j is zero on each path. Hence C_j v=0. Every row of C_j is in the row space of S_j, so B_j S_j=C_j is consistent, even when S_j is singular.

Exact fraction RREF solves each row system S_j b=c with free variables set to zero. There is no ridge, tolerance, numerical rank threshold, pseudoinverse or minimum-norm claim. Rank and d-rank coefficient degrees of freedom per output row are returned. Zero-rank and unvisited nodes canonically use the zero matrix. Free-variable choices are coordinate-dependent; the correction on supported paths is not.

If B and B' solve the same equations, D S_j=0 for D=B-B'. Each row has zero second moment against Z_j, so D Z_j=0 on every supported path. The coefficients need not be unique, but their corrections are. With multiple nodes this conclusion holds node by node.

## Whole-covariance optimum and certificate

Let B* solve all normal equations, and let R=G-sum_j B*_j Z_j. Orthogonality of nodes gives E[R Z_j^T]=C_j-B*_j S_j=0. For any other fixed controls B, set D_j=B_j-B*_j. Since means do not change and both cross terms vanish,

```text
Cov(G_B) - Cov(R) = sum_j D_j S_j D_j^T.
```

Each summand is positive semidefinite: u^T D_j S_j D_j^T u = E[(u^T D_j Z_j)^2] >= 0 for every vector u. Thus B* minimizes covariance in positive-semidefinite order within this class, and minimizes every fixed positive-semidefinite weighted trace of the output covariance. This includes every individual output variance. It is not an optimum over all unbiased estimators. It is also not an optimization of full-return score estimators, fitted controls, learned critics or action-dependent baselines.

The audit returns every node summand, their sum, the exact covariance difference and trace gap for no baseline, value baseline, scalar-optimal baseline, matrix-optimal controls and any supplied custom controls. A scalar baseline b is represented by b times the identity. Original scalar optimality is still only trace optimality inside the smaller scalar class. A positive-semidefinite covariance gap may have negative off-diagonal entries; entrywise nonnegativity is not claimed or tested.

## Fixtures and independent verification

The one-step three-action fixture uses weights [1,2,3], feature vectors (0,0),(1,0),(0,1) and rewards 0,1,2. S=[[2/9,-1/6],[-1/6,1/4]], C=[[7/27,-5/18],[-5/18,1/3]]. Solving gives B*=[[2/3,-2/3],[-1/2,1]]. Direct substitution on all three actions gives the same adjusted gradient [-1/9,1/3], so covariance is zero. The scalar optimum b=64/51 has trace 53/612. This fixture has a complete action-function score span; zero residual variance is not a general guarantee.

The separate two-state, three-step shared-parameter fixture retains all 64 positive paths. Its score ranks [1,0,1,1,1,1] explicitly exercise singular and unvisited nodes. Matrix trace is approximately 1.460918 versus scalar-optimal 1.462519. This is a small descriptive change on one supplied model, not empirical evidence about typical RL tasks.

Eleven matrix-audit tests supplement the original 22 tests. They cover 15 independently enumerated model/horizon settings (12 sequential variations and three one-state horizons), with Cartesian action and next-state strings rather than the production recursive enumerator. They reconstruct probabilities, scores, rewards, reward-to-go gradients, every cross moment, Gram entry, adjusted mean/covariance and residual-score normal equation. They inspect exact positive-semidefinite covariance gaps/node terms using every principal minor up to three dimensions, an arbitrary custom matrix control, rank-deficient three-coordinate features with distinct valid coefficients, invertible coordinate mixing and covariance congruence, one-dimensional equivalence to scalar baselines, horizon zero, unvisited/zero-score/single-action nodes, RREF pivot patterns, strict validation, preflight refusal, CLI and isolation. Three further figure tests verify exact byte regeneration, compressed complete metadata, all four bar geometries/numeric labels and refusal of changes to undisplayed covariance/path data. Total: 36 tests.

## Bounds and attribution

The scalar model limits, strict rational-pair bounds, horizon cap and complete-trajectory preflight still apply. Matrix input adds at most `horizon*states*d*d` pairs. The coordinate Gram has `(horizon*states*d)^2` entries. A path cap is not a total arithmetic, byte, memory or runtime bound. No external service, RNG or dataset is used.

Score-function control variates and optimal-baseline cautions are standard; see Greensmith, Bartlett and Baxter, JMLR 5 (2004), Sections 4-6: https://www.jmlr.org/papers/v5/greensmith04a.html . That paper's setting is GPOMDP/average reward. The implementation does not implement GPOMDP or reproduce the paper's experiments or theorems. This finite-horizon matrix formulation, exact solver, fixtures, full-covariance certificates and independent tests are original MIT-licensed code and derivations here. No external code or data was copied.

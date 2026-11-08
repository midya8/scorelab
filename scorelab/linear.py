"""Exact matrix score controls; rational rank-aware normal equations."""
from fractions import Fraction as F
from .core import audit, pack, rational


def transpose(a):
    return [list(row) for row in zip(*a)]


def multiply(a,b):
    return [[sum((x*y for x,y in zip(row,col)),F(0)) for col in zip(*b)] for row in a]


def zero(n):
    return [[F(0) for _ in range(n)] for _ in range(n)]


def packed(a):
    return [[pack(x) for x in row] for row in a]


def solve(a,b):
    """RREF consistent square system, canonical free variables zero, not a pseudoinverse."""
    n=len(a);r=[row[:]+[v] for row,v in zip(a,b)];pivots=[]
    for col in range(n):
        found=next((i for i in range(len(pivots),n) if r[i][col]),None)
        if found is None:continue
        k=len(pivots);r[k],r[found]=r[found],r[k];scale=r[k][col];r[k]=[v/scale for v in r[k]]
        for i in range(n):
            if i!=k:
                scale=r[i][col];r[i]=[v-scale*w for v,w in zip(r[i],r[k])]
        pivots.append(col)
    if any(not any(row[:n]) and row[n] for row in r):raise ValueError('inconsistent normal equations')
    x=[F(0)]*n
    for i,col in enumerate(pivots):x[col]=r[i][n]
    assert [sum((v*w for v,w in zip(row,x)),F(0)) for row in a]==b
    return x,len(pivots)


def audit_linear(model,*,horizon=3,trajectory_limit=4096,control=None):
    """Extend the unchanged scalar audit with optimal matrix-valued score controls.

    control, when supplied, has shape [horizon][states][dimension][dimension]
    with bounded rational pairs, as in the model. Rows are output coordinates,
    columns are score coordinates. It is fixed across paths at each time/state.
    """
    out=audit(model,horizon=horizon,trajectory_limit=trajectory_limit)
    ns=len(model['weights']);d=len(model['features'][0][0]);nodes=horizon*ns
    custom=None
    if control is not None:
        if (type(control) is not list or len(control)!=horizon or
            any(type(row) is not list or len(row)!=ns for row in control) or
            any(type(m) is not list or len(m)!=d or any(type(row) is not list or len(row)!=d for row in m) for layer in control for m in layer)):
            raise ValueError('control requires shape [horizon][states][dimension][dimension] of rational pairs')
        custom=[[[rational(x) for x in row] for row in m] for layer in control for m in layer]
    s=[zero(d) for _ in range(nodes)];c=[zero(d) for _ in range(nodes)];gram=zero(nodes*d);paths=[]
    for path in out['trajectories']:
        p=F(*path['probability']);g=list(map(lambda x:F(*x),path['gradient_estimates']['reward_to_go']));z=[[F(0)]*d for _ in range(nodes)]
        for step in path['steps']:z[step['time']*ns+step['state']]=[F(*x) for x in step['score']]
        for n in range(nodes):
            for i in range(d):
                for j in range(d):s[n][i][j]+=p*z[n][i]*z[n][j];c[n][i][j]+=p*g[i]*z[n][j]
        flat=[x for v in z for x in v]
        for i in range(nodes*d):
            for j in range(nodes*d):gram[i][j]+=p*flat[i]*flat[j]
        paths.append((path,p,g,z))
    assert all(gram[i][j]==0 for i in range(nodes*d) for j in range(nodes*d) if i//d!=j//d)
    optimal=[];ranks=[]
    for n in range(nodes):
        rows=[solve(s[n],row) for row in c[n]];optimal.append([row[0] for row in rows]);ranks.append(rows[0][1]);assert multiply(optimal[n],s[n])==c[n]
    controls={'optimal_linear':optimal}
    if custom is not None:controls['custom_linear']=custom
    means={};covs={}
    for name,b in controls.items():
        samples=[]
        for path,p,g,z in paths:
            v=[g[i]-sum((b[n][i][j]*z[n][j] for n in range(nodes) for j in range(d)),F(0)) for i in range(d)];path['gradient_estimates'][name]=list(map(pack,v));samples.append((p,v))
        mu=[sum((p*v[i] for p,v in samples),F(0)) for i in range(d)]
        second=[[sum((p*v[i]*v[j] for p,v in samples),F(0)) for j in range(d)] for i in range(d)]
        cov=[[second[i][j]-mu[i]*mu[j] for j in range(d)] for i in range(d)]
        assert list(map(pack,mu))==out['exact_gradient'];means[name]=mu;covs[name]=cov
        out['estimators'][name]={'mean':list(map(pack,mu)),'second_moment':packed(second),'covariance':packed(cov),'trace_variance':pack(sum((cov[i][i] for i in range(d)),F(0)))}
    # Every scalar baseline is a matrix control b*I. Compare entire covariance,
    # not just its diagonal or trace. Full-return belongs to a different class.
    candidates={**controls}
    for name,key in [('reward_to_go',None),('value_baseline','value_baseline'),('optimal_baseline','optimal_baseline')]:
        candidates[name]=[[[F(*out['baseline_nodes'][n][key]) if i==j and key else F(0) for j in range(d)] for i in range(d)] for n in range(nodes)]
    certificates={}
    for name,b in candidates.items():
        terms=[];total=zero(d)
        for n in range(nodes):
            diff=[[b[n][i][j]-optimal[n][i][j] for j in range(d)] for i in range(d)];term=multiply(multiply(diff,s[n]),transpose(diff));terms.append(packed(term))
            for i in range(d):
                for j in range(d):total[i][j]+=term[i][j]
        candidate=[[F(*x) for x in row] for row in out['estimators'][name]['covariance']]
        assert total==[[candidate[i][j]-covs['optimal_linear'][i][j] for j in range(d)] for i in range(d)]
        certificates[name]={'covariance_gap':packed(total),'node_psd_terms':terms,'trace_gap':pack(sum((total[i][i] for i in range(d)),F(0)))}
    for n,node in enumerate(out['baseline_nodes']):node.update({'score_second_moment':packed(s[n]),'gradient_score_cross_moment':packed(c[n]),'optimal_matrix_control':packed(optimal[n]),'score_rank':ranks[n],'coefficient_nullity_per_row':d-ranks[n]})
    out.update({'format':'scorelab-linear-audit-v1','matrix_controls':{name:[[packed(b[t*ns+s]) for s in range(ns)] for t in range(horizon)] for name,b in controls.items()},'coordinate_control_gram':packed(gram),'linear_covariance_certificates':certificates,
        'linear_scope':'G_B = G_RTG - sum_node B_node Z_node; deterministic time/state d-by-d matrices held fixed in the estimator. Optimal is covariance-minimal in the positive-semidefinite order within this matrix-score-control class only. It is not optimal over every unbiased estimator or over full-return estimators. RREF uses free coordinates zero, not a Moore-Penrose inverse or minimum-norm solution. Singular coefficients need not be unique, but their pathwise corrections are unique on the supported distribution. No fitted controls, causal claim or empirical training performance.'})
    return out

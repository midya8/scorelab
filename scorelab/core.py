"""Exact score-function gradient distributions at a rational softmax anchor."""
from fractions import Fraction as F
from copy import deepcopy


def pack(x):return [x.numerator,x.denominator]
def dot(x,y):return sum((a*b for a,b in zip(x,y)),F(0))
def rational(x,*,probability=False):
    if type(x) is not list or len(x)!=2 or any(type(v) is not int for v in x) or not -32<=x[0]<=32 or not 1<=x[1]<=16:raise ValueError('rational must be integer [n,d], -32<=n<=32, 1<=d<=16')
    v=F(*x)
    if probability and not 0<=v<=1:raise ValueError('transition probability must be in [0,1]')
    return v


def validate(model,horizon,trajectory_limit):
    if type(model) is not dict or set(model)!=set(['weights','features','transitions','rewards','initial_state']):raise ValueError('model requires exactly weights, features, transitions, rewards, initial_state')
    w=model['weights'];phi=model['features'];p=model['transitions'];r=model['rewards']
    if type(w) is not list or not 1<=len(w)<=3 or type(w[0]) is not list or not 1<=len(w[0])<=3:raise ValueError('weights require 1..3 states and 1..3 actions')
    ns=len(w);na=len(w[0])
    if any(type(row) is not list or len(row)!=na or any(type(v) is not int or not 1<=v<=16 for v in row) for row in w):raise ValueError('weights must be rectangular integer values in 1..16')
    if type(phi) is not list or len(phi)!=ns or any(type(row) is not list or len(row)!=na for row in phi):raise ValueError('features must match states/actions')
    if type(phi[0][0]) is not list or not 1<=len(phi[0][0])<=3:raise ValueError('feature dimension must be in 1..3')
    dim=len(phi[0][0])
    if any(type(x) is not list or len(x)!=dim or any(type(v) is not int or not -8<=v<=8 for v in x) for row in phi for x in row):raise ValueError('features must have a fixed dimension and integers in -8..8')
    for name,x in [('transitions',p),('rewards',r)]:
        if type(x) is not list or len(x)!=ns or any(type(row) is not list or len(row)!=na or any(type(v) is not list or len(v)!=ns for v in row) for row in x):raise ValueError(name+' must have shape [states][actions][next states] of rational pairs')
    probs=[[[rational(v,probability=True) for v in row] for row in state] for state in p];rewards=[[[rational(v) for v in row] for row in state] for state in r]
    if any(sum(row)!=1 for state in probs for row in state):raise ValueError('each transition row must sum exactly to one')
    start=model['initial_state']
    if type(start) is not int or not 0<=start<ns:raise ValueError('initial_state must be a valid non-boolean integer')
    if type(horizon) is not int or not 0<=horizon<=6:raise ValueError('horizon must be integer in 0..6')
    if type(trajectory_limit) is not int or not 1<=trajectory_limit<=8192:raise ValueError('trajectory_limit must be integer in 1..8192')
    counts=[0]*ns;counts[start]=1
    for _ in range(horizon):
        counts=[sum(counts[s]*sum(probs[s][a][n]>0 for a in range(na)) for s in range(ns)) for n in range(ns)]
    planned=sum(counts)
    if planned>trajectory_limit:raise ValueError('trajectory preflight refused: '+str(planned)+' complete paths exceeds limit '+str(trajectory_limit)+'; no partial result')
    return ns,na,dim,probs,rewards,planned


def _enumerate(model,horizon,probs,rewards,policy,scores):
    ns=len(policy);na=len(policy[0]);rows=[]
    def walk(state,path,prob):
        if len(path)==horizon:
            rows.append((path,prob));return
        for a in range(na):
            for nxt in range(ns):
                p=probs[state][a][nxt]
                if p:walk(nxt,path+[(state,a,nxt,rewards[state][a][nxt],scores[state][a])],prob*policy[state][a]*p)
    walk(model['initial_state'],[],F(1));return rows


def audit(model,*,horizon=3,trajectory_limit=4096,baseline=None):
    ns,na,dim,probs,rewards,planned=validate(model,horizon,trajectory_limit)
    policy=[[F(v,sum(row)) for v in row] for row in model['weights']]
    scores=[[[F(model['features'][s][a][j])-sum(policy[s][b]*model['features'][s][b][j] for b in range(na)) for j in range(dim)] for a in range(na)] for s in range(ns)]
    # Independent backward expected-return sensitivity, including shared parameters.
    values=[[F(0)]*ns for _ in range(horizon+1)];deriv=[[[F(0)]*dim for _ in range(ns)] for _ in range(horizon+1)]
    for t in range(horizon-1,-1,-1):
        for s in range(ns):
            for a in range(na):
                for nxt in range(ns):
                    p=policy[s][a]*probs[s][a][nxt];future=rewards[s][a][nxt]+values[t+1][nxt];values[t][s]+=p*future
                    for j in range(dim):deriv[t][s][j]+=p*(scores[s][a][j]*future+deriv[t+1][nxt][j])
    # z_(t,s) is the score at that visited node, zero on paths not visiting it.
    rows=_enumerate(model,horizon,probs,rewards,policy,scores);nbase=horizon*ns;den=[F(0)]*nbase;num=[F(0)]*nbase;occupancy=[F(0)]*nbase;gram=[[F(0)]*nbase for _ in range(nbase)];prepared=[]
    for path,p in rows:
        ret=sum((step[3] for step in path),F(0));rtg=[sum((x[3] for x in path[t:]),F(0)) for t in range(horizon)]
        g=[sum((path[t][4][j]*rtg[t] for t in range(horizon)),F(0)) for j in range(dim)];z=[[F(0)]*dim for _ in range(nbase)]
        for t,step in enumerate(path):
            idx=t*ns+step[0];z[idx]=step[4][:];occupancy[idx]+=p
        for i in range(nbase):
            den[i]+=p*dot(z[i],z[i]);num[i]+=p*dot(z[i],g)
            for j in range(nbase):gram[i][j]+=p*dot(z[i],z[j])
        prepared.append((path,p,ret,rtg,g,z))
    assert len(rows)==planned and sum(p for _,p in rows)==1
    assert all(gram[i][j]==0 for i in range(nbase) for j in range(nbase) if i!=j)
    optimal=[num[i]/den[i] if den[i] else F(0) for i in range(nbase)];value=[values[t][s] for t in range(horizon) for s in range(ns)]
    baselines={'reward_to_go':[F(0)]*nbase,'value_baseline':value,'optimal_baseline':optimal}
    if baseline is not None:
        if type(baseline) is not list or len(baseline)!=horizon or any(type(row) is not list or len(row)!=ns for row in baseline):raise ValueError('baseline must have shape [horizon][states] of rational pairs')
        baselines['custom_baseline']=[rational(v) for row in baseline for v in row]
    samples={name:[] for name in ['full_return']+list(baselines)};records=[]
    for path,p,ret,rtg,g,z in prepared:
        estimates={'full_return':[ret*sum((step[4][j] for step in path),F(0)) for j in range(dim)]}
        for name,b in baselines.items():estimates[name]=[g[j]-sum((z[i][j]*b[i] for i in range(nbase)),F(0)) for j in range(dim)]
        for name,v in estimates.items():samples[name].append((p,v))
        records.append({'id':len(records),'probability':pack(p),'return':pack(ret),'steps':[{'time':t,'state':s,'action':a,'next_state':n,'reward':pack(r),'score':list(map(pack,sc)),'return_to_go':pack(rtg[t])} for t,(s,a,n,r,sc) in enumerate(path)],'gradient_estimates':{name:list(map(pack,v)) for name,v in estimates.items()}})
    reports={};true_gradient=deriv[0][model['initial_state']]
    for name,dist in samples.items():
        mean=[sum((p*v[j] for p,v in dist),F(0)) for j in range(dim)];second=[[sum((p*v[i]*v[j] for p,v in dist),F(0)) for j in range(dim)] for i in range(dim)];cov=[[second[i][j]-mean[i]*mean[j] for j in range(dim)] for i in range(dim)];variance=sum((cov[i][i] for i in range(dim)),F(0))
        assert mean==true_gradient and variance>=0
        reports[name]={'mean':list(map(pack,mean)),'second_moment':[[pack(v) for v in row] for row in second],'covariance':[[pack(v) for v in row] for row in cov],'trace_variance':pack(variance)}
    best=F(*reports['optimal_baseline']['trace_variance'])
    for name,b in baselines.items():
        excess=sum((den[i]*(b[i]-optimal[i])**2 for i in range(nbase)),F(0));assert F(*reports[name]['trace_variance'])-best==excess;reports[name]['excess_trace_variance_over_optimal']=pack(excess)
    return {'format':'scorelab-audit-v1','horizon':horizon,'model':deepcopy(model),'policy_probabilities':[[pack(v) for v in row] for row in policy],
            'score_vectors':[[list(map(pack,v)) for v in row] for row in scores],'expected_return':pack(values[0][model['initial_state']]),'exact_gradient':list(map(pack,true_gradient)),
            'estimators':reports,'baseline_nodes':[{'time':i//ns,'state':i%ns,'occupancy':pack(occupancy[i]),'value_baseline':pack(value[i]),'optimal_baseline':pack(optimal[i]),'identifiable':bool(den[i]),'score_energy':pack(den[i]),'score_gradient_correlation':pack(num[i])} for i in range(nbase)],
            'control_variate_gram':[[pack(v) for v in row] for row in gram],'trajectories':records,
            'counts':{'preflight_complete_paths':planned,'returned_complete_paths':len(records),'baseline_nodes':nbase,'trajectory_limit':trajectory_limit},
            'scope':'exact finite-model distributions at theta=0 for p_theta(a|s) proportional to integer weight(s,a)*exp(feature(s,a) dot theta); fixed known transitions/rewards independent of theta; shared-parameter score gradients of expected total undiscounted reward; not empirical trials or learning performance; action-independent deterministic time/state baselines held fixed in the estimator; optimal minimizes Euclidean trace covariance only within this scalar baseline class for reward-to-go, not every unbiased estimator or coordinate separately; full-return is outside that optimization class; theta is not fitted; no large-scale RL or universal value-baseline/return-to-go variance-improvement claim'}

"""Full-rank zero-variance and singular sequential matrix-control fixtures."""
from .linear import audit_linear
from .demo import sequential


def three_actions():
    return {'weights':[[1,2,3]],'features':[[[0,0],[1,0],[0,1]]],
        'transitions':[[[[1,1]],[[1,1]],[[1,1]]]],
        'rewards':[[[[0,1]],[[1,1]],[[2,1]]]],'initial_state':0}


def linear_demo():
    one=audit_linear(three_actions(),horizon=1);seq=audit_linear(sequential(),horizon=3)
    assert one['exact_gradient']==[[-1,9],[1,3]]
    assert one['estimators']['optimal_linear']['trace_variance']==[0,1]
    assert one['baseline_nodes'][0]['score_rank']==2
    assert any(node['score_rank']==1 for node in seq['baseline_nodes'])
    assert seq['baseline_nodes'][1]['score_rank']==0
    return {'format':'scorelab-linear-demo-v1','three_actions':one,'sequential':seq,
        'selection':'Two separately prescribed finite-model audits; no training, timing or empirical performance inference. Matrix controls optimize a larger class than scalar baselines, not every unbiased estimator.'}

from .core import audit


def one_step():
    return {'weights':[[1,4]],'features':[[[0],[1]]],'transitions':[[[[1,1]],[[1,1]]]],'rewards':[[[[0,1]],[[1,1]]]],'initial_state':0}


def sequential():
    return {'weights':[[1,3],[2,1]],'features':[[[0,0],[1,0]],[[0,1],[-1,0]]],
            'transitions':[[[[1,2],[1,2]],[[1,4],[3,4]]],[[[2,3],[1,3]],[[1,3],[2,3]]]],
            'rewards':[[[[0,1],[1,1]],[[1,2],[2,1]]],[[[-1,1],[1,2]],[[2,1],[-1,2]]]],'initial_state':0}


def demo():
    one=audit(one_step(),horizon=1);seq=audit(sequential(),horizon=3)
    assert one['exact_gradient']==[[4,25]]
    assert one['baseline_nodes'][0]['value_baseline']==[4,5] and one['baseline_nodes'][0]['optimal_baseline']==[1,5]
    assert one['estimators']['reward_to_go']['trace_variance']==[4,625]
    assert one['estimators']['value_baseline']['trace_variance']==[36,625]
    assert one['estimators']['optimal_baseline']['trace_variance']==[0,1]
    return {'format':'scorelab-demo-v1','one_step':one,'sequential':seq,'selection':'Prescribed exact finite models, not simulated or learned policy trials. One-step baseline failure is a counterexample, not a universal ranking; the sequential shared-parameter model is a separate audit.'}

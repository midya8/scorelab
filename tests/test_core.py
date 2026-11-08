import copy,itertools,json,math,random,subprocess,sys,tempfile,unittest
from fractions import Fraction as F
from pathlib import Path
from unittest.mock import patch
from scorelab import audit
from scorelab.demo import demo,one_step,sequential


def pack(x):return [x.numerator,x.denominator]
class Dual:
    def __init__(self,v,d=0):self.v=F(v);self.d=F(d)
    def __add__(self,b):
        if not isinstance(b,Dual):b=Dual(b)
        return Dual(self.v+b.v,self.d+b.d)
    __radd__=__add__
    def __mul__(self,b):
        if not isinstance(b,Dual):b=Dual(b)
        return Dual(self.v*b.v,self.d*b.v+self.v*b.d)
    __rmul__=__mul__
    def __truediv__(self,b):
        if not isinstance(b,Dual):b=Dual(b)
        return Dual(self.v/b.v,(self.d*b.v-self.v*b.d)/b.v**2)


def direct_objective(m,h,coordinate=None,theta=None):
    # Full Cartesian action and next-state strings, not production enumerator/DP.
    ns=len(m['weights']);na=len(m['weights'][0]);total=Dual(0) if coordinate is not None else 0.0
    policy=[]
    for s in range(ns):
        raw=[Dual(w,w*m['features'][s][a][coordinate]) if coordinate is not None else w*math.exp(sum(f*t for f,t in zip(m['features'][s][a],theta))) for a,w in enumerate(m['weights'][s])];policy.append([v/sum(raw) for v in raw])
    for actions in itertools.product(range(na),repeat=h):
        for nexts in itertools.product(range(ns),repeat=h):
            p=Dual(1) if coordinate is not None else 1.0;reward=F(0);s=m['initial_state']
            for a,n in zip(actions,nexts):p=p*policy[s][a]*F(*m['transitions'][s][a][n]);reward+=F(*m['rewards'][s][a][n]);s=n
            total=total+p*reward
    return total


def independent_distribution(m,h,baseline,full=False):
    ns=len(m['weights']);na=len(m['weights'][0]);dim=len(m['features'][0][0]);dist=[]
    for actions in itertools.product(range(na),repeat=h):
        for nexts in itertools.product(range(ns),repeat=h):
            p=F(1);rewards=[];scores=[];states=[];s=m['initial_state']
            for a,n in zip(actions,nexts):
                states.append(s);policy=[F(w,sum(m['weights'][s])) for w in m['weights'][s]];p*=policy[a]*F(*m['transitions'][s][a][n]);rewards.append(F(*m['rewards'][s][a][n]));scores.append([F(m['features'][s][a][j])-sum(policy[b]*m['features'][s][b][j] for b in range(na)) for j in range(dim)]);s=n
            g=[sum((scores[t][j]*(sum(rewards if full else rewards[t:])-baseline[t][states[t]]) for t in range(h)),F(0)) for j in range(dim)];dist.append((p,g))
    mean=[sum((p*g[j] for p,g in dist),F(0)) for j in range(dim)];cov=[[sum((p*(g[i]-mean[i])*(g[j]-mean[j]) for p,g in dist),F(0)) for j in range(dim)] for i in range(dim)]
    return mean,cov


class CoreTests(unittest.TestCase):
    def test_256_one_step_bernoulli_analytic_gradient_baseline_and_variance(self):
        count=0
        for w0,w1 in itertools.product(range(1,17),repeat=2):
            m=one_step();m['weights']=[[w0,w1]];r=audit(m,horizon=1);p=F(w1,w0+w1);self.assertEqual(r['exact_gradient'],[pack(p*(1-p))]);self.assertEqual(r['baseline_nodes'][0]['optimal_baseline'],pack(1-p));self.assertEqual(r['baseline_nodes'][0]['value_baseline'],pack(p));self.assertEqual(r['estimators']['optimal_baseline']['trace_variance'],[0,1]);self.assertEqual(r['estimators']['reward_to_go']['trace_variance'],pack(p*(1-p)**3));self.assertEqual(r['estimators']['value_baseline']['trace_variance'],pack(p*(1-p)*(1-2*p)**2));count+=1
        self.assertEqual(count,256)

    def test_value_baseline_nine_times_worse_than_no_baseline_counterexample(self):
        r=demo()['one_step'];self.assertEqual(r['exact_gradient'],[[4,25]]);self.assertEqual(r['estimators']['value_baseline']['trace_variance'],[36,625]);self.assertEqual(r['estimators']['reward_to_go']['trace_variance'],[4,625]);self.assertEqual(r['estimators']['optimal_baseline']['trace_variance'],[0,1]);self.assertEqual([n['gradient_estimates']['optimal_baseline'] for n in r['trajectories']],[[[4,25]],[[4,25]]])

    def test_exact_dual_product_gradient_24_shared_parameter_models(self):
        count=0
        for w0,w1,initial,h in itertools.product([1,3],[1,2],[0,1],[1,2,3]):
            m=sequential();m['weights'][0]=[w0,w1];m['initial_state']=initial;r=audit(m,horizon=h)
            for j in range(2):
                ref=direct_objective(m,h,coordinate=j);self.assertEqual(r['expected_return'],pack(ref.v));self.assertEqual(r['exact_gradient'][j],pack(ref.d))
            count+=1
        self.assertEqual(count,24)

    def test_central_finite_differences_all_shared_coordinates(self):
        m=sequential();r=audit(m,horizon=3);eps=1e-5
        for j in range(2):
            plus=[0.0,0.0];minus=plus[:];plus[j]=eps;minus[j]=-eps;fd=(direct_objective(m,3,theta=plus)-direct_objective(m,3,theta=minus))/(2*eps);self.assertAlmostEqual(fd,float(F(*r['exact_gradient'][j])),places=8)

    def test_all_covariance_entries_against_independent_cartesian_distribution(self):
        m=sequential();r=audit(m,horizon=3);ns=2
        for name in ['full_return','reward_to_go','value_baseline','optimal_baseline']:
            b=[[F(0)]*ns for _ in range(3)]
            if name in ['value_baseline','optimal_baseline']:
                for n in r['baseline_nodes']:b[n['time']][n['state']]=F(*n[name])
            mean,cov=independent_distribution(m,3,b,full=name=='full_return');self.assertEqual(r['estimators'][name]['mean'],list(map(pack,mean)));self.assertEqual(r['estimators'][name]['covariance'],[[pack(v) for v in row] for row in cov])

    def test_every_trajectory_probability_score_rtg_and_covariance_psd(self):
        m=sequential();r=audit(m,horizon=3);self.assertEqual(len(r['trajectories']),64)
        for row in r['trajectories']:
            prob=F(1);state=m['initial_state'];reward=F(0)
            for t,step in enumerate(row['steps']):
                self.assertEqual(step['state'],state);a=step['action'];n=step['next_state'];prob*=F(m['weights'][state][a],sum(m['weights'][state]))*F(*m['transitions'][state][a][n]);reward+=F(*step['reward']);self.assertEqual(step['return_to_go'],pack(sum((F(*x['reward']) for x in row['steps'][t:]),F(0))));self.assertEqual(step['score'],r['score_vectors'][state][a]);state=n
            self.assertEqual(row['probability'],pack(prob));self.assertEqual(row['return'],pack(reward))
        self.assertEqual(sum(F(*n['probability']) for n in r['trajectories']),1)
        for v in r['estimators'].values():
            c=[[F(*x) for x in row] for row in v['covariance']];self.assertEqual(c[0][1],c[1][0]);self.assertGreaterEqual(c[0][0],0);self.assertGreaterEqual(c[1][1],0);self.assertGreaterEqual(c[0][0]*c[1][1]-c[0][1]**2,0)

    def test_orthogonal_control_variates_and_exact_baseline_normal_equations(self):
        r=audit(sequential(),horizon=3);gram=r['control_variate_gram']
        for i,n in enumerate(r['baseline_nodes']):
            self.assertEqual(gram[i][i],n['score_energy']);self.assertEqual(F(*n['score_energy'])*F(*n['optimal_baseline']),F(*n['score_gradient_correlation']))
            for j in range(len(gram)):
                if i!=j:self.assertEqual(gram[i][j],[0,1])
        self.assertFalse(r['baseline_nodes'][1]['identifiable']);self.assertEqual(r['baseline_nodes'][1]['optimal_baseline'],[0,1]);self.assertNotEqual(r['baseline_nodes'][1]['value_baseline'],[0,1])

    def test_custom_baseline_excess_variance_identity_and_unvisited_invariance(self):
        m=sequential();r=audit(m,horizon=3,baseline=[[[0,1],[7,1]],[[1,1],[2,1]],[[3,1],[-1,1]]]);b=[0,7,1,2,3,-1];excess=sum(F(*n['score_energy'])*(b[i]-F(*n['optimal_baseline']))**2 for i,n in enumerate(r['baseline_nodes']));self.assertEqual(r['estimators']['custom_baseline']['excess_trace_variance_over_optimal'],pack(excess))
        baseline=[[[0,1],[0,1]],[[1,1],[2,1]],[[3,1],[-1,1]]];other=audit(m,horizon=3,baseline=baseline);self.assertEqual(other['estimators']['custom_baseline'],r['estimators']['custom_baseline'])
        mean,cov=independent_distribution(m,3,[[F(v) for v in row] for row in [[0,7],[1,2],[3,-1]]]);self.assertEqual(r['estimators']['custom_baseline']['covariance'],[[pack(x) for x in row] for row in cov])

    def test_zero_horizon_single_action_and_zero_score_directions(self):
        r=audit(one_step(),horizon=0,baseline=[]);self.assertEqual(r['expected_return'],[0,1]);self.assertEqual(r['exact_gradient'],[[0,1]]);self.assertEqual(r['counts']['returned_complete_paths'],1);self.assertEqual(r['trajectories'][0]['steps'],[])
        for m in [one_step(),{'weights':[[1]],'features':[[[3,-2,1]]],'transitions':[[[[1,1]]]],'rewards':[[[[2,1]]]],'initial_state':0}]:
            m['features']=[[m['features'][0][0][:] for _ in m['weights'][0]]];r=audit(m,horizon=3);self.assertTrue(all(v==[0,1] for v in r['exact_gradient']));self.assertTrue(all(v['trace_variance']==[0,1] for v in r['estimators'].values()));self.assertTrue(all(not n['identifiable'] for n in r['baseline_nodes']))

    def test_exact_path_count_and_preflight_before_enumeration(self):
        r=audit(sequential(),horizon=3,trajectory_limit=64);self.assertEqual(r['counts']['preflight_complete_paths'],64)
        with patch('scorelab.core._enumerate',side_effect=AssertionError('enumerated')):
            with self.assertRaisesRegex(ValueError,'preflight refused'):audit(sequential(),horizon=3,trajectory_limit=63)
        m=sequential();m['transitions']=[[[[1,1],[0,1]],[[1,1],[0,1]]],[[[0,1],[1,1]],[[0,1],[1,1]]]];r=audit(m,horizon=6,trajectory_limit=64);self.assertEqual(r['counts']['preflight_complete_paths'],64);self.assertEqual(r['counts']['returned_complete_paths'],64)

    def test_three_states_actions_dimensions_and_signed_rewards(self):
        m={'weights':[[1,2,3]]*3,'features':[[[1,0,0],[0,1,0],[0,0,1]]]*3,'transitions':[[[[1,3]]*3 for _ in range(3)] for _ in range(3)],'rewards':[[[[s-a,1] for _ in range(3)] for a in range(3)] for s in range(3)],'initial_state':1};r=audit(m,horizon=2);self.assertEqual(r['counts']['returned_complete_paths'],81)
        for j in range(3):self.assertEqual(r['exact_gradient'][j],pack(direct_objective(m,2,coordinate=j).d))

    def test_input_output_rng_and_repeated_call_isolation(self):
        m=sequential();before=copy.deepcopy(m);state=random.getstate();r=audit(m);expected=copy.deepcopy(r);r['model']['weights'][0][0]=999;r['baseline_nodes'][0]['optimal_baseline'][0]=999;r['trajectories'][0]['steps'][0]['score'][0][0]=999;self.assertEqual(m,before);self.assertEqual(state,random.getstate());self.assertEqual(audit(m),expected)

    def test_statewise_feature_shifts_leave_scores_and_audit_unchanged(self):
        m=sequential();r=audit(m,horizon=3);other=copy.deepcopy(m)
        for state in other['features']:
            for v in state:v[0]+=3;v[1]-=2
        q=audit(other,horizon=3)
        for key in ['policy_probabilities','score_vectors','expected_return','exact_gradient','estimators','baseline_nodes','control_variate_gram','trajectories']:self.assertEqual(r[key],q[key])

    def test_constant_reward_shift_preserves_gradient_and_adjusted_baseline_variance(self):
        m=sequential();r=audit(m,horizon=3);other=copy.deepcopy(m)
        for state in other['rewards']:
            for action in state:
                for pair in action:pair[0]+=pair[1]
        q=audit(other,horizon=3);self.assertEqual(q['exact_gradient'],r['exact_gradient']);self.assertEqual(F(*q['expected_return']),F(*r['expected_return'])+3)
        for name in ['value_baseline','optimal_baseline']:self.assertEqual(q['estimators'][name]['covariance'],r['estimators'][name]['covariance'])
        for a,b in zip(r['baseline_nodes'],q['baseline_nodes']):
            if a['identifiable']:self.assertEqual(F(*b['optimal_baseline']),F(*a['optimal_baseline'])+3-a['time'])

    def test_validation_strict_model_and_shapes(self):
        bad=[None,{},[],dict(one_step(),extra=1)]
        for key,vals in {'weights':[None,[],[[True,1]],[[0,1]],[[17,1]],[[1],[1,2]],[[1.0,1]]], 'features':[[],[[[]]],[[[True],[1]]],[[[9],[0]]],[[[0],[1,2]]]],'transitions':[[],[[[[1,2]],[[1,1]]]],[[[[True,1]],[[1,1]]]],[[[[-1,1]],[[1,1]]]]],'rewards':[None,[[[[1,0]],[[0,1]]]],[[[[33,1]],[[0,1]]]]],'initial_state':[True,-1,1,0.0]}.items():
            for v in vals:m=one_step();m[key]=v;bad.append(m)
        for m in bad:
            with self.assertRaises(ValueError):audit(m)
        for h in [True,-1,7,1.0,None]:
            with self.assertRaises(ValueError):audit(one_step(),horizon=h)
        for c in [True,0,8193,1.5,None]:
            with self.assertRaises(ValueError):audit(one_step(),trajectory_limit=c)

    def test_custom_baseline_validation(self):
        for b in [[],{},[[]],[[[True,1]]],[[[1,0]]],[[[33,1]]],[[[1.0,2]]],[[[1,17]]]]:
            with self.assertRaises(ValueError):audit(one_step(),horizon=1,baseline=b)

    def test_cli_matches_api_and_demo(self):
        with tempfile.TemporaryDirectory() as d:
            p=Path(d)/'input.json'
            for custom in [False,True]:
                x={'version':1,'model':one_step(),'horizon':1}
                if custom:x['baseline']=[[[1,5]]]
                p.write_text(json.dumps(x));run=subprocess.run([sys.executable,'-m','scorelab','audit',str(p)],capture_output=True,text=True,check=True);self.assertEqual(json.loads(run.stdout),audit(x['model'],horizon=1,baseline=x.get('baseline')))
            run=subprocess.run([sys.executable,'-m','scorelab','demo'],capture_output=True,text=True,check=True);self.assertEqual(json.loads(run.stdout),demo())

    def test_cli_errors_no_partial_output(self):
        with tempfile.TemporaryDirectory() as d:
            p=Path(d)/'input.json'
            for raw in ['{','{"version":1,"version":1}',' '*65537,json.dumps({'version':True,'model':one_step(),'horizon':1}),json.dumps({'version':1,'model':one_step(),'horizon':1,'extra':0})]:
                p.write_text(raw);run=subprocess.run([sys.executable,'-m','scorelab','audit',str(p)],capture_output=True,text=True);self.assertEqual(run.returncode,2);self.assertEqual(run.stdout,'');self.assertIn('error:',run.stderr)
            p.unlink();run=subprocess.run([sys.executable,'-m','scorelab','audit',str(p)],capture_output=True,text=True);self.assertEqual(run.returncode,2);self.assertEqual(run.stdout,'')

if __name__=='__main__':unittest.main()

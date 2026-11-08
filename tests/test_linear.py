import copy,itertools,json,random,subprocess,sys,tempfile,unittest
from fractions import Fraction as F
from pathlib import Path
from scorelab.linear import audit_linear,solve
from scorelab.linear_demo import three_actions,linear_demo
from scorelab.demo import sequential,one_step
from scorelab import audit


def unpack(a):return [[F(*v) for v in row] for row in a]
def matmul(a,b):return [[sum((a[i][k]*b[k][j] for k in range(len(b))),F(0)) for j in range(len(b[0]))] for i in range(len(a))]
def trans(a):return [list(x) for x in zip(*a)]
def determinant(a):
    n=len(a);return sum(((-1)**sum(p[i]>p[j] for i in range(n) for j in range(i+1,n))*prod(a[i][p[i]] for i in range(n)) for p in itertools.permutations(range(n))),F(0))
def prod(xs):
    v=F(1)
    for x in xs:v*=x
    return v


def direct_paths(m,h):
    # Independent Cartesian action/next-state strings, including zero mass.
    ns=len(m['weights']);na=len(m['weights'][0]);d=len(m['features'][0][0]);rows=[]
    for actions in itertools.product(range(na),repeat=h):
        for nexts in itertools.product(range(ns),repeat=h):
            p=F(1);s=m['initial_state'];zs=[[F(0)]*d for _ in range(h*ns)];rewards=[];scores=[]
            for t,(a,n) in enumerate(zip(actions,nexts)):
                ps=[F(w,sum(m['weights'][s])) for w in m['weights'][s]];p*=ps[a]*F(*m['transitions'][s][a][n]);score=[F(m['features'][s][a][j])-sum(ps[b]*m['features'][s][b][j] for b in range(na)) for j in range(d)];zs[t*ns+s]=score;scores.append(score);rewards.append(F(*m['rewards'][s][a][n]));s=n
            g=[sum((scores[t][j]*sum(rewards[t:]) for t in range(h)),F(0)) for j in range(d)];rows.append((p,g,zs))
    return rows


class LinearTests(unittest.TestCase):
    def assert_psd(self,a):
        self.assertEqual(a,trans(a))
        for size in range(1,len(a)+1):
            for ix in itertools.combinations(range(len(a)),size):self.assertGreaterEqual(determinant([[a[i][j] for j in ix] for i in ix]),0)

    def test_full_rank_three_action_analytic_coefficients_and_constant_estimates(self):
        r=audit_linear(three_actions(),horizon=1);n=r['baseline_nodes'][0]
        self.assertEqual(r['exact_gradient'],[[-1,9],[1,3]]);self.assertEqual(n['score_rank'],2);self.assertEqual(n['coefficient_nullity_per_row'],0)
        self.assertEqual(n['optimal_matrix_control'],[[[2,3],[-2,3]],[[-1,2],[1,1]]]);self.assertEqual(n['optimal_baseline'],[64,51])
        self.assertEqual(r['estimators']['optimal_baseline']['trace_variance'],[53,612]);self.assertEqual(r['estimators']['optimal_linear']['covariance'],[[[0,1],[0,1]],[[0,1],[0,1]]])
        self.assertEqual([x['gradient_estimates']['optimal_linear'] for x in r['trajectories']],[r['exact_gradient']]*3)
        # Hand solution: for each action, B*score = score*reward - true gradient.
        b=[[F(2,3),F(-2,3)],[F(-1,2),F(1)]]
        for p,g,z in direct_paths(three_actions(),1):self.assertEqual([g[i]-sum(b[i][j]*z[0][j] for j in range(2)) for i in range(2)],[F(-1,9),F(1,3)])

    def test_independent_cross_moments_gram_and_covariance_15_models(self):
        models=[]
        for h,w,start in itertools.product([1,2,3],[1,3],[0,1]):
            m=sequential();m['weights'][0][0]=w;m['initial_state']=start;models.append((m,h))
        models += [(three_actions(),h) for h in [1,2,3]]
        self.assertEqual(len(models),15)
        for m,h in models:
            r=audit_linear(m,horizon=h);paths=direct_paths(m,h);d=len(m['features'][0][0]);n=h*len(m['weights']);flat=[(p,g,[x for z in zs for x in z]) for p,g,zs in paths]
            gram=[[sum((p*z[i]*z[j] for p,g,z in flat),F(0)) for j in range(n*d)] for i in range(n*d)];self.assertEqual(unpack(r['coordinate_control_gram']),gram)
            for node in range(n):
                s=[[sum((p*z[node][i]*z[node][j] for p,g,z in paths),F(0)) for j in range(d)] for i in range(d)];c=[[sum((p*g[i]*z[node][j] for p,g,z in paths),F(0)) for j in range(d)] for i in range(d)];self.assertEqual(unpack(r['baseline_nodes'][node]['score_second_moment']),s);self.assertEqual(unpack(r['baseline_nodes'][node]['gradient_score_cross_moment']),c)
                b=unpack(r['baseline_nodes'][node]['optimal_matrix_control']);self.assertEqual(matmul(b,s),c)
            samples=[]
            for p,g,z in paths:
                bs=[unpack(node['optimal_matrix_control']) for node in r['baseline_nodes']];v=[g[i]-sum(bs[k][i][j]*z[k][j] for k in range(n) for j in range(d)) for i in range(d)];samples.append((p,v,z))
            mu=[sum(p*v[i] for p,v,z in samples) for i in range(d)];cov=[[sum(p*(v[i]-mu[i])*(v[j]-mu[j]) for p,v,z in samples) for j in range(d)] for i in range(d)];self.assertEqual(unpack(r['estimators']['optimal_linear']['covariance']),cov);self.assert_psd(cov)
            for k in range(n):
                for i,j in itertools.product(range(d),repeat=2):self.assertEqual(sum(p*v[i]*z[k][j] for p,v,z in samples),0)
            for name,cert in r['linear_covariance_certificates'].items():
                gap=unpack(cert['covariance_gap']);self.assert_psd(gap);candidate=unpack(r['estimators'][name]['covariance']);self.assertEqual(gap,[[candidate[i][j]-cov[i][j] for j in range(d)] for i in range(d)])
                for term in cert['node_psd_terms']:self.assert_psd(unpack(term))

    def test_arbitrary_matrix_controls_exact_excess_covariance_and_cartesian_oracle(self):
        m=sequential();control=[[[[[i-j+t+s,2] for j in range(2)] for i in range(2)] for s in range(2)] for t in range(3)];r=audit_linear(m,horizon=3,control=control);paths=direct_paths(m,3);b=[unpack(x) for layer in control for x in layer];samples=[]
        for p,g,z in paths:samples.append((p,[g[i]-sum(b[k][i][j]*z[k][j] for k in range(6) for j in range(2)) for i in range(2)]))
        mu=[sum(p*v[i] for p,v in samples) for i in range(2)];cov=[[sum(p*(v[i]-mu[i])*(v[j]-mu[j]) for p,v in samples) for j in range(2)] for i in range(2)];self.assertEqual(unpack(r['estimators']['custom_linear']['covariance']),cov);self.assert_psd(unpack(r['linear_covariance_certificates']['custom_linear']['covariance_gap']))
        terms=[unpack(x) for x in r['linear_covariance_certificates']['custom_linear']['node_psd_terms']];best=unpack(r['estimators']['optimal_linear']['covariance']);self.assertEqual([[sum(term[i][j] for term in terms) for j in range(2)] for i in range(2)],[[cov[i][j]-best[i][j] for j in range(2)] for i in range(2)])

    def test_singular_three_coordinate_nonunique_coefficients_unique_correction(self):
        m=three_actions();m['features']=[[[0,0,0],[1,0,1],[0,1,1]]];r=audit_linear(m,horizon=1);node=r['baseline_nodes'][0];self.assertEqual(node['score_rank'],2);self.assertEqual(node['coefficient_nullity_per_row'],1);self.assertEqual(r['estimators']['optimal_linear']['trace_variance'],[0,1]);b=unpack(node['optimal_matrix_control']);other=[[v+offset*delta for v,delta in zip(row,[1,1,-1])] for offset,row in enumerate(b,1)]
        for p,g,z in direct_paths(m,1):self.assertEqual(matmul(b,trans(z)),matmul(other,trans(z)))
        self.assertEqual(matmul(other,unpack(node['score_second_moment'])),unpack(node['gradient_score_cross_moment']))
        for cert in r['linear_covariance_certificates'].values():
            self.assert_psd(unpack(cert['covariance_gap']))
            for term in cert['node_psd_terms']:self.assert_psd(unpack(term))

    def test_invertible_reparameterization_covariance_congruence(self):
        m=sequential();q=copy.deepcopy(m);a=[[1,1],[0,1]]
        for state in q['features']:
            for v in state:v[:]=[v[0]+v[1],v[1]]
        r=audit_linear(m,horizon=3);s=audit_linear(q,horizon=3);self.assertEqual(unpack(s['estimators']['optimal_linear']['covariance']),matmul(matmul(a,unpack(r['estimators']['optimal_linear']['covariance'])),trans(a)))
        for p1,p2 in zip(r['trajectories'],s['trajectories']):self.assertEqual([[F(*x)] for x in p2['gradient_estimates']['optimal_linear']],matmul(a,[[F(*x)] for x in p1['gradient_estimates']['optimal_linear']]))

    def test_one_dimension_equals_original_scalar_baseline(self):
        for h in [0,1,2,3]:
            r=audit_linear(one_step(),horizon=h);self.assertEqual(r['estimators']['optimal_baseline']['covariance'],r['estimators']['optimal_linear']['covariance'])
            self.assertEqual(r['linear_covariance_certificates']['optimal_baseline']['trace_gap'],[0,1])

    def test_horizon_zero_unvisited_zero_scores_and_single_action(self):
        r=audit_linear(three_actions(),horizon=0,control=[]);self.assertEqual(r['coordinate_control_gram'],[]);self.assertEqual(r['baseline_nodes'],[]);self.assertEqual(r['estimators']['optimal_linear']['trace_variance'],[0,1]);self.assertEqual(len(r['trajectories']),1)
        r=audit_linear(sequential(),horizon=3);self.assertEqual(r['baseline_nodes'][1]['score_rank'],0);self.assertEqual(r['baseline_nodes'][1]['coefficient_nullity_per_row'],2)
        m=three_actions();m['features']=[[[1,1]]*3];r=audit_linear(m,horizon=2);self.assertTrue(all(n['score_rank']==0 for n in r['baseline_nodes']));self.assertEqual(r['estimators']['optimal_linear']['trace_variance'],[0,1])
        m=one_step();m['weights']=[[1]];m['features']=[[[1,2,3]]];m['transitions']=[[[[1,1]]]];m['rewards']=[[[[2,1]]]];r=audit_linear(m,horizon=3);self.assertTrue(all(n['score_rank']==0 for n in r['baseline_nodes']))

    def test_rref_pivot_patterns_and_inconsistent_system(self):
        for a,x,expected in [([[0,0],[0,0]],[0,0],[0,0]),([[0,0],[0,2]],[0,4],[0,2]),([[1,1],[1,1]],[3,3],[3,0]),([[2,1],[1,1]],[3,2],[1,1])]:
            got,rank=solve([[F(v) for v in row] for row in a],list(map(F,x)));self.assertEqual(got,expected);self.assertEqual(rank,0 if not any(any(row) for row in a) else 2 if determinant(a) else 1)
        with self.assertRaisesRegex(ValueError,'inconsistent'):solve([[F(0)]],[F(1)])

    def test_strict_custom_control_validation_and_preflight(self):
        for c in [[],{},[[[]]],[[[[[True,1],[0,1]],[[0,1],[1,1]]]]],[[[[[1,0],[0,1]],[[0,1],[1,1]]]]],[[[[[33,1],[0,1]],[[0,1],[1,1]]]]],[[[[[1.0,1],[0,1]],[[0,1],[1,1]]]]]]:
            with self.assertRaises(ValueError):audit_linear(three_actions(),horizon=1,control=c)
        with self.assertRaisesRegex(ValueError,'preflight'):audit_linear(sequential(),horizon=3,trajectory_limit=63)

    def test_input_output_rng_and_original_api_isolation(self):
        m=three_actions();before=copy.deepcopy(m);base=audit(m,horizon=1);state=random.getstate();r=audit_linear(m,horizon=1);r['baseline_nodes'][0]['optimal_matrix_control'][0][0][0]=999;r['trajectories'][0]['gradient_estimates']['optimal_linear'][0][0]=999;self.assertEqual(m,before);self.assertEqual(random.getstate(),state);self.assertEqual(audit(m,horizon=1),base);self.assertNotEqual(r,audit_linear(m,horizon=1))

    def test_linear_cli_demo_and_input_and_no_partial_errors(self):
        run=subprocess.run([sys.executable,'-m','scorelab','linear-demo'],capture_output=True,text=True,check=True);self.assertEqual(json.loads(run.stdout),linear_demo())
        with tempfile.TemporaryDirectory() as d:
            path=Path(d)/'m.json';x={'version':1,'model':three_actions(),'horizon':1,'control':[[[[[1,1],[0,1]],[[0,1],[1,1]]]]]};path.write_text(json.dumps(x));run=subprocess.run([sys.executable,'-m','scorelab','linear',str(path)],capture_output=True,text=True,check=True);self.assertEqual(json.loads(run.stdout),audit_linear(x['model'],horizon=1,control=x['control']))
            for raw in ['{"version":1,"version":1}',' '*65537,json.dumps(dict(x,baseline=[])),json.dumps(dict(x,control=[]))]:
                path.write_text(raw);run=subprocess.run([sys.executable,'-m','scorelab','linear',str(path)],capture_output=True,text=True);self.assertEqual(run.returncode,2);self.assertEqual(run.stdout,'')

if __name__=='__main__':unittest.main()

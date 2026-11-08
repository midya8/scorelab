"""Reproducible matrix-control figure with complete exact audit metadata."""
import argparse,base64,html,json,zlib
from fractions import Fraction as F
from pathlib import Path
from scorelab.linear_demo import linear_demo


def render(d):
    if d!=linear_demo():raise ValueError('requires complete unchanged linear demo')
    out=['<svg xmlns="http://www.w3.org/2000/svg" width="1120" height="760" viewBox="0 0 1120 760">','<rect width="1120" height="760" fill="#101827"/>','<style>text{font-family:Arial,sans-serif;fill:#e7edf8}</style>','<metadata data-encoding="zlib+base64">'+base64.b64encode(zlib.compress(json.dumps(d,sort_keys=True,separators=(',',':')).encode())).decode()+'</metadata>']
    def text(x,y,s,size=17,color='#e7edf8'):
        out.append(f'<text x="{float(x):.8f}" y="{float(y):.8f}" font-size="{size}" style="fill:{color}">'+html.escape(s)+'</text>')
    text(30,44,'ScoreLab | matrix score controls go beyond a scalar baseline',26)
    text(30,80,'Exact finite-model distributions. Same correct mean; a larger control class.',18)
    text(55,126,'One step, three actions, two score coordinates',19)
    text(55,156,'Action probabilities: 1/6, 1/3, 1/2. Rewards: 0, 1, 2.',16,'#b7c4d9')
    for v in [F(0),F(3,10),F(6,10),F(9,10)]:
        y=420-270*v;out.append(f'<path d="M 70 {float(y):.8f} H 550" stroke="#33445e"/>');text(28,y+5,str(float(v)),14,'#b7c4d9')
    names=[('reward_to_go','None'),('value_baseline','Value'),('optimal_baseline','Scalar'),('optimal_linear','Matrix')]
    for i,(name,label) in enumerate(names):
        v=F(*d['three_actions']['estimators'][name]['trace_variance']);x=94+119*i;height=270*v;color='#63d8bc' if name=='optimal_linear' else '#e9bb79' if name=='optimal_baseline' else '#98a8c0'
        out.append(f'<rect data-estimator="{name}" x="{x}" y="{float(420-height):.8f}" width="48" height="{float(height):.8f}" fill="{color}"/>');text(x-2,449,label,16);text(x-2,408-height,f'{float(v):.4f}',16,color)
    text(55,482,'Vertical axis: trace covariance (gradient variance sum).',16,'#b7c4d9')
    text(610,126,'A full matrix, not one number per node',19)
    text(610,168,'G_B = G_RTG - B Z',21,'#63d8bc')
    text(610,216,'B = [  2/3    -2/3 ]',23)
    text(610,251,'       [ -1/2       1 ]',23)
    text(610,295,'Every action gives the exact gradient:',17)
    text(610,332,'[-1/9, 1/3]',25,'#63d8bc')
    text(610,376,'Covariance = the zero 2 x 2 matrix.',18)
    text(610,414,'Best scalar trace: 53/612. Best matrix: 0.',17)
    text(610,452,'Supplied fixture, not a universal zero-variance claim.',15,'#b7c4d9')
    text(30,530,'Singular controls are handled without a fake inverse',22)
    text(30,568,'Separate 64-path, three-step model: node score ranks = [1, 0, 1, 1, 1, 1].',17)
    text(30,600,'Scalar-optimal trace: 1.462519. Matrix-optimal trace: 1.460918. This improvement is small.',17)
    text(30,638,'B* S = C, solved by exact rational RREF. Cov(B) - Cov(B*) = sum (B-B*) S (B-B*)^T.',17,'#63d8bc')
    text(30,674,'Each gap term is positive semidefinite. Singular coefficients can differ; supported corrections cannot.',16)
    text(30,712,'Optimal only within fixed time/state matrix-score controls for reward-to-go. No fitted critic or training trials.',16,'#b7c4d9')
    text(30,744,'Full exact distributions, normal equations and covariance certificates are embedded in SVG metadata.',16,'#b7c4d9')
    return '\n'.join(out+['</svg>'])+'\n'

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--check',action='store_true');args=p.parse_args();value=render(linear_demo());path=Path(__file__).resolve().parents[1]/'assets'/'linear-audit.svg'
    if args.check:
        if path.read_text()!=value:raise SystemExit('SVG differs')
    else:path.write_text(value)

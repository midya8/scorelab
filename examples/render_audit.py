"""Two finite-model gradient variance panels with full exact metadata."""
import argparse,base64,html,json,zlib
from fractions import Fraction as F
from pathlib import Path
from scorelab.demo import demo


def render(d):
    if d!=demo():raise ValueError('requires complete unchanged score demo')
    out=['<svg xmlns="http://www.w3.org/2000/svg" width="1120" height="780" viewBox="0 0 1120 780">','<rect width="1120" height="780" fill="#101827"/>','<style>text{font-family:Arial,sans-serif;fill:#e7edf8}</style>','<metadata data-encoding="zlib+base64">'+base64.b64encode(zlib.compress(json.dumps(d,sort_keys=True,separators=(',',':')).encode())).decode()+'</metadata>']
    def text(x,y,s,size=16,color='#e7edf8'):
        out.append(f'<text x="{coord(x)}" y="{coord(y)}" font-size="{size}" style="fill:{color}">'+html.escape(s)+'</text>')
    def coord(x):return format(float(x),'.9f').rstrip('0').rstrip('.')
    text(30,44,'ScoreLab | a correct value baseline can make gradients noisier',26)
    text(30,80,'Exact finite-policy distributions. All estimators have the same correct mean gradient.',18)
    text(65,127,'One-step Bernoulli policy: p(action 1) = 4/5',19)
    text(610,127,'Separate three-step, two-state shared policy',19)
    text(65,158,'Scalar gradient variance; reward = action',17,'#b7c4d9')
    text(610,158,'Trace covariance; two shared parameters',17,'#b7c4d9')
    for value,label in [(F(0),'0'),(F(3,100),'0.03'),(F(6,100),'0.06')]:
        y=455-4000*value;out.append(f'<path d="M 80 {coord(y)} H 505" stroke="#33445e"/>');text(32,y+5,label,15,'#b7c4d9')
    for value in [0,3,6]:
        y=455-40*value;out.append(f'<path d="M 655 {y} H 1090" stroke="#33445e"/>');text(622,y+5,str(value),15,'#b7c4d9')
    panels=[('one_step',[('reward_to_go','None'),('value_baseline','Value'),('optimal_baseline','Optimal')],125,135,4000),('sequential',[('full_return','Full'),('reward_to_go','RTG'),('value_baseline','Value'),('optimal_baseline','Optimal')],675,105,40)]
    for model,items,start,gap,scale in panels:
        for i,(name,label) in enumerate(items):
            v=F(*d[model]['estimators'][name]['trace_variance']);x=start+i*gap;height=scale*v;color='#63d8bc' if name=='optimal_baseline' else '#e9bb79' if name=='value_baseline' else '#98a8c0'
            out.append(f'<rect data-model="{model}" data-estimator="{name}" x="{x}" y="{coord(455-height)}" width="48" height="{coord(height)}" fill="{color}"/>');text(x-3,488,label,17);text(x-3,445-height,('0' if not v else f'{float(v):.4f}'),16,color)
    text(65,535,'No baseline: 4/625. Value: 36/625. Optimal: 0.',17)
    text(65,568,'The value baseline is NINE times noisier here.',18,'#e9bb79')
    text(65,600,'Value baseline 4/5; variance-minimizing baseline 1/5.',16)
    text(610,535,'Full = full return; RTG = reward-to-go, no baseline.',16)
    text(610,568,'Optimal minimizes trace variance within scalar',17,'#63d8bc')
    text(610,596,'time/state baselines for the RTG estimator only.',17,'#63d8bc')
    text(30,645,'One-step true gradient: 4/25. The optimal-baseline estimator equals 4/25 for BOTH actions.',18)
    text(30,680,'Complete path enumeration + exact rational means, covariances, score energies and baseline normal equations.',17)
    text(30,718,'Separate models and separate vertical scales. Labels are rounded; full exact fractions are in SVG metadata.',16,'#b7c4d9')
    text(30,754,'Not training trials, empirical confidence intervals, a universal baseline ranking or a large-scale RL method.',16,'#b7c4d9')
    return '\n'.join(out+['</svg>'])+'\n'

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--check',action='store_true');a=p.parse_args();s=render(demo());path=Path(__file__).resolve().parents[1]/'assets'/'score-audit.svg'
    if a.check:
        if path.read_text()!=s:raise SystemExit('SVG differs')
    else:path.parent.mkdir(exist_ok=True);path.write_text(s)

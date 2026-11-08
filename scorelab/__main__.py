import argparse,json
from .core import audit
from .demo import demo
from .linear import audit_linear
from .linear_demo import linear_demo


def pairs(items):
    out={}
    for k,v in items:
        if k in out:raise ValueError('duplicate JSON key: '+k)
        out[k]=v
    return out


def main(argv=None):
    p=argparse.ArgumentParser(description='Exact finite stochastic-policy gradient variance audits');s=p.add_subparsers(dest='command',required=True);s.add_parser('demo');s.add_parser('linear-demo')
    for command in ['audit','linear']:
        a=s.add_parser(command);a.add_argument('input');a.add_argument('--trajectory-limit',type=int,default=4096)
    args=p.parse_args(argv)
    try:
        if args.command=='demo':out=demo()
        elif args.command=='linear-demo':out=linear_demo()
        else:
            with open(args.input,'rb') as f:raw=f.read(65537)
            if len(raw)>65536:raise ValueError('input JSON exceeds 64 KiB')
            x=json.loads(raw,object_pairs_hook=pairs)
            optional='control' if args.command=='linear' else 'baseline'
            if type(x) is not dict or set(x) not in (set(['version','model','horizon']),set(['version','model','horizon',optional])) or type(x['version']) is not int or x['version']!=1:raise ValueError('requires version=1, model, horizon and optional '+optional+', with no extra keys')
            if args.command=='linear':out=audit_linear(x['model'],horizon=x['horizon'],trajectory_limit=args.trajectory_limit,control=x.get('control'))
            else:out=audit(x['model'],horizon=x['horizon'],trajectory_limit=args.trajectory_limit,baseline=x.get('baseline'))
        print(json.dumps(out,indent=2,allow_nan=False))
    except (ValueError,TypeError,OSError,OverflowError) as e:p.exit(2,'error: '+str(e)+'\n')

if __name__=='__main__':main()

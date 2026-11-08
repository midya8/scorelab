import base64,copy,json,unittest,zlib,xml.etree.ElementTree as ET
from fractions import Fraction as F
from pathlib import Path
from scorelab.demo import demo
from examples.render_audit import render
NS={'s':'http://www.w3.org/2000/svg'}

class FigureTests(unittest.TestCase):
    def test_all7_bars_exact_geometry_under_labeled_separate_scales(self):
        d=demo();root=ET.fromstring(render(d));bars=[n for n in root.findall('s:rect',NS) if 'data-model' in n.attrib];self.assertEqual(len(bars),7)
        for label in root.findall('s:text',NS):float(label.attrib['x']);float(label.attrib['y'])
        for n in bars:
            model=n.attrib['data-model'];name=n.attrib['data-estimator'];items=['reward_to_go','value_baseline','optimal_baseline'] if model=='one_step' else ['full_return','reward_to_go','value_baseline','optimal_baseline'];i=items.index(name);start,gap,scale=(125,135,4000) if model=='one_step' else (675,105,40);v=F(*d[model]['estimators'][name]['trace_variance']);self.assertEqual(int(n.attrib['x']),start+i*gap);self.assertAlmostEqual(float(n.attrib['height']),float(scale*v),places=8);self.assertAlmostEqual(float(n.attrib['y']),float(455-scale*v),places=8)
    def test_full_metadata_and_scope(self):
        root=ET.fromstring(render(demo()));self.assertEqual(json.loads(zlib.decompress(base64.b64decode(root.find('s:metadata',NS).text))),demo());text=' '.join(root.itertext());self.assertIn('separate vertical scales',text);self.assertIn('within scalar',text);self.assertIn('not',text.lower())
    def test_regeneration(self):self.assertEqual((Path(__file__).resolve().parents[1]/'assets'/'score-audit.svg').read_text(),render(demo()))
    def test_undisplayed_trajectory_covariance_tamper_rejected(self):
        for key in ['trajectory','covariance']:
            d=copy.deepcopy(demo())
            if key=='trajectory':d['sequential']['trajectories'][0]['steps'][0]['return_to_go'][0]+=1
            else:d['sequential']['estimators']['full_return']['covariance'][0][1][0]+=1
            with self.assertRaises(ValueError):render(d)

import base64,copy,importlib.util,json,unittest,xml.etree.ElementTree as ET,zlib
from fractions import Fraction as F
from pathlib import Path
from scorelab.linear_demo import linear_demo
P=Path(__file__).resolve().parents[1]
spec=importlib.util.spec_from_file_location('render_linear',P/'examples'/'render_linear.py');mod=importlib.util.module_from_spec(spec);spec.loader.exec_module(mod)

class LinearFigureTests(unittest.TestCase):
    def test_complete_exact_metadata_and_byte_regeneration(self):
        d=linear_demo();s=mod.render(d);self.assertEqual(s,(P/'assets'/'linear-audit.svg').read_text());root=ET.fromstring(s);metadata=root.find('{http://www.w3.org/2000/svg}metadata');self.assertEqual(json.loads(zlib.decompress(base64.b64decode(metadata.text))),d)

    def test_every_bar_position_height_and_numeric_label_coordinates(self):
        d=linear_demo();root=ET.fromstring(mod.render(d));bars=[x for x in root.iter() if x.tag.endswith('rect') and x.get('data-estimator')];self.assertEqual(len(bars),4)
        for i,b in enumerate(bars):
            v=F(*d['three_actions']['estimators'][b.get('data-estimator')]['trace_variance']);self.assertEqual(float(b.get('x')),94+119*i);self.assertAlmostEqual(float(b.get('height')),float(270*v),places=7);self.assertAlmostEqual(float(b.get('y')),float(420-270*v),places=7)
        for t in root.iter():
            if t.tag.endswith('text'):self.assertTrue(0<=float(t.get('x'))<1120);self.assertTrue(0<float(t.get('y'))<760)
        labels=[t.text for t in root.iter() if t.tag.endswith('text')];self.assertIn('0.8302',labels);self.assertIn('0.0866',labels);self.assertIn('0.0000',labels)

    def test_refuse_changed_undisplayed_path_or_covariance(self):
        for key in ['path','cov']:
            d=linear_demo()
            if key=='path':d['sequential']['trajectories'][0]['probability'][0]+=1
            else:d['sequential']['linear_covariance_certificates']['value_baseline']['node_psd_terms'][0][0][0][0]+=1
            with self.assertRaises(ValueError):mod.render(d)

if __name__=='__main__':unittest.main()

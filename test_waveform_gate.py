import copy
import json
from pathlib import Path
import tempfile
import unittest


class WaveformGateTests(unittest.TestCase):
    def test_protocol_rejects_seed_or_gate_changes(self):
        from waveform_gate import default_protocol,validate_protocol
        p=default_protocol();validate_protocol(p)
        for key,value in [('heldout_seeds',[401]),('train_pairs',0),('history_pulses',0),('noise',float('nan')),('std',-1),('thresholds',{'accuracy':.1})]:
            with self.assertRaises(ValueError):validate_protocol({**p,key:value})

    def test_gate_requires_accuracy_controls_invariants_and_refinement(self):
        from waveform_gate import summarize
        row={'seed':401,'arms':{name:{'accuracy':value} for name,value in
             [('joint',.95),('fixed',.5),('shuffled',.5),('erased',.5)]},
             'invariants':{'passed':True},'refinement':{'passed':True}}
        self.assertTrue(summarize([row])['passed'])
        for kind in ('joint','fixed','shuffled','erased','invariants','refinement'):
            bad=copy.deepcopy(row)
            if kind in bad['arms']:bad['arms'][kind]['accuracy']=.75
            else:bad[kind]['passed']=False
            self.assertFalse(summarize([bad])['passed'])

    def test_default_smoke_output_preserves_development_receipt(self):
        import os
        from waveform_gate import main,default_protocol
        with tempfile.TemporaryDirectory() as d:
            root=Path(d);protocol=root/'protocol.json';protocol.write_text(json.dumps(default_protocol()))
            target=root/'results/history_waveform';target.mkdir(parents=True)
            previous=target/'development.json';previous.write_text('development sentinel')
            old=Path.cwd()
            try:
                os.chdir(root)
                self.assertEqual(main(['--mode','smoke','--protocol',str(protocol)]),0)
            finally:os.chdir(old)
            self.assertEqual(previous.read_text(),'development sentinel')
            self.assertTrue((target/'smoke.json').exists())

    def test_smoke_receipt_cannot_be_held_out_success(self):
        from waveform_gate import main,default_protocol
        with tempfile.TemporaryDirectory() as d:
            p=Path(d)/'protocol.json';p.write_text(json.dumps(default_protocol()))
            out=Path(d)/'receipt.json'
            self.assertEqual(main(['--mode','smoke','--protocol',str(p),'--out',str(out)]),0)
            r=json.loads(out.read_text())
            self.assertTrue(r['complete']);self.assertFalse(r['held_out']);self.assertFalse(r['scientific_gate_passed'])
            self.assertEqual([x['seed'] for x in r['seed_results']],[401])
            row=r['seed_results'][0]
            self.assertEqual(row['arms']['fixed']['accuracy'],.5)
            self.assertEqual(row['arms']['erased']['accuracy'],.5)
            self.assertTrue(row['invariants']['passed'])
            self.assertEqual(main(['--mode','full','--protocol',str(p),'--out',str(Path(d)/'no.json')]),2)


if __name__=='__main__':unittest.main()

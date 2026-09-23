"""No GPU: validate experiment orchestration and paired metadata guardrails."""
import contextlib
import io
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import MagicMock, patch

import run_kv_experiment as suite


class KVSuiteTests(unittest.TestCase):
    def run_suite(self, fail_gate=False, mismatch=False):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp); model = root/'model'; model.mkdir(); (model/'config.json').write_text('{}')
            dest = root/'results'
            commands = []
            def popen(command, **kwargs):
                commands.append(command)
                out = Path(command[command.index('--output-dir')+1]); out.mkdir(parents=True)
                if command[2] == 'tests/check_mixed_gpu.py':
                    (out/'check.json').write_text(json.dumps({'status':'complete', 'pressure':{'preemptions':1,'recomputed_tokens':256}}))
                else:
                    allocation = command[command.index('--kv-allocation')+1]
                    blocks = int(command[command.index('--kv-cache-blocks')+1])
                    meta = {k:'same' for k in ('gpu','versions','workload_sha256','trace_sha256',
                            'model_metadata_sha256','weight_file_sizes','git_commit','git_status')}
                    meta.update(status='complete',effective_scheduling_policy='mixed',effective_kv_allocation=allocation,
                                kv_pool={'num_blocks':blocks + int(mismatch and allocation=='on_demand')},
                                arguments={'kv_allocation':allocation,'output_dir':str(out),'observe_kv':True})
                    (out/'metadata.json').write_text(json.dumps(meta))
                    (out/'summary.json').write_text(json.dumps({'requests':{}}))
                    for i in (1,2,3):
                        d=out/f'measure-{i:02}';d.mkdir()
                        (d/'timeline.json').write_text(json.dumps({
                            'requests':[{'output_tokens':128}], 'steps':[{}], 'wall_seconds':1,
                            'output_tokens_per_second':128,'kv_metrics':{'peak_allocated_blocks':blocks}}))
                proc = MagicMock();proc.__enter__.return_value=proc
                proc.stdout=iter(['a diagnostic line\n']);proc.wait.return_value=int(fail_gate)
                return proc
            with patch('sys.argv',['run_kv_experiment.py','--model',str(model),'--output-dir',str(dest)]),patch.object(suite.subprocess,'Popen',side_effect=popen),contextlib.redirect_stdout(io.StringIO()):
                if fail_gate or mismatch:
                    with self.assertRaises(RuntimeError):suite.main()
                else:suite.main()
            return commands,json.loads((dest/'suite.json').read_text())

    def test_failed_gate_stops_before_performance_runs(self):
        cmds,state = self.run_suite(fail_gate=True)
        self.assertEqual(len(cmds),1)
        self.assertEqual(state['status'],'failed')

    def test_success_has_one_gate_and_four_paired_runs(self):
        cmds,state = self.run_suite()
        self.assertEqual(len(cmds),5)
        self.assertEqual(state['status'],'complete')
        self.assertTrue(all('--observe-kv' in c and '--scheduling-policy' in c for c in cmds[1:]))

    def test_unequal_pool_is_not_a_performance_result(self):
        _,state = self.run_suite(mismatch=True)
        self.assertEqual(state['status'],'failed')
        self.assertIn('kv_pool',state['error'])


if __name__ == '__main__':unittest.main()

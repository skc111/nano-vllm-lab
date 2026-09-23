"""The cloud driver must stop after a failed gate and preserve its logs."""
import contextlib
import io
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch, MagicMock
import run_mixed_experiment as suite


class MixedSuiteTests(unittest.TestCase):
    def execute(self, *, fail_gate=False, mismatched_pool=False):
        with tempfile.TemporaryDirectory() as temp:
            root=Path(temp);model=root/'model';model.mkdir();(model/'config.json').write_text('{}')
            output=root/'run'
            commands=[]
            def popen(command, **kwargs):
                commands.append(command)
                if command[2]=='bench_timeline.py':
                    dest=Path(command[command.index('--output-dir')+1]);dest.mkdir()
                    meta={k:'same' for k in ('gpu','versions','workload_sha256','trace_sha256','model_metadata_sha256','git_commit','git_status')}
                    meta['kv_pool']={'num_blocks':121 if mismatched_pool and dest.name=='mixed' else 85}
                    meta.update(status='complete',effective_scheduling_policy=dest.name)
                    (dest/'metadata.json').write_text(json.dumps(meta))
                    for i in (1,2,3):
                        rd=dest/f'measure-{i:02}';rd.mkdir()
                        (rd/'timeline.json').write_text(json.dumps({'steps':[{'phase':'mixed','prefill_tokens':127,'decode_tokens':1}]}))
                process=MagicMock();process.__enter__.return_value=process
                process.stdout=iter(['diagnostic line\n'])
                process.wait.return_value=1 if fail_gate else 0
                return process
            with patch('sys.argv',['run_mixed_experiment.py','--model',str(model),'--output-dir',str(output)]),patch.object(suite.subprocess,'Popen',side_effect=popen),contextlib.redirect_stdout(io.StringIO()):
                if fail_gate or mismatched_pool:
                    with self.assertRaises(RuntimeError):suite.main()
                else:suite.main()
            state=json.loads((output/'suite.json').read_text())
            self.assertEqual((output/'correctness.log').read_text(),'diagnostic line\n')
            return commands,state

    def test_failed_gpu_gate_never_starts_benchmarks(self):
        commands,state=self.execute(fail_gate=True)
        self.assertEqual(len(commands),1)
        self.assertEqual(state['status'],'failed')

    def test_success_runs_all_three_policies_after_gate(self):
        commands,state=self.execute()
        self.assertEqual(len(commands),4)
        self.assertEqual([j['name'] for j in state['jobs']],['correctness','prefill_first','interleave','mixed'])
        self.assertEqual(state['status'],'complete')

    def test_different_kv_capacity_is_not_silently_accepted(self):
        _,state=self.execute(mismatched_pool=True)
        self.assertEqual(state['status'],'failed')
        self.assertIn('kv_pool',state['error'])

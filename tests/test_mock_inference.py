"""No-network integration tests for the real Neuroforge benchmark runner."""
import csv
import importlib
import os
import sys
import tempfile
import types
import unittest
from pathlib import Path
from unittest.mock import patch

from src.benchmark_quality import make_case


class MockInferenceTests(unittest.TestCase):
    def setUp(self):
        self.old_main = sys.modules.pop('main', None)
        self.old_task = sys.modules.pop('src.agi_benchmark', None)
        self.kaggle = types.ModuleType('kaggle_benchmarks')

        def task(**_metadata):
            def decorate(func):
                def run(**kwargs):
                    passed = func(**kwargs)
                    return types.SimpleNamespace(results=[{
                        'type': 'AGGREGATED', 'booleanResult': passed
                    }])
                func.run = run
                return func
            return decorate

        self.kaggle.task = task
        self.requests = types.ModuleType('requests')
        self.requests.post = lambda *_a, **_kw: self.fail('No HTTP should occur')
        self.modules = patch.dict(sys.modules, {
            'kaggle_benchmarks': self.kaggle,
            'requests': self.requests,
        })
        self.modules.start()
        self.runner = importlib.import_module('main')

    def tearDown(self):
        self.modules.stop()
        sys.modules.pop('main', None)
        sys.modules.pop('src.agi_benchmark', None)
        if self.old_main is not None:
            sys.modules['main'] = self.old_main
        if self.old_task is not None:
            sys.modules['src.agi_benchmark'] = self.old_task

    def run_benchmark(self, outcomes):
        class FakeModel:
            def prompt(self, _message):
                seed = int(os.environ['NEUROFORGE_CASE_SEED'])
                outcome = outcomes[seed - 1]
                if outcome == 'error':
                    raise ConnectionError('simulated model outage')
                _grid, expected = make_case(seed)
                if outcome == 'failure':
                    expected = [[0] * len(expected) for _ in expected]
                return repr(expected)

        with tempfile.TemporaryDirectory() as temp, patch.object(self.runner, 'OllamaLLM', FakeModel), patch.dict(os.environ, {}, clear=False):
            old_cwd = os.getcwd()
            try:
                os.chdir(temp)
                if 'error' in outcomes:
                    with self.assertRaisesRegex(RuntimeError, 'execution errors'):
                        self.runner.run_final_benchmark(iterations=len(outcomes))
                else:
                    self.runner.run_final_benchmark(iterations=len(outcomes))
                files = list(Path(temp).glob('neuroforge_results_*.csv'))
                self.assertEqual(len(files), 1)
                with files[0].open(newline='') as f:
                    rows = list(csv.DictReader(f))
                self.assertEqual(len(rows), len(outcomes))
                self.assertEqual([r['Trial'] for r in rows], [str(i) for i in range(1, len(outcomes) + 1)])
                return rows
            finally:
                os.chdir(old_cwd)

    def test_correct_model_response_records_success(self):
        rows = self.run_benchmark(['success', 'success'])
        self.assertEqual([r['Status'] for r in rows], ['SUCCESS', 'SUCCESS'])

    def test_wrong_response_records_failure_not_success(self):
        rows = self.run_benchmark(['failure'])
        self.assertEqual(rows[0]['Status'], 'FAILURE')
        self.assertEqual(rows[0]['Error_Type'], '')

    def test_model_outage_records_error_and_fails_closed(self):
        rows = self.run_benchmark(['success', 'failure', 'error'])
        self.assertEqual([r['Status'] for r in rows], ['SUCCESS', 'FAILURE', 'ERROR'])
        self.assertEqual(rows[2]['Error_Type'], 'ConnectionError')

    def test_generated_code_execution_is_blocked(self):
        import solver
        with self.assertRaisesRegex(RuntimeError, 'disabled'):
            solver.ARCSolver(None).validate_and_run('raise Exception("must not execute")', [], [])


if __name__ == '__main__':
    unittest.main()

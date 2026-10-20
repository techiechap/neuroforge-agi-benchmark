import unittest
from types import SimpleNamespace
from src.benchmark_quality import make_case, parse_grid, extract_pass


class BenchmarkQualityTests(unittest.TestCase):
    def test_seed_replay_and_expected_shift(self):
        for seed in range(50):
            grid, expected = make_case(seed)
            self.assertEqual((grid, expected), make_case(seed))
            self.assertEqual(sum(v == 1 for row in expected for v in row), 1)
            self.assertEqual(sum(v == 2 for row in expected for v in row), 1)

    def test_grid_response_from_prose(self):
        self.assertEqual(parse_grid('Here is the grid:\n```python\n[[0, 0, 1], [0, 2, 0], [0, 0, 0]]\n```'),
                         [[0, 0, 1], [0, 2, 0], [0, 0, 0]])

    def test_reject_malformed_grid(self):
        for text in ('[[0,1],[0,0]]', '[[0,0,1], [0,0], [0,0,0]]',
                     "__import__('os').system('echo BAD')", '[[0, True, 1],[0,0,2],[0,0,0]]'):
            self.assertIsNone(parse_grid(text))

    def test_score_completed_not_equal_success(self):
        self.assertFalse(extract_pass(SimpleNamespace(state='BENCHMARK_TASK_RUN_STATE_COMPLETED',
                                                   results=[{'type':'AGGREGATED','booleanResult':False}])))
        self.assertTrue(extract_pass(SimpleNamespace(results=[{'type':'AGGREGATED','booleanResult':True}])))
        with self.assertRaises(ValueError):
            extract_pass(SimpleNamespace(result={'unexpected': 'shape'}))

if __name__ == '__main__':
    unittest.main()

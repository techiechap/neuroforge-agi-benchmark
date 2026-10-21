import unittest
from tools.live_diagnostics import classify

class LiveDiagnosticTests(unittest.TestCase):
    def setUp(self):
        self.input = [[1,0,0],[0,2,0],[0,0,0]]
        self.expected = [[0,1,0],[0,2,0],[0,0,0]]

    def test_correct(self):
        self.assertEqual(classify(str(self.expected),self.expected,self.input)[0], 'CORRECT')

    def test_input_echo(self):
        self.assertEqual(classify(str(self.input),self.expected,self.input)[0], 'INPUT_ECHO')

    def test_format_failure(self):
        self.assertEqual(classify('```python\nprint(5)\n```',self.expected,self.input)[0], 'FORMAT_FAILURE')

    def test_wrong_grid(self):
        self.assertEqual(classify('[[0,0,1],[0,2,0],[0,0,0]]',self.expected,self.input)[0], 'WRONG_GRID')

if __name__ == '__main__':
    unittest.main()

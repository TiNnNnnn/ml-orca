"""Component boundaries and entry points, without PostgreSQL or training data."""
import ast
import importlib
import subprocess
import sys
import unittest
from unittest.mock import patch

from ml_orca.__main__ import COMMANDS
from ml_orca.common.paths import ML_ORCA_ROOT, PGORCA_ROOT, TEST_ASSETS, package_sources, source_file
from ml_orca.objectives.policy import objective_channels, policy_target


class ModuleContractTest(unittest.TestCase):
    def test_import_boundaries(self):
        for path in package_sources().values():
            for node in ast.walk(ast.parse(path.read_text())):
                names = ([node.module or ''] if isinstance(node, ast.ImportFrom) else
                         [alias.name for alias in node.names] if isinstance(node, ast.Import) else [])
                for name in names:
                    with self.subTest(file=str(path), dependency=name):
                        self.assertFalse(name == 'test' or name.startswith(('test.', 'ml_orca.tests.')))
                        if path.parent.name in ('models', 'objectives'):
                            self.assertFalse(name.startswith(('ml_orca.training', 'ml_orca.collect')))
                        if path.parent.name == 'models':
                            self.assertFalse(name.startswith('ml_orca.objectives'))
                        if path.parent.name == 'objectives':
                            self.assertFalse(name.startswith(('torch', 'ml_orca.models')))

    def test_commands_and_asset_paths(self):
        for command, module in COMMANDS.items():
            if command.startswith('train-'):
                continue  # Network tests are optional; help must work without torch.
            with self.subTest(command=command):
                self.assertTrue(callable(importlib.import_module('ml_orca.' + module).main))
        from ml_orca.collect.run_workload_comparison import DEFAULT_WORKLOADS, DEFAULT_POLICY
        self.assertEqual(DEFAULT_WORKLOADS, TEST_ASSETS / 'workloads')
        self.assertEqual(DEFAULT_POLICY, TEST_ASSETS / 'rules/empty_workload_cbo.policy')
        self.assertTrue((ML_ORCA_ROOT / 'collect/trace_corpus.sh').is_file())
        for path in package_sources().values():
            if path.name not in ('__init__.py', 'observed_search.py'):
                self.assertEqual(source_file(path.name), path)
        with self.assertRaises(ValueError):
            source_file('observed_search.py')  # Encoding and objective are intentionally distinct.
        self.assertEqual(ML_ORCA_ROOT.name, 'ml_orca')

    def test_custom_rule_library_does_not_inherit_builtin_policy(self):
        from ml_orca.collect.run_workload_comparison import parse_args
        base = ['run_workload_comparison', '--pg-config=x', '--audit-bin=y']
        with patch.object(sys, 'argv', base):
            self.assertIsNotNone(parse_args().policy_file)
        with patch.object(sys, 'argv', base + ['--rule-file=/tmp/custom.rules']):
            self.assertIsNone(parse_args().policy_file)

    def test_cli_help_from_outside_repository_without_site_packages(self):
        # -S hides optional installed dependencies; the lazy entry point still works.
        result = subprocess.run([sys.executable, '-B', '-S', str(ML_ORCA_ROOT), '--help'],
                                cwd='/tmp', text=True, capture_output=True, check=True)
        self.assertIn('train-observed', result.stdout)
        self.assertIn('train-policy', result.stdout)

    def test_objectives_keep_distinct_measured_channels(self):
        self.assertEqual(objective_channels('planning'), ('planning_ms',))
        self.assertEqual(objective_channels('plan_execution'), ('planning_ms', 'execution_ms'))
        self.assertEqual(policy_target({'response': {'planning_ms_median': 0., 'execution_ms_median': 0.}}), [0., 0.])
        with self.assertRaises(ValueError):
            policy_target({'response': {'planning_ms_median': float('nan'), 'execution_ms_median': 0.}})

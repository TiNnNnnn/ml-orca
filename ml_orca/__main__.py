"""Lazy command entry point: collecting/processing traces does not require torch."""
import importlib
from pathlib import Path
import sys

if not __package__:
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

COMMANDS = {
    'collect': 'collect.profile_corpus_attempts',
    'capture-context': 'collect.capture_rule_history_context',
    'trace-corpus': 'collect.run_trace_corpus',
    'compare-workload': 'collect.run_workload_comparison',
    'audit': 'data.audit_history_corpus',
    'recover': 'data.recover_history_corpus',
    'export-history': 'data.export_history_corpus',
    'export-policy': 'data.export_policy_learning_samples',
    'merge-graph': 'graph.merge_rule_graph',
    'render-graph': 'graph.render_rule_dependency_graph',
    'train-observed': 'training.train_observed_search',
    'train-policy': 'training.train_policy_baseline',
    'calibrate-dro': 'experiments.calibrate_rule_dro',
    'profile-observed': 'experiments.profile_observed_search',
}


def main():
    if len(sys.argv) < 2 or sys.argv[1] in ('-h', '--help'):
        print('usage: python3 -m ml_orca COMMAND [arguments]\n\nCommands:')
        for name, module in COMMANDS.items():
            print(f'  {name:18} {module}')
        return 0
    command = sys.argv.pop(1)
    if command not in COMMANDS:
        raise SystemExit('unknown ml-orca command: ' + command)
    return importlib.import_module('ml_orca.' + COMMANDS[command]).main()


if __name__ == '__main__':
    raise SystemExit(main())

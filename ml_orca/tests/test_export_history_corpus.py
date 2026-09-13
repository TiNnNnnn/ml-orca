from copy import deepcopy
import gzip
import json
from pathlib import Path
import tempfile
import unittest

from ml_orca.data.export_history_corpus import export_bundle, iter_history_runs
from ml_orca.common.artifacts import artifact_snapshot


class HistoryCorpusTest(unittest.TestCase):
    def test_label_subset_cannot_reassign_historical_holdout(self):
        from ml_orca.training.train_policy_baseline import assigned_history_queries
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / 'cohort.json'
            entries = [{'case_id': 'app:' + str(i), 'query': 'SELECT ' + str(i),
                        'family': str(i), 'split': split}
                       for i, split in enumerate(('train', 'train', 'validation', 'test'))]
            path.write_text(json.dumps({'entries': entries}))
            manifest = {'history_population': artifact_snapshot({'cohort': path})['cohort'],
                        'queries': [entries[0], entries[2]]}
            self.assertEqual(assigned_history_queries(manifest), {'app:0': 'SELECT 0', 'app:1': 'SELECT 1'})
            changed = deepcopy(manifest)
            changed['queries'][1]['split'] = 'train'
            with self.assertRaisesRegex(ValueError, 'preregistered'):
                assigned_history_queries(changed)

    def test_shard_matches_inline_and_rejects_changed_payload_or_identity(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / 'graph.json.gz'
            case = {'case_id': 'app:1', 'query': 'SELECT 1'}
            payload = {'schema_version': 1, 'scope': 'audited_planning_history_not_execution_labels',
                       'case': case, 'denominators': {'attempts': 1}, 'source_files': {},
                       'trees': [{'root': 0}], 'contexts': [{'attempts': 1}], 'edges': []}
            with gzip.open(path, 'wt') as stream:
                json.dump(payload, stream)
            snapshot = artifact_snapshot({'graph': path})['graph']
            run = {'source': snapshot, 'graph_payload': snapshot, 'case': case,
                   'unit': {'workload': 'app', 'query': '1'},
                   'inputs': {'query_sql': 'SELECT 1'}, 'denominators': payload['denominators']}
            bundle = {'runs': [run]}
            loaded, = iter_history_runs(bundle)
            for key in ('trees', 'contexts', 'edges'):
                self.assertEqual(loaded[key], payload[key])
                self.assertNotIn(key, run)  # No materialized payload retained in index.
            inline = {k: v for k, v in loaded.items() if k != 'graph_payload'}
            self.assertIs(next(iter_history_runs({'runs': [inline]})), inline)
            wrong = deepcopy(run)
            wrong['denominators']['attempts'] = 2
            with self.assertRaisesRegex(ValueError, 'denominators'):
                list(iter_history_runs({'runs': [wrong]}))
            wrong = deepcopy(run)
            wrong['inputs']['query_sql'] = 'SELECT 2'
            with self.assertRaisesRegex(ValueError, 'identity mismatch'):
                list(iter_history_runs({'runs': [wrong]}))
            path.write_bytes(b'changed')
            with self.assertRaisesRegex(ValueError, 'snapshot content changed'):
                list(iter_history_runs(bundle))

    def test_progress_report_cannot_freeze_training_history(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            path = root / 'admission.json'
            path.write_text(json.dumps({'progress_only': True, 'minimum_training_graphs_met': True,
                                        'collection_finalized': True}))
            with self.assertRaisesRegex(ValueError, 'not progress counts'):
                export_bundle(root, path)

    def test_explicit_cutoff_cannot_bypass_actual_dynamic_query_minimum(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            (root / 'audit').mkdir()
            (root / 'app').mkdir()
            case = {'case_id': 'app:1', 'query': 'SELECT 1'}
            manifest = {'history_split': 'train', 'artifact_start': {},
                        'datasets': [{'cases': [case]}]}
            (root / 'manifest.json').write_text(json.dumps(manifest))
            (root / 'audit/rule_graph.json').write_text('{}')
            context = {'status': 'ok', 'capture_input_endpoints_equal': True,
                       'resolved_policies': {'behavior': {'status': 'ok',
                            'snapshot': {'rules': [{'rule_hash': 'a'}]}}}}
            (root / 'app/pre-workload-context.json').write_text(json.dumps(context))
            payload = {'schema_version': 1, 'scope': 'audited_planning_history_not_execution_labels',
                       'case': case, 'source_files': artifact_snapshot({
                           'manifest': root / 'manifest.json',
                           'context': root / 'app/pre-workload-context.json'}),
                       'trees': [{'nodes': [{'path': 'r'}]}],
                       'contexts': [{'rule_hash': 'a', 'tree': 0, 'attempts': 1}],
                       'edges': [{'src_rule': 'a', 'dst_rule': 'a', 'tree': 0,
                                  'root': 0, 'dst_binding_path': 'r'}],
                       'denominators': {'attempts': 1, 'observed_edges': 1,
                                        'admitted_edges': 1, 'exclusions': {}}}
            graph = root / 'app/1.graph.json.gz'
            with gzip.open(graph, 'wt') as stream:
                json.dump(payload, stream)
            audit = {'scope': 'query_graph_admission_not_execution_labels_or_model_benefit',
                     'progress_only': True, 'collection_finalized': False,
                     'minimum_training_graphs_met': False, 'eligible_graphs_with_dynamic_edges': 2000,
                     'assigned_queries': 1, 'queries': [{'case_id': 'app:1', 'eligible': True,
                        'graph_snapshot': artifact_snapshot({'graph': graph})['graph']}]}
            path = root / 'admission.json'
            path.write_text(json.dumps(audit))
            with self.assertRaisesRegex(ValueError, 'not progress counts'):
                export_bundle(root, path)
            with self.assertRaisesRegex(ValueError, 'queries 1 < declared minimum 2000'):
                export_bundle(root, path, freeze_available=True)


if __name__ == '__main__':
    unittest.main()

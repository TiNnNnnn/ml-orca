import gzip
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from ml_orca.data.recover_history_corpus import check_artifacts, decode, main, publish
from ml_orca.common.artifacts import artifact_snapshot


class RecoveryTest(unittest.TestCase):
    def test_parallel_recovery_keeps_failed_sql_in_the_denominator(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            folder = root / 'app'
            (folder / 'logs').mkdir(parents=True)
            schema = root / 'schema.sql'
            schema.write_text('CREATE TABLE a (id int);')
            cases = [{'case_id': 'app:' + str(i), 'dataset': 'app', 'query': 'SELECT 1'} for i in (1, 2)]
            manifest = {'datasets': [{'dataset': 'app', 'cases': cases}],
                        'artifact_start': artifact_snapshot({'schema': schema})}
            (root / 'manifest.json').write_text(json.dumps(manifest))
            (folder / 'status.tsv').write_text('1\t3\n2\t3\n')
            (folder / 'pre-workload-context.json').write_text('{}')
            for i in (1, 2):
                with gzip.open(folder / 'logs' / f'{i}.log.gz', 'wt') as stream:
                    stream.write('ERROR: canceling statement due to statement timeout\n')
            with patch('sys.argv', ['recover_history_corpus.py', '--corpus', str(root),
                                    '--jobs', '2', '--memory-gb', '2', '--timeout', '30']):
                main()
            report = json.loads((root / 'summary.json').read_bytes())
            self.assertEqual(report['assigned_queries'], 2)
            self.assertEqual(report['decoder_failures'], [])
            self.assertEqual(report['decoder_limits']['jobs'], 2)
            for i in (1, 2):
                result = json.loads((folder / f'{i}.summary.json').read_bytes())
                self.assertFalse(result['complete'])
                self.assertEqual(result['returncode'], 3)
                self.assertIn('plan_timeout', result['exclusions'])
            self.assertEqual(schema.read_text(), 'CREATE TABLE a (id int);')

    def test_deferred_failure_retains_status_without_fabricating_attempt_counts(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            folder = root / 'app'
            (folder / 'logs').mkdir(parents=True)
            (root / 'manifest.json').write_text('{}')
            (folder / 'status.tsv').write_text('1\t3\n')
            (folder / 'pre-workload-context.json').write_text('{}')
            with gzip.open(folder / 'logs/1.log.gz', 'wt') as stream:
                stream.write('failed execution prefix')
            with patch('ml_orca.data.recover_history_corpus.trace_run', side_effect=AssertionError('must defer')):
                decode(root, {'case_id': 'app:1', 'query': 'SELECT 1'}, defer_failed=True)
            result = json.loads((folder / '1.summary.json').read_bytes())
            self.assertFalse(result['complete'])
            self.assertEqual(result['returncode'], 3)
            self.assertTrue(result['partial_trace_decode_pending'])
            self.assertNotIn('attempts', result)
            self.assertTrue(Path(result['trace']).exists())

    def test_exclusive_publication_and_artifact_guard(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / 'graph.json.gz'
            publish(path, {'complete': False}, compressed=True)
            with gzip.open(path, 'rt') as stream:
                self.assertEqual(json.load(stream), {'complete': False})
            manifest = {'artifact_start': artifact_snapshot({'graph': path})}
            self.assertEqual(check_artifacts(manifest), manifest['artifact_start'])
            with self.assertRaises(FileExistsError):
                publish(path, {'complete': True}, compressed=True)
            self.assertEqual(check_artifacts(manifest), manifest['artifact_start'])
            path.write_bytes(b'changed')
            with self.assertRaisesRegex(ValueError, 'artifacts changed'):
                check_artifacts(manifest)

    def test_existing_summary_and_missing_evidence_fail_closed(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            (root / 'app').mkdir()
            case = {'case_id': 'app:1', 'query': 'SELECT 1'}
            with self.assertRaisesRegex(ValueError, 'missing captured inputs'):
                decode(root, case)
            target = root / 'app/1.summary.json'
            target.write_text('original')
            with self.assertRaisesRegex(ValueError, 'must not be overwritten'):
                decode(root, case)
            self.assertEqual(target.read_text(), 'original')


if __name__ == '__main__':
    unittest.main()

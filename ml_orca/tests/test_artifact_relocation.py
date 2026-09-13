from pathlib import Path
import tempfile
import unittest

from ml_orca.common.artifacts import artifact_snapshot, read_snapshot, relocate_artifacts


class ArtifactRelocationTest(unittest.TestCase):
    def test_relocation_preserves_identity_and_content_validation(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            source, target = root / 'original', root / 'copied'
            source.mkdir()
            target.mkdir()
            (source / 'data').write_bytes(b'frozen input')
            snapshot = artifact_snapshot({'input': source / 'data'})['input']
            (target / 'data').write_bytes(b'frozen input')
            (source / 'data').unlink()  # Original machine need not be mounted.
            with relocate_artifacts([source, target]):
                self.assertEqual(read_snapshot(snapshot), b'frozen input')
                self.assertEqual(artifact_snapshot({'input': target / 'data'})['input'], snapshot)
                with relocate_artifacts():
                    with self.assertRaises(FileNotFoundError):
                        read_snapshot(snapshot)
                (target / 'data').write_bytes(b'changed input')
                with self.assertRaisesRegex(ValueError, 'snapshot content changed'):
                    read_snapshot(snapshot)
            with self.assertRaises(FileNotFoundError):
                read_snapshot(snapshot)

    def test_roots_are_explicit_and_prefix_matching_is_by_path_component(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            source = root / 'source'
            outside = root / 'source-suffix'
            outside.write_bytes(b'outside')
            snapshot = artifact_snapshot({'input': outside})['input']
            with relocate_artifacts([source, root]):
                self.assertEqual(read_snapshot(snapshot), b'outside')
            for roots in (['relative', root], [root], [root, root / 'missing']):
                with self.assertRaises(ValueError), relocate_artifacts(roots):
                    pass

"""Repository assets and source provenance; no imports from the test harness."""
from pathlib import Path
import os

ML_ORCA_ROOT = Path(__file__).resolve().parents[1]
# Standalone training only needs its manifest; collection may use a separate
# pgORCA checkout. Never infer that checkout from an installed site-packages path.
PGORCA_ROOT = Path(os.environ['PGORCA_ROOT']).resolve() if os.environ.get('PGORCA_ROOT') else next(
    (p for p in (*ML_ORCA_ROOT.parents, Path.cwd(), *Path.cwd().parents)
     if (p / 'libgpopt').is_dir() and (p / 'test/dsl').is_dir()), Path.cwd())
TEST_ASSETS = PGORCA_ROOT / 'test/dsl'


def source_file(filename):
    matches = [p for p in ML_ORCA_ROOT.rglob(filename) if 'tests' not in p.relative_to(ML_ORCA_ROOT).parts]
    if len(matches) != 1:
        raise ValueError('require one ml-orca source file: ' + filename)
    return matches[0]


def package_sources():
    return {'ml-orca:' + str(p.relative_to(ML_ORCA_ROOT)): p
            for p in sorted(ML_ORCA_ROOT.rglob('*.py'))
            if 'tests' not in p.relative_to(ML_ORCA_ROOT).parts}

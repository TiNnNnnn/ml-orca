"""Content-checked artifact IO shared by collection, encoding and training."""
from pathlib import Path
from contextlib import contextmanager
from contextvars import ContextVar
import zlib

_relocation = ContextVar('artifact_relocation', default=None)


@contextmanager
def relocate_artifacts(roots=None):
    """Resolve a copied repository without rewriting immutable provenance paths."""
    mapping = None
    if roots is not None:
        if len(roots) != 2 or any(not Path(p).is_absolute() for p in roots):
            raise ValueError('artifact relocation requires two absolute roots')
        mapping = tuple(Path(p).resolve() for p in roots)
        if not mapping[1].is_dir():
            raise ValueError('artifact destination root must exist')
    token = _relocation.set(mapping)
    try:
        yield
    finally:
        _relocation.reset(token)


def _mapped_path(path, reverse=False):
    path = Path(path).resolve()
    mapping = _relocation.get()
    if mapping is not None:
        source, destination = mapping[::-1] if reverse else mapping
        if path.is_relative_to(source):
            return destination / path.relative_to(source)
    return path


def artifact_snapshot(paths: dict[str, Path]) -> dict:
    """Content provenance, not a cryptographic identity or a lock against concurrent installs."""
    snapshot = {}
    for name, path in paths.items():
        item = {"path": str(_mapped_path(path, reverse=True))}
        try:
            checksum = size = 0
            with path.open("rb") as stream:
                for chunk in iter(lambda: stream.read(1024 * 1024), b""):
                    checksum = zlib.crc32(chunk, checksum)
                    size += len(chunk)
            item.update(size=size, crc32=f"{checksum:08x}")
        except OSError as error:
            item["error"] = str(error)
        snapshot[name] = item
    return snapshot

def read_snapshot(snapshot):
    raw = _mapped_path(snapshot['path']).read_bytes()
    if len(raw) != snapshot['size'] or f'{zlib.crc32(raw):08x}' != snapshot['crc32']:
        raise ValueError('snapshot content changed: ' + snapshot['path'])
    return raw

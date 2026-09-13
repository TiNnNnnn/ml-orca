"""Bounded CPU feature reuse and ordered prefetch; no model, optimizer or SQL work."""
from collections import OrderedDict
import sys
import time

import torch
from torch.utils.data import DataLoader

from ml_orca.common.artifacts import read_snapshot, relocate_artifacts
from ml_orca.encoding.rule_policy_encoding import encode_sequence_groups


def resident_size(value):
    """Conservative per-entry Python size, including shared static structures.

    Only plain input objects are permitted: learned tensors/autograd state must
    never enter this cache. The bound excludes worker/runtime and in-flight data.
    """
    pending, seen, size = [value], set(), 0
    while pending:
        item = pending.pop()
        if id(item) in seen:
            continue
        seen.add(id(item))
        size += sys.getsizeof(item)
        if type(item) is dict:
            pending.extend(item.keys())
            pending.extend(item.values())
        elif type(item) in (list, tuple):
            pending.extend(item)
        elif type(item) not in (str, int, float, bool, type(None)):
            raise TypeError('input cache only accepts parameter-free Python values')
    return size


def unchanged(value):
    # Default DataLoader conversion changes tuples/lists and creates tensors.
    return value


class ObservedInputDataset:
    def __init__(self, items, builder, vocabulary, cache_bytes=0, artifact_roots=None):
        if type(cache_bytes) is not int or cache_bytes < 0:
            raise ValueError('nonnegative input cache byte budget required')
        self.items, self.builder, self.vocabulary = items, builder, vocabulary
        self.cache_bytes, self.artifact_roots = cache_bytes, artifact_roots
        self.cache, self.used_bytes = OrderedDict(), 0

    def __len__(self):
        return len(self.items)

    def __getitem__(self, index):
        started = time.perf_counter()
        item = self.items[index]
        with relocate_artifacts(self.artifact_roots):
            # Even warm entries retain the original graph content check. Reuse
            # avoids decompression/JSON/SQL/token preparation, not input auditing.
            raw = read_snapshot(item['graph_snapshot'])
            checked = time.perf_counter()
            cached = self.cache.pop(index, None)
            hit = cached is not None
            if hit:
                data, size = cached
                self.cache[index] = cached
            else:
                data = self.builder(item, raw)
                data['sequences'] = encode_sequence_groups(data['sequences'], self.vocabulary)
                size = resident_size(data) if self.cache_bytes else 0
                if self.cache_bytes and size <= self.cache_bytes:
                    while self.used_bytes + size > self.cache_bytes:
                        _, (_, removed) = self.cache.popitem(last=False)
                        self.used_bytes -= removed
                    self.cache[index] = data, size
                    self.used_bytes += size
        return index, data, {'cache_hit': hit, 'cache_bytes': self.used_bytes,
                             'input_object_bytes': size,
                             'read_check_seconds': checked - started,
                             'prepare_seconds': time.perf_counter() - checked,
                             'worker_seconds': time.perf_counter() - started}


def input_loader(dataset, order, workers=0, prefetch=2):
    """Mutate order between exhausted epochs; DataLoader preserves its exact order."""
    if type(workers) is not int or workers < 0 or type(prefetch) is not int or prefetch < 1:
        raise ValueError('nonnegative workers and positive prefetch required')
    options = ({'multiprocessing_context': 'spawn', 'persistent_workers': True,
                'prefetch_factor': prefetch} if workers else {})
    return DataLoader(dataset, batch_size=None, sampler=order, num_workers=workers,
                      collate_fn=unchanged, generator=torch.Generator().manual_seed(0), **options)

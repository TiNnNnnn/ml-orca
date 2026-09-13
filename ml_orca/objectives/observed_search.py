"""Retrospective trace work labels; not policy value or total planning latency."""
import math

TARGETS = ('traced_match_constraint_ms', 'traced_instantiate_ms')


def validate_frozen_target(summary, target):
    """Check a content-verified summary; KEEP the original manifest's labels.

    Platform libm log1p can differ by one adjacent binary64 value. This is not
    a relative tolerance: larger changes, nonfinite labels and changed zeros fail.
    """
    expected = observed_target(summary)
    if (not isinstance(target, list) or len(target) != len(expected)
            or any(type(v) not in (int, float) or not math.isfinite(v) or v < 0 for v in target)):
        raise ValueError('invalid frozen timing target')
    for value, saved in zip(expected, target):
        if value == saved:
            continue
        if (value == 0 or saved == 0
                or saved not in (math.nextafter(value, -math.inf), math.nextafter(value, math.inf))):
            raise ValueError('frozen timing target changed')


def observed_target(summary):
    if not summary.get('complete') or summary.get('returncode') != 0:
        raise ValueError('incomplete_query')
    groups = summary.get('groups')
    if not isinstance(groups, list) or not groups:
        raise ValueError('timing_groups_unavailable')
    if sum(g['attempts'] for g in groups) != summary['attempts']:
        raise ValueError('timing_attempt_denominator_mismatch')
    values = []
    for fields in (('match_us', 'constraint_us'), ('instantiate_us',)):
        total = 0
        for group in groups:
            for field in fields:
                value = group.get(field)
                if type(value) not in (int, float) or not math.isfinite(value) or value < 0:
                    raise ValueError('invalid_timing_' + field)
                total += value
        values.append(math.log1p(total / 1000.))
    return values

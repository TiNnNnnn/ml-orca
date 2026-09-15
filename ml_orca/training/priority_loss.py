"""Tensor loss for the dependency-free priority objective contract."""
import math
import torch


def priority_cost_loss(scores, pairs):
    """Fit relative policy scores for ONE query/scenario, retaining exact ties.

    Each pair is (left_index, right_index, priority_pair report). No automatic
    filtering: callers must retain excluded cells in their experiment ledger.
    The score's additive offset is unidentified; it is not a predicted cost.
    Work counters remain separate diagnostics, not a hidden time penalty.
    """
    if (scores.ndim != 1 or not scores.is_floating_point() or not torch.isfinite(scores).all()
            or not pairs):
        raise ValueError('finite policy scores and admitted pairs required')
    predicted, targets, seen = [], [], set()
    for left, right, report in pairs:
        if (any(type(i) is not int or not 0 <= i < len(scores) for i in (left, right))
                or left == right or tuple(sorted((left, right))) in seen):
            raise ValueError('invalid or duplicate priority pair indices')
        seen.add(tuple(sorted((left, right))))
        if (report.get('contract') != 'priority-pair-v1' or report.get('exclusions') != []
                or report.get('cost_exclusions') != [] or report.get('work_exclusions') != []
                or report.get('trace_complete') is not True or report.get('work_delta') is None):
            raise ValueError('priority training requires validated cost, work and complete provenance')
        cost = report.get('cost') or {}
        values = [cost.get(k) for k in ('left', 'right', 'log1p_margin')]
        try:
            valid = all(type(v) in (int, float) and math.isfinite(v) for v in values)
        except OverflowError:
            valid = False
        if (not valid or min(values[:2]) < 0
                or values[2] != math.log1p(values[0]) - math.log1p(values[1])):
            raise ValueError('invalid priority cost margin')
        predicted.append(scores[left] - scores[right])
        targets.append(values[2])
    target = scores.new_tensor(targets)
    if not torch.isfinite(target).all():
        raise ValueError('priority target is outside model precision')
    return torch.nn.functional.smooth_l1_loss(torch.stack(predicted), target)

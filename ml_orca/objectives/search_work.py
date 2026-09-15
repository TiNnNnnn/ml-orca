"""Observable work-vector contract; no implicit conversion to time or scalar C."""
from ml_orca.trace.profile_rule_candidates import (
    candidate_evidence, cost_evidence, search_check_evidence)

WORK_CONTRACT = 'observed-search-events-v1'


def observed_work_vector(run):
    """Use complete event streams, including rejected work and budget skips.

    A cost-entry event is not always a cost computation. Likewise a skipped
    attempt is not an evaluated match. These counters are different units;
    scalarization requires separately frozen weights and scales.
    """
    attempts = candidate_evidence(run)
    costs = cost_evidence(run)
    checks = search_check_evidence(run)
    exclusions = sorted(set(attempts['exclusions'] + costs['exclusions'] + checks['exclusions']))
    complete = attempts['complete'] and costs['complete'] and checks['complete']
    values = {'dsl_dispatched_attempts': attempts['attempts'],
              'dsl_evaluated_attempts': attempts['evaluations'],
              'dsl_budget_skips': attempts['attempts'] - attempts['evaluations'],
              'cost_entries': len(costs['events']),
              'cost_entries_by_status': costs['status_counts'],
              'search_checks_by_stage_and_status': checks['status_counts']} if complete else None
    return {'contract': WORK_CONTRACT, 'complete': complete, 'exclusions': exclusions, 'values': values,
            'scope': 'observed_dispatch_cost_entry_and_search_check_events',
            'not_included': ['binding_construction_work', 'all_scheduler_rejections',
                             'exclusive_insert_and_statistics_work', 'hardware_time_conversion'],
            'is_scalar_C': False}

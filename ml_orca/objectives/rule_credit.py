"""Conservative observational credit accounting, not causal effect estimation."""
import math

from ml_orca.objectives.rule_utility import _finite

CREDIT_CONTRACT = 'new-inserter-equal-credit-v1'


def allocate_gain(gain, direct, ancestors, *, dsl_share):
    """Split an explicitly budgeted share equally over unique eligible instances.

    The share is a sensitivity parameter, NOT measured by provenance. No default
    is supplied. Background/unidentified credit keeps the residual. D wins when
    an instance is both direct and an ancestor; diamonds never multiply credit.
    """
    _finite(gain, 'gain', 0)
    if _finite(dsl_share, 'DSL credit share', 0) > 1:
        raise ValueError('DSL credit share must not exceed one')
    direct = list(direct)
    if any(type(seq) is not int or seq <= 0 for seq in direct):
        raise ValueError('invalid direct instance')
    direct = set(direct)
    future = set()
    for seq in direct:
        if seq not in ancestors:
            raise ValueError('missing ancestry, not an empty ancestry')
        for parent in ancestors[seq]:
            if type(parent) is not int or not 0 < parent < seq:
                raise ValueError('ancestor must strictly precede its descendant')
            future.add(parent)
    future -= direct
    eligible = direct | future
    per_instance = gain * dsl_share / len(eligible) if eligible else 0.
    credits = [{'sequence': seq, 'role': 'D' if seq in direct else 'F', 'credit': per_instance}
               for seq in sorted(eligible)]
    allocated = math.fsum(c['credit'] for c in credits)
    return {'gain': gain, 'dsl_share': dsl_share, 'credits': credits,
            'unattributed': max(0., gain - allocated),
            'scope': 'conserved_bookkeeping_not_identified_causal_contributions'}


def root_gain_credit(report, *, dsl_share):
    """Credit only comparable root improvements with new recorded inserters.

    All generators of the OLD costed plan are excluded from direct credit.
    A later duplicate generator is not a new insertion, so accumulating aliases
    cannot by itself manufacture a beneficiary. First feasibility, failure and
    missing provenance remain masked, never synthetic zero-gain labels.
    """
    allocate_gain(0., [], {}, dsl_share=dsl_share)  # Validate even empty/failed runs.
    plans = report.get('physical_plan_source_audit', {})
    work = report.get('instance_work_ledger')
    result = {'contract': CREDIT_CONTRACT, 'dsl_share': dsl_share, 'events': [],
              'by_instance': [], 'dfc_training_ready': False,
              'scope': 'root_observational_auxiliary_credit_not_terminal_policy_reward'}
    if (not report.get('audits') or not all(a['complete'] for a in report['audits'].values())
            or not report.get('cost_origin_audit', {}).get('complete')
            or not plans.get('complete') or work is None):
        return {**result, 'complete': False, 'exclusions': ['incomplete_source_audit']}
    instances = {r['sequence']: r for r in work['instances']}
    ancestors = {seq: r['ancestors'] for seq, r in instances.items()}
    # Audit output may be in memory or round-tripped through JSON object keys.
    plan_by_id = {int(seq): p for seq, p in plans['plans'].items()}
    totals, seen = {}, set()
    for event in report['root_cost_updates']:
        identity = event['event_sequence']
        if identity in seen:
            raise ValueError('duplicate gain event')
        seen.add(identity)
        row = {**event, 'allocation': None, 'credit_exclusions': list(event['exclusions'])}
        gain = event['cost_reduction']
        if event['first_feasible']:
            row['credit_exclusions'].append('first_feasible_not_an_infinite_gain')
        elif gain is None:
            row['credit_exclusions'].append('missing_comparable_gain')
        elif gain <= 0:
            row['credit_exclusions'].append('not_a_positive_gain')
        if not row['credit_exclusions']:
            old = plan_by_id[event['previous_candidate_sequence']]
            new = plan_by_id[event['candidate_sequence']]
            direct = set(new['recorded_inserters']) - set(old['generators'])
            row['eligible_new_inserters'] = sorted(direct)
            row['allocation'] = allocate_gain(gain, direct, ancestors, dsl_share=dsl_share)
            for credit in row['allocation']['credits']:
                seq, role = credit['sequence'], credit['role']
                entry = totals.setdefault(seq, {'sequence': seq, 'rule_hash': instances[seq]['rule_hash'],
                                                'D': [], 'F': []})
                entry[role].append(credit['credit'])
        result['events'].append(row)
    result['by_instance'] = [{**r, 'D': math.fsum(r['D']), 'F': math.fsum(r['F'])}
                             for _, r in sorted(totals.items())]
    return {**result, 'complete': True, 'exclusions': [],
            'complete_means': 'bookkeeping_valid_not_causal_identification_or_full_label_coverage'}

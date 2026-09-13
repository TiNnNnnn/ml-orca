"""Ordered recursive DSL trees, bound constraint factors, and rooted multi-round messages."""

import math
from collections import defaultdict
import torch
from torch import nn

from ml_orca.models.rule_policy_model import WholePolicyPredictor
from ml_orca.models.message_passing import DirectedRuleAggregation
from ml_orca.models.sequence import query_context_embedding


def context_summary(values, destinations, weights, rule_count, variance=False):
    """Attempt-weighted empirical moments of encoded contexts, not new graph nodes.

    Central moments avoid subtracting two nearly equal second moments. This is
    a finite summary, not an injective encoding of the context distribution.
    """
    counts = values.new_zeros((rule_count, 1)).index_add(0, destinations, weights)
    mean = values.new_zeros((rule_count, values.shape[1])).index_add(0, destinations, weights * values)
    mean = mean / counts.clamp_min(1)
    summary = [mean]
    if variance:
        residual = values - mean[destinations]
        moment = torch.zeros_like(mean).index_add(0, destinations, weights * residual.square())
        summary.append(moment / counts.clamp_min(1))
    return torch.cat((*summary, torch.log1p(counts)), 1)


class RuleTreeEncoder(nn.Module):
    """Ordered, arbitrary-arity Tree-LSTM with constraint factors linked through symbols.

    A sibling GRU supplies the ordered child summary to the Tree-LSTM gates;
    children retain individual cell states and forget gates. No DSL subtree is
    serialized into a recurrent sentence. Local attribute lists remain ordered.
    """

    def __init__(self, width):
        super().__init__()
        self.child_order = nn.GRU(width, width, batch_first=True)
        self.iou = nn.Linear(2 * width, 3 * width)
        self.forget = nn.Linear(2 * width, width)
        self.slot = nn.Linear(1, width, bias=False)
        self.symbol_context = nn.Linear(2 * width + 1, width)
        self.constraint_context = nn.Linear(width, width, bias=False)
        self.rule_readout = nn.Linear(3 * width + 1, width)

    def tree_states(self, inputs, nodes):
        if len(inputs) != len(nodes) or not nodes:
            raise ValueError('one feature vector per tree node required')
        levels, batches = [], defaultdict(list)
        for index, node in enumerate(nodes):
            children = node['children']
            if any(type(child) is not int or not 0 <= child < index for child in children):
                raise ValueError('tree children must precede their parent')
            level = 1 + max((levels[c] for c in children), default=-1)
            levels.append(level)
            batches[level, len(children)].append(index)
        # Equal-height, equal-arity nodes are independent. Batch their ORIGINAL
        # ordered sibling GRUs (no pooling replacement, padding or child sorting).
        # unbind shares one backward node instead of N dense SelectBackward paths.
        values = inputs.unbind(0)
        hidden, cells = [None] * len(nodes), [None] * len(nodes)
        for (_, arity), indices in sorted(batches.items()):
            value = torch.stack([values[i] for i in indices])
            summary, child_cell = torch.zeros_like(value), torch.zeros_like(value)
            if arity:
                children = [c for i in indices for c in nodes[i]['children']]
                child_hidden = torch.stack([hidden[c] for c in children]).reshape(len(indices), arity, -1)
                summary = self.child_order(child_hidden)[1][0]
                forget = torch.sigmoid(self.forget(torch.cat((value[:, None, :].expand(-1, arity, -1), child_hidden), 2)))
                child_cell = (forget * torch.stack([cells[c] for c in children]).reshape_as(child_hidden)).sum(1)
            i, o, u = self.iou(torch.cat((value, summary), 1)).chunk(3, dim=1)
            cell = torch.sigmoid(i) * torch.tanh(u) + child_cell
            for index, h, c in zip(indices, (torch.sigmoid(o) * torch.tanh(cell)).unbind(0), cell.unbind(0)):
                hidden[index], cells[index] = h, c
        return torch.stack(hidden)

    def forward(self, structure, sequences, encoder):
        return self.forward_many([structure], sequences, encoder)[0]

    def forward_many(self, structures, sequences, encoder):
        """Disjoint rule forests; only tensor addresses, never encoded symbol IDs, are rebased."""
        if not structures:
            return []
        inputs = {k: [] for k in ('rule_node', 'rule_symbol', 'rule_constraint')}
        forest, occurrences, references, roots, sizes, constraint_sizes = [], [], [], [], [], []
        for structure in structures:
            features = {}
            for channel in inputs:
                bounds = structure[channel + '_range']
                if (len(bounds) != 2 or any(type(i) is not int for i in bounds)
                        or not 0 <= bounds[0] <= bounds[1] <= len(sequences[channel])):
                    raise ValueError('invalid rule feature range')
                features[channel] = sequences[channel][bounds[0]:bounds[1]]
            nodes, local_roots = structure['nodes'], structure['roots']
            sites, args = structure['symbol_occurrences'], structure['constraint_references']
            if (not nodes or len(nodes) != len(features['rule_node']) or len(sites) != len(features['rule_symbol'])
                    or len(args) != len(features['rule_constraint']) or set(local_roots) != {'source', 'target'}
                    or any(type(i) is not int or not 0 <= i < len(nodes) or nodes[i]['side'] != side
                           or nodes[i]['path'] != 'r' for side, i in local_roots.items())):
                raise ValueError('invalid rule tree/constraint layout')
            parents, actual = [0] * len(nodes), [[] for _ in sites]
            for index, node in enumerate(nodes):
                for position, child in enumerate(node['children']):
                    if (type(child) is not int or not 0 <= child < index or nodes[child]['side'] != node['side']
                            or nodes[child]['path'] != node['path'] + '/' + str(position)):
                        raise ValueError('invalid ordered tree edge')
                    parents[child] += 1
                for slot, ref in enumerate(node['symbols']):
                    if type(ref) is not int or not 0 <= ref < len(sites):
                        raise ValueError('invalid node symbol reference')
                    actual[ref].append([index, slot])
            if actual != sites or any(n != (0 if i in local_roots.values() else 1) for i, n in enumerate(parents)):
                raise ValueError('inconsistent tree ownership or symbol incidence')
            node_offset, symbol_offset = len(forest), len(occurrences)
            for sequence, refs in zip(features['rule_constraint'], args):
                tokens, symbols = [], []
                for arg in refs:
                    token, ref = arg['token'], arg['symbol']
                    if (type(token) is not int or not 0 <= token < len(sequence['tokens']) or token in tokens
                            or type(ref) is not int or not 0 <= ref < len(sites)
                            or not sequence['known_number'][token] or sequence['numbers'][token] != ref):
                        raise ValueError('invalid bound constraint argument')
                    tokens.append(token)
                    symbols.append(symbol_offset + ref)
                references.append((tokens, symbols))
            forest.extend({'children': [node_offset + c for c in n['children']],
                           'symbols': [symbol_offset + r for r in n['symbols']]} for n in nodes)
            occurrences.extend([[(node_offset + n, slot) for n, slot in group] for group in sites])
            roots.append([node_offset + local_roots[side] for side in ('source', 'target')])
            sizes.append(len(nodes))
            constraint_sizes.append(len(args))
            for channel in inputs:
                inputs[channel].extend(features[channel])
        node_features, symbol_features = encoder(inputs['rule_node']), encoder(inputs['rule_symbol'])
        initial = self.tree_states(node_features, forest)

        def mean_rows(values, destinations, count):
            pooled = initial.new_zeros((count, initial.shape[1]))
            if not destinations:
                return pooled
            indices = torch.tensor(destinations, dtype=torch.long, device=initial.device)
            counts = initial.new_zeros((count, 1)).index_add(0, indices, initial.new_ones((len(indices), 1)))
            return pooled.index_add(0, indices, values) / counts.clamp_min(1)

        site_nodes, site_slots, site_symbols = [], [], []
        for symbol, sites in enumerate(occurrences):
            for node, slot in sites:
                site_nodes.append(node)
                site_slots.append([math.log1p(slot)])
                site_symbols.append(symbol)
        context = initial.new_zeros(symbol_features.shape)
        if site_nodes:
            values = initial.index_select(0, torch.tensor(site_nodes, device=initial.device)) + self.slot(initial.new_tensor(site_slots))
            context = mean_rows(values, site_symbols, len(occurrences))
        symbols = torch.tanh(self.symbol_context(torch.cat((symbol_features, context,
                    initial.new_tensor([[math.log1p(len(sites))] for sites in occurrences]).reshape(-1, 1)), 1)))
        # One gather for all constraint ports, retaining every occurrence/gradient.
        port_indices = [r for _, refs in references for r in refs]
        ports = symbols.index_select(0, torch.tensor(port_indices, dtype=torch.long, device=initial.device))
        port_groups = ports.split([len(refs) for _, refs in references])
        contexts = []
        for sequence, (tokens, _), values in zip(inputs['rule_constraint'], references, port_groups):
            context = initial.new_zeros((len(sequence['tokens']), initial.shape[1]))
            if tokens:
                context = context.index_copy(0, torch.tensor(tokens, device=initial.device), values)
            contexts.append(context)
        constraints = encoder(inputs['rule_constraint'], contexts)
        incidence = [i for i, (_, refs) in enumerate(references) for _ in refs]
        bound_symbols = mean_rows(constraints.index_select(0, torch.tensor(incidence, dtype=torch.long, device=initial.device)),
                                  port_indices, len(occurrences))
        refs = [r for n in forest for r in n['symbols']]
        destinations = [i for i, n in enumerate(forest) for _ in n['symbols']]
        bound_nodes = mean_rows(bound_symbols.index_select(0, torch.tensor(refs, dtype=torch.long, device=initial.device)),
                                destinations, len(forest))
        states = self.tree_states(node_features + self.constraint_context(bound_nodes), forest)
        summaries = mean_rows(constraints, [i for i, size in enumerate(constraint_sizes) for _ in range(size)], len(structures))
        paired = states.index_select(0, torch.tensor(roots, device=initial.device).flatten()).reshape(len(structures), -1)
        rules = torch.tanh(self.rule_readout(torch.cat((paired, summaries,
                          initial.new_tensor([[math.log1p(n)] for n in constraint_sizes])), 1)))
        return list(zip(rules.unbind(0), states.split(sizes)))


class TreePolicyPredictor(WholePolicyPredictor):
    """Constraint-aware tree roots condition each directed message at every graph round."""

    def __init__(self, vocabulary_size, width=32, graph_mode='static', message_rounds=3, history=False,
                 history_mode='graph', history_pooling='mean'):
        if graph_mode not in ('none', 'static', 'self') or type(message_rounds) is not int or message_rounds < 1:
            raise ValueError('invalid tree graph configuration')
        super().__init__(vocabulary_size, width, 'none')
        self.graph_mode = graph_mode
        self.rule_trees = RuleTreeEncoder(width)
        self.rounds = nn.ModuleList([DirectedRuleAggregation(width, rooted=True)
                                    for _ in range(message_rounds if graph_mode != 'none' else 0)])
        self.use_history = history
        if history_mode not in ('none', 'context', 'graph'):
            raise ValueError('invalid history ablation')
        self.history_mode = history_mode
        if history_pooling not in ('mean', 'mean_variance'):
            raise ValueError('invalid historical context pooling')
        self.history_pooling = history_pooling
        if history:
            self.history_run = nn.Linear(2 * width + 1, width)
            self.history_attempt = nn.Linear(4 * width, width)
            self.history_update = nn.Linear((2 if history_pooling == 'mean_variance' else 1) * width + 1,
                                            width, bias=False)
            self.history_ports = nn.Linear(width, 2 * width)

    def history_inputs(self, features, query, local):
        """Historical actual trees with behavior query/catalog/policy, no current Memo."""
        if features.get('history_admission', {}).get('capture') != 'audited_history_frozen':
            raise ValueError('historical observations require admission')
        return self.context_inputs(features, query, local)

    def context_inputs(self, features, query, local):
        """Shared rooted context computation; callers enforce their observation boundary."""
        sequences = features['sequences']

        def take(channel, bounds):
            values = sequences[channel]
            if (len(bounds) != 2 or any(type(i) is not int for i in bounds)
                    or not 0 <= bounds[0] <= bounds[1] <= len(values)):
                raise ValueError('invalid history feature range')
            return values[bounds[0]:bounds[1]]

        runs, policies = [], []
        for run in features['history_runs']:
            context = query_context_embedding(self.encoder, {'query_binding': run['query_binding'], 'sequences': {
                'query': take('history_query', run['query_range']),
                'relation': take('history_relation', run['relation_range'])}})[0]
            policy = self.encoder(take('history_policy', run['policy_range']))
            membership = run['loaded_rules']
            if len(membership) != len(policy) or any(type(v) is not bool for v in membership):
                raise ValueError('invalid historical loaded policy membership')
            active = [i for i, loaded in enumerate(membership) if loaded]
            summary = policy[active].mean(0) if active else torch.zeros_like(context)
            runs.append(torch.tanh(self.history_run(torch.cat((context, summary,
                                    context.new_tensor([math.log1p(len(active))]))))))
            policies.append(policy)
        forest, node_sequences, sizes, offsets = [], [], [], []
        for tree in features['history_trees']:
            offsets.append(len(forest))
            if tree is None:
                sizes.append(0)
                continue
            nodes, root = tree['nodes'], tree['root']
            if type(root) is not int or not 0 <= root < len(nodes) or nodes[root]['path'] != 'r':
                raise ValueError('invalid historical tree root')
            ownership = [0] * len(nodes)
            for node in nodes:
                for slot, child in enumerate(node['children']):
                    if (type(child) is not int or not 0 <= child < len(nodes)
                            or nodes[child]['path'] != node['path'] + '/' + str(slot)):
                        raise ValueError('invalid historical ordered child')
                    ownership[child] += 1
            if any(n != (0 if i == root else 1) for i, n in enumerate(ownership)):
                raise ValueError('invalid historical tree ownership')
            values = take('history_node', tree['node_range'])
            if len(values) != len(nodes):
                raise ValueError('one feature vector per tree node required')
            forest.extend({'children': [offsets[-1] + c for c in n['children']]} for n in nodes)
            node_sequences.extend(values)
            sizes.append(len(nodes))
        states = (self.rule_trees.tree_states(self.encoder(node_sequences), forest) if forest
                  else query.new_zeros((0, query.shape[1])))
        attempts = self.encoder(sequences['history_attempt'])
        if len(attempts) != len(features['history_contexts']):
            raise ValueError('historical attempt feature mismatch')
        destinations, weights, attempt_indices, context_runs, context_roots, context_policies = [], [], [], [], [], []
        policy_offsets, policy_size = [], 0
        for policy in policies:
            policy_offsets.append(policy_size)
            policy_size += len(policy)
        for i, item in enumerate(features['history_contexts']):
            rule, run, tree, count = (item[k] for k in ('rule', 'run', 'tree', 'count'))
            if type(count) is not int or count < 1:
                raise ValueError('invalid historical attempt count')
            if rule not in local:
                continue
            if (type(run) is not int or not 0 <= run < len(runs)
                    or type(tree) is not int or not 0 <= tree < len(sizes)
                    or type(rule) is not int or not 0 <= rule < len(policies[run])):
                raise ValueError('invalid historical context index')
            context_roots.append(offsets[tree] + features['history_trees'][tree]['root'] if sizes[tree] else len(states))
            context_runs.append(run)
            context_policies.append(policy_offsets[run] + rule)
            attempt_indices.append(i)
            destinations.append(local[rule])
            weights.append([count])
        destinations = torch.tensor(destinations, dtype=torch.long, device=query.device)
        weights = query.new_tensor(weights).reshape(-1, 1)
        values = query.new_zeros((0, query.shape[1]))
        if attempt_indices:
            # Like binding edges, attempts require a batched gather: scalar
            # attempts[i] creates a dense gradient per occurrence in backward.
            def gather(tensor, indices):
                return tensor.index_select(0, torch.tensor(indices, device=query.device))
            roots = gather(torch.cat((states, query.new_zeros((1, query.shape[1])))), context_roots)
            inputs = torch.cat((roots, gather(torch.stack(runs), context_runs),
                                gather(torch.cat(policies), context_policies), gather(attempts, attempt_indices)), 1)
            values = torch.tanh(self.history_attempt(inputs))
        update = self.history_update(context_summary(values, destinations, weights, len(local),
                                     variance=self.history_pooling == 'mean_variance'))
        edges, edge_indices, run_indices, root_indices = [], [], [], []
        if len(sequences['history_edge']) != len(features['history_edges']):
            raise ValueError('historical edge feature mismatch')
        if self.history_mode != 'graph':
            return update, [], query.new_zeros((0, query.shape[1])), query.new_zeros((0, 2, query.shape[1]))
        encoded_edges = self.encoder(sequences['history_edge'])
        for i, edge in enumerate(features['history_edges']):
            if not all(rule in local for rule in edge['rules']):
                continue
            tree, root = edge['tree'], edge['root']
            if (type(tree) is not int or not 0 <= tree < len(sizes)
                    or sizes[tree] == 0 or type(root) is not int or not 0 <= root < sizes[tree]):
                raise ValueError('historical binding port is unobserved')
            run = edge['run']
            if type(run) is not int or not 0 <= run < len(runs):
                raise ValueError('invalid historical edge run')
            edges.append([local[rule] for rule in edge['rules']])
            edge_indices.append(i)
            run_indices.append(run)
            root_indices.append(offsets[tree] + root)
        if not edges:
            return update, edges, query.new_zeros((0, query.shape[1])), query.new_zeros((0, 2, query.shape[1]))
        # Gather all occurrences together: independent scalar selects create an
        # E-by-width dense gradient for EACH edge in backward (quadratic work).
        # Repeated edges/roots remain repeated; index_select accumulates every
        # gradient without detaching, sampling, or caching trainable states.
        positions = encoded_edges.index_select(0, torch.tensor(edge_indices, device=query.device))
        positions = positions + torch.stack(runs).index_select(0, torch.tensor(run_indices, device=query.device))
        roots = states.index_select(0, torch.tensor(root_indices, device=query.device))
        return update, edges, positions, self.history_ports(roots).reshape(-1, 2, query.shape[1])

    def forward(self, features):
        sequences, loaded, trees = features['sequences'], features['loaded_rules'], features['rule_trees']
        if (len(loaded) != len(trees) or len(loaded) != len(sequences['policy'])
                or any(type(flag) is not bool for flag in loaded)):
            raise ValueError('invalid loaded tree rule membership')
        query = query_context_embedding(self.encoder, features)
        active = [i for i, flag in enumerate(loaded) if flag]
        local = {index: position for position, index in enumerate(active)}
        encoded = self.rule_trees.forward_many([trees[i] for i in active], sequences, self.encoder)
        rules = torch.stack([r for r, _ in encoded]) if encoded else query.new_zeros((0, query.shape[1]))
        policies = self.encoder([sequences['policy'][i] for i in active])
        nodes = self.rule_context(torch.cat((rules, policies, query.expand(len(rules), -1)), 1))
        history = None
        if self.use_history and self.history_mode != 'none':
            history = self.history_inputs(features, query, local)
            nodes = nodes + history[0]
        elif not self.use_history and 'history_runs' in features:
            raise ValueError('model must explicitly enable historical inputs')
        if self.rounds:
            edges, endpoints = features['rule_edges'], features['edge_roots']
            if (features.get('rule_graph_scope') != 'native_static_template'
                    or len(edges) != len(sequences['edge']) or len(edges) != len(endpoints)
                    or any(len(e) != 2 or any(type(i) is not int or not 0 <= i < len(trees) for i in e) for e in edges)):
                raise ValueError('require verified root-bound static graph')
            selected = [i for i, (s, t) in enumerate(edges) if loaded[s] and loaded[t]] if self.graph_mode == 'static' else []
            pairs = []
            for i in selected:
                if len(endpoints[i]) != 2:
                    raise ValueError('require two dependency roots')
                pair = []
                for rule, root, side in zip(edges[i], endpoints[i], ('target', 'source')):
                    if (type(root) is not int or not 0 <= root < len(trees[rule]['nodes'])
                            or trees[rule]['nodes'][root]['side'] != side):
                        raise ValueError('dependency root has wrong side or index')
                    pair.append(encoded[local[rule]][1][root])
                pairs.append(torch.stack(pair))
            root_vectors = torch.stack(pairs) if pairs else query.new_zeros((0, 2, query.shape[1]))
            positions = self.encoder([sequences['edge'][i] for i in selected])
            links = [[local[r] for r in edges[i]] for i in selected]
            if history is not None and self.graph_mode == 'static':
                links += history[1]
                positions = torch.cat((positions, history[2]))
                root_vectors = torch.cat((root_vectors, history[3]))
            for layer in self.rounds:
                nodes = layer(nodes, links, positions, root_vectors)
        pooled = nodes.sum(0, keepdim=True) / max(1, len(nodes))
        return self.readout(torch.cat((query, pooled, query.new_tensor([[math.log1p(len(nodes))]])), 1))[0]


class ObservedSearchPredictor(TreePolicyPredictor):
    """Same DSL trees/constraints/rooted GNN, with CURRENT observed GE context, not prospective history."""

    def history_inputs(self, features, query, local):
        if features.get('observation_scope') != 'retrospective_current_trace':
            raise ValueError('require explicit retrospective observation scope')
        if 'history_admission' in features:
            raise ValueError('do not confuse observed-current inputs with prospective historical admission')
        return self.context_inputs(features, query, local)

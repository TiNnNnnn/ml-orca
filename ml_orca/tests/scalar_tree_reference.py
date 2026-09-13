"""Scalar executable reference for batching equivalence, not a production fallback."""
import math
import unittest
try:
    import torch
except ModuleNotFoundError as error:
    if error.name != 'torch':
        raise
    raise unittest.SkipTest('PyTorch is optional')
from ml_orca.models.rule_tree_model import RuleTreeEncoder


class ScalarRuleTreeEncoder(RuleTreeEncoder):
    def tree_states(self, inputs, nodes):
        if len(inputs) != len(nodes) or not nodes:
            raise ValueError('one feature vector per tree node required')
        hidden, cells = [], []
        for index, (value, node) in enumerate(zip(inputs, nodes)):
            children = node['children']
            if any(type(child) is not int or not 0 <= child < index for child in children):
                raise ValueError('tree children must precede their parent')
            summary, child_cell = torch.zeros_like(value), torch.zeros_like(value)
            if children:
                child_hidden = torch.stack([hidden[c] for c in children])
                summary = self.child_order(child_hidden.unsqueeze(0))[1][0, 0]
                forget = torch.sigmoid(self.forget(torch.cat((value.expand(len(children), -1), child_hidden), 1)))
                child_cell = (forget * torch.stack([cells[c] for c in children])).sum(0)
            i, o, u = self.iou(torch.cat((value, summary))).chunk(3)
            cell = torch.sigmoid(i) * torch.tanh(u) + child_cell
            cells.append(cell)
            hidden.append(torch.sigmoid(o) * torch.tanh(cell))
        return torch.stack(hidden)

    def forward(self, structure, sequences, encoder):
        def features(channel):
            bounds = structure[channel + '_range']
            if (len(bounds) != 2 or any(type(i) is not int for i in bounds)
                    or not 0 <= bounds[0] <= bounds[1] <= len(sequences[channel])):
                raise ValueError('invalid rule feature range')
            return sequences[channel][bounds[0]:bounds[1]]

        node_features = encoder(features('rule_node'))
        symbol_features = encoder(features('rule_symbol'))
        constraint_sequences = features('rule_constraint')
        nodes, roots = structure['nodes'], structure['roots']
        occurrences, references = structure['symbol_occurrences'], structure['constraint_references']
        if (not nodes or len(nodes) != len(node_features) or len(occurrences) != len(symbol_features)
                or len(references) != len(constraint_sequences) or set(roots) != {'source', 'target'}
                or any(type(i) is not int or not 0 <= i < len(nodes) or nodes[i]['side'] != side
                       or nodes[i]['path'] != 'r' for side, i in roots.items())):
            raise ValueError('invalid rule tree/constraint layout')
        # The compiler supplies a forest, not a DAG. Validate ownership and links
        # before using them as tensor gather/scatter indices.
        parents = [0] * len(nodes)
        actual_occurrences = [[] for _ in occurrences]
        for index, node in enumerate(nodes):
            for position, child in enumerate(node['children']):
                if (type(child) is not int or not 0 <= child < index
                        or nodes[child]['side'] != node['side']
                        or nodes[child]['path'] != node['path'] + '/' + str(position)):
                    raise ValueError('invalid ordered tree edge')
                parents[child] += 1
            for slot, ref in enumerate(node['symbols']):
                if type(ref) is not int or not 0 <= ref < len(occurrences):
                    raise ValueError('invalid node symbol reference')
                actual_occurrences[ref].append([index, slot])
        if actual_occurrences != occurrences or any(n != (0 if i in roots.values() else 1) for i, n in enumerate(parents)):
            raise ValueError('inconsistent tree ownership or symbol incidence')
        initial = self.tree_states(node_features, nodes)
        symbols = []
        for value, sites in zip(symbol_features, occurrences):
            context = torch.zeros_like(value)
            if sites:
                slots = value.new_tensor([[math.log1p(slot)] for _, slot in sites])
                context = (initial[[node for node, _ in sites]] + self.slot(slots)).mean(0)
            symbols.append(torch.tanh(self.symbol_context(torch.cat((value, context,
                                      value.new_tensor([math.log1p(len(sites))]))))))
        contexts = []
        incidence = [[] for _ in symbols]
        for index, (sequence, args) in enumerate(zip(constraint_sequences, references)):
            context = initial.new_zeros((len(sequence['tokens']), initial.shape[1]))
            tokens, values = [], []
            for arg in args:
                token, ref = arg['token'], arg['symbol']
                if (type(token) is not int or not 0 <= token < len(sequence['tokens']) or token in tokens
                        or type(ref) is not int or not 0 <= ref < len(symbols)
                        or not sequence['known_number'][token] or sequence['numbers'][token] != ref):
                    raise ValueError('invalid bound constraint argument')
                tokens.append(token)
                values.append(symbols[ref])
                incidence[ref].append(index)
            if tokens:
                context = context.index_copy(0, torch.tensor(tokens, device=context.device), torch.stack(values))
            contexts.append(context)
        constraints = encoder(constraint_sequences, contexts)
        symbol_constraints = [constraints[indices].mean(0) if indices else initial.new_zeros(initial.shape[1])
                              for indices in incidence]
        local = []
        for value, node in zip(node_features, nodes):
            bound = torch.stack([symbol_constraints[r] for r in node['symbols']]).mean(0) if node['symbols'] else torch.zeros_like(value)
            local.append(value + self.constraint_context(bound))
        states = self.tree_states(torch.stack(local), nodes)
        constraint_summary = constraints.mean(0) if len(constraints) else torch.zeros_like(states[0])
        rule = torch.tanh(self.rule_readout(torch.cat((states[roots['source']], states[roots['target']],
                   constraint_summary, states.new_tensor([math.log1p(len(constraints))])))))
        return rule, states

    def forward_many(self, structures, sequences, encoder):
        return [self.forward(s, sequences, encoder) for s in structures]

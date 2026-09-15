"""Optional PyTorch encoders and experimental complete-policy readout."""

import math
import torch
from torch import nn
from ml_orca.models.sequence import SharedSequenceEncoder, query_context_embedding, RuleAttemptEncoder
from ml_orca.models.message_passing import DirectedRuleAggregation


class WholePolicyPredictor(nn.Module):
    """Shared set/static-graph model; the caller declares the output objective.

    The default two channels retain the legacy planning/execution model layout.
    Explicit output_size changes require a separate checkpoint/objective contract.
    Only prospective rule/policy/query/catalog channels are read. Current Memo,
    historical edges/counts, and response labels are intentionally not inputs.
    """

    def __init__(self, vocabulary_size, width=32, graph_mode='none', output_size=2):
        super().__init__()
        if graph_mode not in ('none', 'static', 'self'):
            raise ValueError('unsupported graph ablation')
        if type(output_size) is not int or output_size < 1:
            raise ValueError('positive output size required')
        self.graph_mode = graph_mode
        self.encoder = SharedSequenceEncoder(vocabulary_size, width)
        self.rule_context = nn.Sequential(nn.Linear(3 * width, width), nn.Tanh())
        self.readout = nn.Sequential(nn.Linear(2 * width + 1, width), nn.Tanh(), nn.Linear(width, output_size))
        # Registered after baseline parameters, preserving their initialization.
        if graph_mode != 'none':
            self.messages = DirectedRuleAggregation(width)

    def forward(self, features):
        sequences, loaded = features['sequences'], features['loaded_rules']
        if (len(loaded) != len(sequences['rule']) or len(loaded) != len(sequences['policy'])
                or any(type(flag) is not bool for flag in loaded)):
            raise ValueError('invalid loaded rule membership')
        query = query_context_embedding(self.encoder, features)
        # Not-loaded graph nodes cannot influence this complete candidate policy.
        rules = self.encoder([s for s, keep in zip(sequences['rule'], loaded) if keep])
        policies = self.encoder([s for s, keep in zip(sequences['policy'], loaded) if keep])
        nodes = self.rule_context(torch.cat((rules, policies, query.expand(len(rules), -1)), dim=1))
        if self.graph_mode != 'none':
            edges = features['rule_edges']
            if (features.get('rule_graph_scope') != 'native_static_template'
                    or len(edges) != len(sequences['edge'])
                    or any(len(e) != 2 or any(type(i) is not int or not 0 <= i < len(loaded) for i in e) for e in edges)):
                raise ValueError('require verified native static edges')
            local = {i: j for j, i in enumerate(i for i, flag in enumerate(loaded) if flag)}
            selected = [i for i, (s, t) in enumerate(edges) if loaded[s] and loaded[t]] if self.graph_mode == 'static' else []
            positions = self.encoder([sequences['edge'][i] for i in selected])
            nodes = self.messages(nodes, [[local[j] for j in edges[i]] for i in selected], positions)
        # Sum saturated the downstream tanh for the real 103-rule catalog. Mean
        # plus count retains the sum information without its unbounded scale.
        pooled = nodes.sum(0, keepdim=True) / max(1, len(nodes))
        count = query.new_tensor([[math.log1p(len(nodes))]])
        return self.readout(torch.cat((query, pooled, count), dim=1))[0]

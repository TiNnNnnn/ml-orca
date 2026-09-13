"""Root-bound directed messages; no label or training-loop dependencies."""
import torch
from torch import nn

class DirectedRuleAggregation(nn.Module):
    """One synchronous step, separate predecessor/successor messages and position edges."""

    def __init__(self, width, rooted=False):
        super().__init__()
        self.rooted = rooted
        self.predecessor = nn.Linear((4 if rooted else 2) * width, width)
        self.successor = nn.Linear((4 if rooted else 2) * width, width)
        self.update = nn.Linear(3 * width + 2, width)

    def forward(self, nodes, edges, positions, roots=None):
        width = self.predecessor.out_features
        node_count = nodes.shape[0] if nodes.ndim == 2 else 0
        if (nodes.ndim != 2 or nodes.shape[1] != width or not torch.isfinite(nodes).all()
                or positions.shape != (len(edges), width) or positions.dtype != nodes.dtype
                or positions.device != nodes.device or not torch.isfinite(positions).all()
                or any(len(e) != 2 or any(type(i) is not int or not 0 <= i < node_count for i in e) for e in edges)):
            raise ValueError('invalid directed graph tensors or endpoints')
        if self.rooted:
            if (not isinstance(roots, torch.Tensor) or roots.shape != (len(edges), 2, width)
                    or roots.dtype != nodes.dtype or roots.device != nodes.device or not torch.isfinite(roots).all()):
                raise ValueError('require source-target and destination-source root embeddings')
        elif roots is not None:
            raise ValueError('root features require a rooted aggregation layer')
        incoming, outgoing = torch.zeros_like(nodes), torch.zeros_like(nodes)
        indegree = nodes.new_zeros((len(nodes), 1))
        outdegree = torch.zeros_like(indegree)
        if edges:
            source, target = torch.tensor(edges, device=nodes.device, dtype=torch.long).T
            incoming_features = (nodes[source], roots[:, 0], roots[:, 1], positions) if self.rooted else (nodes[source], positions)
            outgoing_features = (nodes[target], roots[:, 1], roots[:, 0], positions) if self.rooted else (nodes[target], positions)
            incoming = incoming.index_add(0, target, torch.tanh(self.predecessor(
                torch.cat(incoming_features, 1))))
            outgoing = outgoing.index_add(0, source, torch.tanh(self.successor(
                torch.cat(outgoing_features, 1))))
            ones = nodes.new_ones((len(edges), 1))
            indegree.index_add_(0, target, ones)
            outdegree.index_add_(0, source, ones)
        # Mean plus degree retains multiplicity of position-distinct edges.
        context = torch.cat((nodes, incoming / indegree.clamp_min(1), outgoing / outdegree.clamp_min(1),
                             indegree.log1p(), outdegree.log1p()), 1)
        return nodes + torch.tanh(self.update(context))

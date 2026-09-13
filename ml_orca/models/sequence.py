"""Shared sequence/numeric encoders, independent of training targets."""
import json
import torch
from torch import nn
from torch.nn.utils.rnn import pack_padded_sequence, pad_sequence

class SharedSequenceEncoder(nn.Module):
    """Ordered IR/metadata tokens plus numeric values and explicit availability."""

    def __init__(self, vocabulary_size, width=32):
        super().__init__()
        if type(vocabulary_size) is not int or vocabulary_size < 2 or type(width) is not int or width < 1:
            raise ValueError('invalid encoder dimensions')
        self.tokens = nn.Embedding(vocabulary_size, width, padding_idx=0)
        self.numbers = nn.Linear(2, width, bias=False)
        self.sequence = nn.GRU(width, width, batch_first=True)

    def forward(self, sequences, token_contexts=None):
        if token_contexts is None and len(sequences) > 1:
            # Deterministic shared f_theta(x): encode equal immutable inputs once
            # in THIS forward, then gather every occurrence back. Autograd sums
            # all occurrence gradients; no edge/count is removed or cached across
            # optimizer steps. Per-token differentiable contexts must stay separate.
            unique, lookup, indices, identities = [], {}, [], {}
            for sequence in sequences:
                identity = id(sequence)
                if identity in identities:
                    indices.append(identities[identity])
                    continue
                key = json.dumps(sequence, sort_keys=True, allow_nan=False)
                if key not in lookup:
                    lookup[key] = len(unique)
                    unique.append(sequence)
                indices.append(lookup[key])
                identities[identity] = lookup[key]
            if len(unique) < len(sequences):
                encoded = self._encode(unique)
                return encoded.index_select(0, torch.tensor(indices, device=encoded.device))
        return self._encode(sequences, token_contexts)

    def _encode(self, sequences, token_contexts=None):
        if token_contexts is not None and len(token_contexts) != len(sequences):
            raise ValueError('one token context per sequence is required')
        if not sequences:
            return self.tokens.weight.new_zeros((0, self.tokens.embedding_dim))
        device, dtype = self.tokens.weight.device, self.tokens.weight.dtype
        ids, values, lengths = [], [], []
        for sequence in sequences:
            tokens, numbers, known = (sequence[k] for k in ('tokens', 'numbers', 'known_number'))
            if (not tokens or len(tokens) != len(numbers) or len(tokens) != len(known)
                    or any(type(t) is not int or not 0 < t < self.tokens.num_embeddings for t in tokens)
                    or any(type(mask) is not bool for mask in known)):
                raise ValueError('invalid encoded sequence')
            # Immutable Python inputs are validated/packed on CPU, then copied in
            # batches. One device tensor + synchronous check per sequence stalls CUDA.
            token_ids = torch.tensor(tokens, dtype=torch.long)
            numeric = torch.tensor(list(zip(numbers, known)), dtype=dtype)
            if any(value != 0 for value, mask in zip(numbers, known) if not mask):
                raise ValueError('numeric input must be finite with zero storage for missing values')
            ids.append(token_ids)
            values.append(numeric)
            lengths.append(len(tokens))
        numeric = pad_sequence(values, batch_first=True)
        if not torch.isfinite(numeric).all():
            raise ValueError('numeric input must be finite with zero storage for missing values')
        embedded = self.tokens(pad_sequence(ids, batch_first=True).to(device)) + self.numbers(numeric.to(device))
        if token_contexts is not None:
            if isinstance(token_contexts, torch.Tensor):
                contexts = token_contexts
                if contexts.shape != embedded.shape or contexts.device != device or contexts.dtype != dtype:
                    raise ValueError('invalid token context')
            else:
                for context, length in zip(token_contexts, lengths):
                    if (not isinstance(context, torch.Tensor) or context.shape != (length, self.tokens.embedding_dim)
                            or context.device != device or context.dtype != dtype):
                        raise ValueError('invalid token context')
                contexts = pad_sequence(token_contexts, batch_first=True)
            if not torch.isfinite(contexts).all():
                raise ValueError('invalid token context')
            embedded = embedded + contexts
        packed = pack_padded_sequence(embedded, lengths, batch_first=True, enforce_sorted=False)
        _, hidden = self.sequence(packed)
        # A direct, padding-free path for early numeric fields: final GRU state alone
        # lost catalog row-count changes at float32 precision on real long sequences.
        valid = torch.arange(embedded.shape[1], device=device)[None, :] < torch.tensor(lengths, device=device)[:, None]
        pooled = (embedded * valid.unsqueeze(-1)).sum(1) / embedded.new_tensor(lengths).unsqueeze(-1)
        return hidden[0] + pooled

def query_context_embedding(encoder, features):
    """Inject pre-run relation metadata at bound SQL references, not derived-input cardinalities."""
    binding, sequences = features['query_binding'], features['sequences']
    if (binding.get('complete') is not True or len(sequences['query']) != 1
            or 'source_references' not in binding):
        raise ValueError('require one completely bound query')
    query, sources = sequences['query'][0], binding['sources']
    relations = encoder(sequences['relation'])
    context = relations.new_zeros((len(query['tokens']), encoder.tokens.embedding_dim))
    positions, rows, seen = [], [], set()
    for reference in binding['source_references']:
        token, source = reference['token'], reference['source']
        if (type(token) is not int or not 0 <= token < len(query['tokens']) or token in seen
                or type(source) is not int or not 0 <= source < len(sources)
                or sources[source]['source'] != source or query['known_number'][token] is not True
                or query['numbers'][token] != source):
            raise ValueError('invalid query source reference')
        seen.add(token)
        record = sources[source]
        if record['kind'] == 'relation':
            relation = record['relation']
            if type(relation) is not int or not 0 <= relation < len(relations):
                raise ValueError('query relation missing from encoded catalog')
            positions.append(token)
            rows.append(relations[relation])
        elif record['kind'] != 'derived':
            raise ValueError('unsupported query source kind')
    if positions:
        context = context.index_copy(0, torch.tensor(positions, device=context.device), torch.stack(rows))
    return encoder([query], [context])

class RuleAttemptEncoder(nn.Module):
    """Shared rule × local Memo context fusion; not a trained value or policy predictor."""

    def __init__(self, vocabulary_size, width=32):
        super().__init__()
        self.encoder = SharedSequenceEncoder(vocabulary_size, width)
        self.fusion = nn.Linear(3 * width + 1, width)

    def forward(self, rules, contexts, queries=None, query_embeddings=None):
        if len(rules) != len(contexts) or (queries is not None and len(queries) != len(rules)):
            raise ValueError('one rule and local context per attempt are required')
        rule, local = self.encoder(rules), self.encoder(contexts)
        if query_embeddings is not None:
            if (queries is not None or not isinstance(query_embeddings, torch.Tensor)
                    or query_embeddings.shape != rule.shape or query_embeddings.device != rule.device
                    or query_embeddings.dtype != rule.dtype or not torch.isfinite(query_embeddings).all()):
                raise ValueError('invalid or conflicting query embeddings')
            query = query_embeddings
        else:
            query = torch.zeros_like(rule) if queries is None else self.encoder(queries)
        available = rule.new_full((len(rules), 1), float(queries is not None or query_embeddings is not None))
        return torch.tanh(self.fusion(torch.cat((rule, local, query, available), dim=1)))

#
# Copyright (c) 2020 Idiap Research Institute, http://www.idiap.ch/
# Written by Angelos Katharopoulos <angelos.katharopoulos@idiap.ch>,
# Apoorv Vyas <avyas@idiap.ch>
#

"""Implement unmasked linear attention."""

import torch
from torch.nn import Module

from ..attention_registry import AttentionRegistry, Optional, Callable, Int, \
    EventDispatcherInstance
from ..events import EventDispatcher
from ..feature_maps import elu_feature_map


class LinearAttention(Module):
    """Implement unmasked attention using dot product of feature maps in
    O(N D^2) complexity.

    Given the queries, keys and values as Q, K, V instead of computing

        V' = softmax(Q.mm(K.t()), dim=-1).mm(V),

    we make use of a feature map function Φ(.) and perform the following
    computation

        V' = normalize(Φ(Q).mm(Φ(K).t())).mm(V).

    The above can be computed in O(N D^2) complexity where D is the
    dimensionality of Q, K and V and N is the sequence length. Depending on the
    feature map, however, the complexity of the attention might be limited.

    Arguments
    ---------
        feature_map: callable, a callable that applies the feature map to the
                     last dimension of a tensor (default: elu(x)+1)
        eps: float, a small number to ensure the numerical stability of the
             denominator (default: 1e-6)
        event_dispatcher: str or EventDispatcher instance to be used by this
                          module for dispatching events (default: the default
                          global dispatcher)
    """
    def __init__(self, query_dimensions, feature_map=None, eps=1e-6,
                 event_dispatcher=""):
        super(LinearAttention, self).__init__()
        self.feature_map = (
            feature_map(query_dimensions) if feature_map else
            elu_feature_map(query_dimensions)
        )
        self.eps = eps
        self.event_dispatcher = EventDispatcher.get(event_dispatcher)

    def forward(self, queries, keys, values, attn_mask, query_lengths,
                key_lengths):
        # Apply the feature map to the queries and keys
        self.feature_map.new_feature_map(queries.device)
        Q = self.feature_map.forward_queries(queries)
        K = self.feature_map.forward_keys(keys)

        # Apply the key padding mask and make sure that the attn_mask is
        # all_ones
        if not attn_mask.all_ones:
            raise RuntimeError(("LinearAttention does not support arbitrary "
                                "attention masks"))
        K = K * key_lengths.float_matrix[:, :, None, None]

        # Compute the KV matrix, namely the dot product of keys and values so
        # that we never explicitly compute the attention matrix and thus
        # decrease the complexity
        KV = torch.einsum("nshd,nshm->nhmd", K, values)

        # Compute the normalizer
        Z = 1/(torch.einsum("nlhd,nhd->nlh", Q, K.sum(dim=1))+self.eps)

        # Finally compute and return the new values
        V = torch.einsum("nlhd,nhmd,nlh->nlhm", Q, KV, Z)

        return V.contiguous()


# Register the attention implementation so that it becomes available in our
# builders
AttentionRegistry.register(
    "linear", LinearAttention,
    [
        ("query_dimensions", Int),
        ("feature_map", Optional(Callable)),
        ("event_dispatcher", Optional(EventDispatcherInstance, ""))
    ]
)





#### added content:


# attention_similarity_linear.py
# attention_similarity_linear.py
from __future__ import annotations
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, Callable

# Factory: feature_map_factory(C) -> FeatureMap with
#   .new_feature_map(device), .forward_queries(x), .forward_keys(x)
FeatureMapFactory = Optional[Callable[[int], object]]

class AttentionSimilarityLinear(nn.Module):
    """
    Linear-time attention-as-similarity τ ∈ [0,1], using the same variable
    naming style as fast-transformers' linear_attention.py.

    Inputs:
      chosen_hidden_states   : (B, S, D)  # "queries" source
      rejected_hidden_states : (B, S, D)  # "keys" source
      chosen_mask  (optional): (B, S) {0/1 or bool}
      rejected_mask(optional): (B, S) {0/1 or bool}

    Output:
      tau: (B,) in [0,1]

    Definition (no S×S matrices):
      With φ the positive feature map (ELU+1 by default or a provided factory),
      the linearized row attention is α_ij ∝ φ(q_i)·φ(k_j).
      Row concentration ∑_j α_ij^2 ∈ [1/N,1] is computed in O(S·C^2) via:
        numerator_i   = φ(q_i)^T (Σ_j φ(k_j)φ(k_j)^T) φ(q_i)
        denominator_i = (φ(q_i)·Σ_j φ(k_j))^2
      Average over queries and map [1/N,1]→[0,1].
    """
    def __init__(
        self,
        hidden_dim: int,
        proj_dim: Optional[int] = None,
        feature_map_factory: FeatureMapFactory = None,
        eps: float = 1e-8,
    ):
        super().__init__()
        C = proj_dim or hidden_dim
        self.W_Q = nn.Linear(hidden_dim, C, bias=False)
        self.W_K = nn.Linear(hidden_dim, C, bias=False)
        self.eps = eps

        # Match original naming: self.feature_map
        # If no factory provided, fall back to φ(x)=ELU(x)+1
        self.feature_map = feature_map_factory(C) if feature_map_factory is not None else None

    def forward(
        self,
        chosen_hidden_states: torch.Tensor,    # (B, S, D)
        rejected_hidden_states: torch.Tensor,  # (B, S, D)
        chosen_mask: Optional[torch.Tensor] = None,   # (B, S)
        rejected_mask: Optional[torch.Tensor] = None, # (B, S)
    ) -> torch.Tensor:                                  # (B,)
        assert chosen_hidden_states.dim() == 3 and rejected_hidden_states.dim() == 3, \
            "Pass hidden states with shape (B, S, D), not vocab logits."

        B, S, _ = chosen_hidden_states.shape

        # Project then apply the feature map like in LinearAttention
        Q = self.W_Q(chosen_hidden_states)      # (B, S, C)
        K = self.W_K(rejected_hidden_states)    # (B, S, C)

        if self.feature_map is not None:
            self.feature_map.new_feature_map(Q.device)
            Q = self.feature_map.forward_queries(Q)  # φ(Q)
            K = self.feature_map.forward_keys(K)     # φ(K)
        else:
            # Default φ: ELU(x)+1 (positive features)
            Q = F.elu(Q) + 1.0
            K = F.elu(K) + 1.0

        # Apply masks (same spirit as linear_attention: zero out padded keys/queries)
        if chosen_mask is not None:
            if chosen_mask.dtype not in (torch.float16, torch.float32, torch.bfloat16):
                chosen_mask = chosen_mask.float()
            Q = Q * chosen_mask.unsqueeze(-1)
        if rejected_mask is not None:
            if rejected_mask.dtype not in (torch.float16, torch.float32, torch.bfloat16):
                rejected_mask = rejected_mask.float()
            K = K * rejected_mask.unsqueeze(-1)

        # ----- Linear-attention summaries (mirroring variable names) -----
        # Z normalizer uses the summed keys like in LinearAttention
        # Z = 1 / (Q · sum_j K_j + eps)  -> shape (B, S)
        K_sum = K.sum(dim=1)                                              # (B, C)
        denom_vec = torch.einsum("bsc,bc->bs", Q, K_sum)                  # (B, S)
        Z = 1.0 / (denom_vec + self.eps)                                  # (B, S)

        # KV-like accumulator but for K K^T (no values are used here)
        # KV = Σ_j K_j K_j^T  -> (B, C, C)
        KV = torch.einsum("bsc,bsd->bcd", K, K)                           # (B, C, C)

        # ----- Row concentration in linear time -----
        # numerator_i   = Q_i^T KV Q_i
        # denominator_i = (Q_i · K_sum)^2  = (1/Z_i)^2
        tmp = torch.einsum("bsc,bcd->bsd", Q, KV)                         # (B, S, C)
        numerator = (tmp * Q).sum(dim=-1).clamp_min(self.eps)             # (B, S)
        denominator = (1.0 / Z).pow(2).clamp_min(self.eps)                # (B, S)
        row_l2_sq = numerator / denominator                               # (B, S) ∈ (0,1]

        # Average over queries (respect mask if provided)
        if chosen_mask is not None:
            q_counts = chosen_mask.sum(dim=1).clamp_min(1.0)              # (B,)
            mean_l2_sq = (row_l2_sq * chosen_mask).sum(dim=1) / q_counts  # (B,)
        else:
            mean_l2_sq = row_l2_sq.mean(dim=1)                            # (B,)

        # Map [1/N, 1] → [0, 1], with N = effective #keys
        if rejected_mask is not None:
            n_keys = rejected_mask.sum(dim=1).clamp_min(1.0)              # (B,)
        else:
            n_keys = torch.full((B,), float(S), device=mean_l2_sq.device, dtype=mean_l2_sq.dtype)

        tau = (n_keys * mean_l2_sq - 1.0) / (n_keys - 1.0).clamp_min(1e-6)  # (B,)
        return tau.clamp(0.0, 1.0)

############ added 2:

# attention_similarity_linear.py

import torch
from torch.nn import Module

from ..attention_registry import AttentionRegistry, Optional, Callable, Int, \
    EventDispatcherInstance
from ..events import EventDispatcher
from ..feature_maps import elu_feature_map


class LinearAttentionSimilarity(Module):
    """Compute an attention-based similarity τ ∈ [0,1] between two sequences
    using the same linear-attention machinery and naming as LinearAttention.

    Shapes
    ------
    queries: (N, L, H, D)    # from the "chosen" sequence
    keys:    (N, S, H, D)    # from the "rejected" sequence
    values:  (ignored; kept for API compatibility)
    attn_mask: must be all-ones (same restriction as LinearAttention)
    query_lengths: LengthMask for queries (to average over valid L)
    key_lengths:   LengthMask for keys    (to mask S and compute N)

    Returns
    -------
    tau: (N,) similarity per example in [0, 1]

    Method
    ------
    We use the linear-attention feature map Φ (ELU+1 by default) and compute,
    for each query token i, the row "concentration" of its normalized attention:
        α_ij ∝ Φ(q_i)·Φ(k_j),   ∑_j α_ij = 1
        concentration_i = ∑_j α_ij^2 ∈ [1/|K|, 1]
    which can be evaluated in O(S·D^2) per head using:
        numerator_i   = Φ(q_i)^T (Σ_j Φ(k_j)Φ(k_j)^T) Φ(q_i)
        denominator_i = (Φ(q_i) · Σ_j Φ(k_j))^2
        concentration_i = numerator_i / denominator_i
    We then average over L (respecting query_lengths), map [1/|K|,1]→[0,1],
    and finally average across heads to get a single τ per example.
    """
    def __init__(self, query_dimensions, feature_map=None, eps=1e-6,
                 event_dispatcher=""):
        super(LinearAttentionSimilarity, self).__init__()
        self.feature_map = (
            feature_map(query_dimensions) if feature_map else
            elu_feature_map(query_dimensions)
        )
        self.eps = eps
        self.event_dispatcher = EventDispatcher.get(event_dispatcher)

    def forward(self, queries, keys, values, attn_mask, query_lengths,
                key_lengths):
        # Apply the feature map to the queries and keys (same as LinearAttention)
        self.feature_map.new_feature_map(queries.device)
        Q = self.feature_map.forward_queries(queries)  # (N, L, H, D)
        K = self.feature_map.forward_keys(keys)        # (N, S, H, D)

        # Enforce the same mask restriction as LinearAttention
        if not attn_mask.all_ones:
            raise RuntimeError(("LinearAttentionSimilarity does not support arbitrary "
                                "attention masks"))

        # Apply key padding mask exactly like LinearAttention
        # key_lengths.float_matrix: (N, S); broadcast to (N, S, 1, 1)
        K = K * key_lengths.float_matrix[:, :, None, None]

        # Summaries over keys (linear-time in S)
        # K_sum: Σ_j K_j          → (N, H, D)
        # KK:    Σ_j K_j K_j^T    → (N, H, D, D)
        K_sum = K.sum(dim=1)                                            # (N, H, D)
        KK = torch.einsum("nshd,nshm->nhdm", K, K)                      # (N, H, D, D)

        # Normalizer Z like in LinearAttention (but we keep it per (N, L, H))
        # Z = 1 / (Q · K_sum + eps)
        Z = 1.0 / (torch.einsum("nlhd,nhd->nlh", Q, K_sum) + self.eps)  # (N, L, H)

        # Row concentration in linear time (no SxS):
        # numerator_i   = Q_i^T KK Q_i
        # denominator_i = (Q_i · K_sum)^2 = (1/Z_i)^2
        tmp = torch.einsum("nlhd,nhdm->nlhm", Q, KK)                    # (N, L, H, D)
        numerator = (tmp * Q).sum(dim=-1).clamp_min(self.eps)           # (N, L, H)
        denominator = (1.0 / Z).pow(2).clamp_min(self.eps)              # (N, L, H)
        row_l2_sq = numerator / denominator                             # (N, L, H), in (0,1]

        # Average over valid query positions using query_lengths
        # query_lengths.float_matrix: (N, L) -> broadcast to (N, L, H)
        qmask = query_lengths.float_matrix
        row_l2_sq = row_l2_sq * qmask[:, :, None]
        q_counts = qmask.sum(dim=1).clamp_min(1.0)                       # (N,)
        mean_l2_sq = row_l2_sq.sum(dim=1) / q_counts[:, None]            # (N, H)

        # Map from [1/N_keys, 1] → [0, 1], using effective number of keys
        k_counts = key_lengths.float_matrix.sum(dim=1).clamp_min(1.0)    # (N,)
        tau_per_head = (k_counts[:, None] * mean_l2_sq - 1.0) / (k_counts[:, None] - 1.0).clamp_min(1e-6)  # (N, H)
        tau_per_head = tau_per_head.clamp(0.0, 1.0)

        # Aggregate over heads to return (N,)
        tau = tau_per_head.mean(dim=1)                                    # (N,)
        return tau.contiguous()


# Register so it mirrors the pattern of LinearAttention
AttentionRegistry.register(
    "linear_similarity", LinearAttentionSimilarity,
    [
        ("query_dimensions", Int),
        ("feature_map", Optional(Callable)),
        ("event_dispatcher", Optional(EventDispatcherInstance, ""))
    ]
)

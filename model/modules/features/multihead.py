"""Multi-head feature transformer (v5): three SEPARATE feature transformers whose
outputs are recombined by explicit cross-net product gating, instead of the v4
ComposedFeatureTransformer which sums all features into one shared accumulator.

Heads (lean default):
  0 = placement  (HalfKAv2_hm^)  width 1024 (+psqt)   -> split [self 512 | gate 512]
  1 = threats    (ThreatsV1)     width 512
  2 = enginethr  (EngineThreats) width 512

The dataloader still emits ONE global, concatenated index tensor (same as v4:
HalfKA [0,N0), ThreatsV1 [N0,N0+N1), ET [..]).  We route those global indices to
the three heads by range-split here -- so NO dataloader/ABI change is needed for
the prototype.  The actual pooling + cross-products live in NNUEModel.forward
(they need the us/them side-to-move orientation).
"""

import torch
from torch import nn

from ..feature_transformer import SparseLinearFunction
from .input_feature import InputFeature


class MultiHeadFeatureTransformer(nn.Module):
    is_multihead = True  # NNUEModel.forward branches on this

    def __init__(self, features: list[InputFeature], widths: list[int], psqt_buckets: int):
        super().__init__()
        assert len(features) >= 2, "multi-head = placement + >=1 tactical head"
        self.features = nn.ModuleList(features)
        self.widths = list(widths)          # per-head FT width EXCLUDING psqt: e.g. [1024, 512]
        self.psqt_buckets = psqt_buckets

        # PLAIN CONCAT combine: each head self-pools (pairwise product of its two halves,
        # per perspective).  The product halves the per-perspective width but the two
        # perspectives concatenate, so a head's pooled width == its FT width.  fc0 input
        # is therefore just the sum of the head widths: [1024, 512] -> 1536.
        self.l0_width = sum(self.widths)

        # one bias per head, sized to that head's num_outputs (placement carries psqt).
        self.biases = nn.ParameterList(
            [nn.Parameter(torch.empty(f.num_outputs, dtype=torch.float32)) for f in features]
        )

        # global-index offsets for routing. The C++ loader packs features by NUM_INPUTS
        # (HalfKA's factorizer space is 24576, not the coalesced 22528), in string order.
        # merged_weight() returns NUM_INPUTS rows during training, so this matches.
        self.offsets = [0]
        for f in features:
            self.offsets.append(self.offsets[-1] + f.NUM_INPUTS)

        # aggregate attributes the rest of the stack / loader expect
        self.num_outputs = features[0].num_outputs           # placement head owns psqt
        self.NUM_INPUTS = sum(f.NUM_INPUTS for f in features)
        self.NUM_REAL_FEATURES = sum(f.NUM_REAL_FEATURES for f in features)
        self.MAX_ACTIVE_FEATURES = sum(f.MAX_ACTIVE_FEATURES for f in features)
        # loader still emits all three groups concatenated with '+':
        self.INPUT_FEATURE_NAME = "+".join(f.INPUT_FEATURE_NAME for f in features)
        # '|' marks the multi-head arch in the net header / hash:
        self.FEATURE_NAME = "|".join(f.FEATURE_NAME for f in features)
        self.HASH = self._compute_hash()

        self._reset_biases()

    def _compute_hash(self) -> int:
        h = 0x5645_0005  # "VE" + v5 tag, distinct from composed
        for f in self.features:
            h = ((h << 1) | (h >> 31)) & 0xFFFFFFFF
            h ^= f.HASH
        return h

    def _reset_biases(self):
        import math
        with torch.no_grad():
            for k, f in enumerate(self.features):
                sigma = math.sqrt(1 / f.NUM_INPUTS)
                self.biases[k].uniform_(-sigma, sigma)

    @staticmethod
    def _route(indices, off, n):
        """Keep only global indices in [off, off+n); rebase to head-local, -1 elsewhere."""
        local = indices - off
        in_range = (indices >= off) & (indices < off + n)
        return torch.where(in_range, local, torch.full_like(local, -1))

    def forward(self, wi, wv, bi, bv):
        """Returns [(w0,b0),(w1,b1),(w2,b2)] -- per head, per perspective."""
        outs = []
        for k, f in enumerate(self.features):
            off, n = self.offsets[k], f.NUM_INPUTS
            w = f.merged_weight()
            wik = self._route(wi, off, n)
            bik = self._route(bi, off, n)
            ow = SparseLinearFunction.apply(wik, wv, w, self.biases[k])
            ob = SparseLinearFunction.apply(bik, bv, w, self.biases[k])
            outs.append((ow, ob))
        return outs

    # ---- delegation to heads (mirror ComposedFeatureTransformer) ----
    @torch.no_grad()
    def coalesce(self) -> None:
        for f in self.features:
            f.coalesce()

    @torch.no_grad()
    def init_weights(self, num_psqt_buckets: int, nnue2score: float) -> None:
        # placement head keeps psqt; tactical heads have none.
        self.features[0].init_weights(num_psqt_buckets, nnue2score)
        for f in self.features[1:]:
            f.init_weights(0, nnue2score)
        # zero placement bias psqt tail
        L1 = self.features[0].num_outputs - num_psqt_buckets
        for i in range(num_psqt_buckets):
            self.biases[0][L1 + i] = 0.0

    @torch.no_grad()
    def get_export_weights(self) -> torch.Tensor:
        # concat per-head export rows (engine reads three FT blocks in head order)
        return torch.cat([f.get_export_weights() for f in self.features], dim=0)

    @torch.no_grad()
    def load_export_weights(self, export_weight: torch.Tensor) -> None:
        off = 0
        for f in self.features:
            n = f.NUM_REAL_FEATURES
            f.load_export_weights(export_weight[off:off + n])
            off += n

    def clip_weights(self, quantization) -> None:
        for f in self.features:
            f.clip_weights(quantization)


def combine_multihead_features(*feature_classes: type):
    """Factory: features[0]=placement (keeps psqt), features[1:]=tactical (no psqt).
    Lean default widths: placement = L1, tactical = L1//2 each."""
    def factory(num_outputs: int) -> MultiHeadFeatureTransformer:
        psqt = 8  # == NNUEModel num_psqt_buckets
        ps_w = num_outputs - psqt          # 1024
        tac_w = ps_w // 2                  # 512
        feats = [feature_classes[0](ps_w + psqt)]
        for c in feature_classes[1:]:
            feats.append(c(tac_w))
        return MultiHeadFeatureTransformer(feats, [ps_w] + [tac_w] * (len(feats) - 1), psqt)
    return factory
